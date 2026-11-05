import argparse
import logging
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from evaluate_beams import build_beam_pattern_figure, build_gain_figure
from evaluate_link import build_ber_figure, build_dataset_links_figure, build_diagnostics_figure, link_results
from evaluate_predictor import build_ai_coverage_figure, build_position_ber_figure, build_prediction_figure
from evaluate_style import (
    CATEGORICAL,
    METHOD_COLORS,
    METHOD_LABELS,
    NEUTRAL,
    map_panel,
    map_scatter,
    save_figure,
    style_axes,
)
from matplotlib.axes import Axes

from beamlink.beamforming import (
    effective_channel,
    spread_over_subcarriers,
    sweep_beams,
    wideband_mrt_weights,
    wideband_power,
)
from beamlink.channel import paths_to_cfr, time_aligned_delays
from beamlink.config import (
    CHECKPOINT_FILENAME,
    MIN_SERVICE_SNR_DB,
    RT_TRACE_YAW,
    SCENE_IMAGE_FILENAME,
    BeamMethod,
    SystemConfig,
    load_config,
    single_isotropic_panel,
)
from beamlink.data import (
    ChannelPaths,
    Dataset,
    SceneMeta,
    load_dataset,
    load_meta,
    nearest_position_idx,
)
from beamlink.link import array_response, system_codebook
from beamlink.linkbudget import linear_to_db, noise_variance_for_tx_power
from beamlink.predictor import load_predictor

logger = logging.getLogger("beamlink.offline")

COVERAGE_CHUNK_POSITIONS = 500
WORSE_THRESHOLD_DB = 1.0
REFERENCE_POSITION_XY_M = (30.0, 0.0)
EXPERIMENT_DROPPED_SECTOR = 3
SNR_LOW_PERCENTILE = 5.0
V1_CODEBOOK_OVERSAMPLING = 1


@dataclass(frozen=True, slots=True)
class SweepCoverage:
    best_beam: npt.NDArray[np.int64]
    isotropic_snr_db: npt.NDArray[np.float32]
    sweep_snr_db: npt.NDArray[np.float32]
    mrt_snr_db: npt.NDArray[np.float32]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--sector-experiment", type=Path, default=None)
    return parser.parse_args()


def _paths_slice(paths: ChannelPaths, rows: slice) -> ChannelPaths:
    return ChannelPaths(
        path_gain=paths.path_gain[rows],
        path_delay_s=paths.path_delay_s[rows],
        path_azimuth=paths.path_azimuth[rows],
        path_zenith=paths.path_zenith[rows],
        path_valid=paths.path_valid[rows],
    )


def _cfr_and_reference(
    paths: ChannelPaths, config: SystemConfig
) -> tuple[npt.NDArray[np.complex64], npt.NDArray[np.complex64]]:
    gain = np.where(paths.path_valid, paths.path_gain, 0.0).astype(np.complex64)
    delay_s = time_aligned_delays(paths.path_delay_s, paths.path_valid)
    grid = {"n_subcarriers": config.n_subcarriers, "subcarrier_spacing_hz": config.subcarrier_spacing_hz}
    cfr = paths_to_cfr(gain, delay_s, array_response(paths.path_azimuth, paths.path_zenith, config), **grid)
    isotropic = paths_to_cfr(gain, delay_s, np.ones((1, gain.shape[-1]), dtype=np.complex64), **grid)
    return cfr, isotropic


def _snr_db(wideband_power_value: npt.NDArray[np.float32], config: SystemConfig) -> npt.NDArray[np.float32]:
    noise_variance = noise_variance_for_tx_power(
        tx_power_dbm=config.tx_power_dbm,
        bandwidth_hz=config.bandwidth_hz,
        noise_figure_db=config.noise_figure_db,
    )
    return linear_to_db(wideband_power_value / config.n_subcarriers / noise_variance)


def sweep_coverage(dataset: Dataset, config: SystemConfig) -> SweepCoverage:
    codebook = system_codebook(config)
    parts: dict[str, list[npt.NDArray[np.generic]]] = {"best": [], "isotropic": [], "sweep": [], "mrt": []}
    for start in range(0, dataset.positions_m.shape[0], COVERAGE_CHUNK_POSITIONS):
        rows = slice(start, start + COVERAGE_CHUNK_POSITIONS)
        cfr, isotropic = _cfr_and_reference(_paths_slice(dataset.paths, rows), config)
        beam_power = sweep_beams(cfr, codebook)
        mrt_weights = spread_over_subcarriers(wideband_mrt_weights(cfr), n_subcarriers=config.n_subcarriers)
        parts["best"].append(np.argmax(beam_power, axis=-1))
        parts["isotropic"].append(np.sum(np.abs(isotropic) ** 2, axis=(-2, -1)))
        parts["sweep"].append(np.max(beam_power, axis=-1))
        parts["mrt"].append(wideband_power(effective_channel(cfr, mrt_weights)))
    return SweepCoverage(
        best_beam=np.concatenate(parts["best"]),
        isotropic_snr_db=_snr_db(np.concatenate(parts["isotropic"]), config),
        sweep_snr_db=_snr_db(np.concatenate(parts["sweep"]), config),
        mrt_snr_db=_snr_db(np.concatenate(parts["mrt"]), config),
    )


def single_ula_config(config: SystemConfig) -> SystemConfig:
    v1_codebook = replace(config, codebook_oversampling=V1_CODEBOOK_OVERSAMPLING)
    return single_isotropic_panel(v1_codebook, boresight_azimuth=RT_TRACE_YAW)


def log_sector_comparison(
    dataset: Dataset, config: SystemConfig, sectors: SweepCoverage, single_ula: SweepCoverage, seed: int
) -> None:
    gain_db = sectors.sweep_snr_db - single_ula.sweep_snr_db
    worse = gain_db < -WORSE_THRESHOLD_DB
    first_beam_share = float(np.mean(single_ula.best_beam == 0))
    for name, coverage in (("single_ula", single_ula), ("four_sectors", sectors)):
        logger.info(
            f"sectors config={name} median_los_snr_db={np.median(coverage.sweep_snr_db[dataset.is_los]):.1f} "
            f"median_nlos_snr_db={np.median(coverage.sweep_snr_db[~dataset.is_los]):.1f} "
            f"no_service_pct={100 * np.mean(coverage.sweep_snr_db < MIN_SERVICE_SNR_DB):.1f} "
            f"top_beam_share_pct={100 * np.bincount(coverage.best_beam).max() / coverage.best_beam.size:.1f}"
        )
    distance_m = np.linalg.norm(dataset.positions_m[:, :2] - np.asarray(config.bs_position_xy_m), axis=-1)
    logger.info(
        f"sectors single_ula_beam0_share_pct={100 * first_beam_share:.1f} "
        f"median_gain_db={np.median(gain_db):.1f} worse_than_1db_pct={100 * worse.mean():.1f} "
        f"worse_median_distance_m={np.median(distance_m[worse]):.0f} "
        f"worse_median_new_snr_db={np.median(sectors.sweep_snr_db[worse]):.1f} "
        f"lost_service={int(np.sum((sectors.sweep_snr_db < 0) & (single_ula.sweep_snr_db >= 0)))} "
        f"gained_service={int(np.sum((sectors.sweep_snr_db >= 0) & (single_ula.sweep_snr_db < 0)))}"
    )
    position_idx = nearest_position_idx(dataset.positions_m, *REFERENCE_POSITION_XY_M)
    for name, system in (("single_ula", single_ula_config(config)), ("four_sectors", config)):
        result = link_results(dataset, system, position_idx, seed)[0]
        logger.info(
            f"sectors config={name} position={position_idx} sweep_gain_db={result.beamforming_gain_db:.2f} "
            f"nominal_gain_db={float(linear_to_db(system.n_antennas)) + system.element_peak_gain_dbi:.2f} "
            f"snr_db={result.snr_db:.1f} beam_idx={result.beam_idx} sector={result.serving_sector}"
        )


def _snr_cdf_panel(panel: Axes, dataset: Dataset, coverage: SweepCoverage, single_ula: SweepCoverage) -> None:
    curves = (
        ("One isotropic antenna", NEUTRAL, coverage.isotropic_snr_db),
        ("v1 single ULA, DFT sweep", CATEGORICAL[4], single_ula.sweep_snr_db),
        (METHOD_LABELS[BeamMethod.SWEEP], METHOD_COLORS[BeamMethod.SWEEP], coverage.sweep_snr_db),
        (METHOD_LABELS[BeamMethod.MRT_WIDEBAND], METHOD_COLORS[BeamMethod.MRT_WIDEBAND], coverage.mrt_snr_db),
    )
    for label, color, snr_db in curves:
        for region, mask, dash in (("LOS", dataset.is_los, "-"), ("NLOS", ~dataset.is_los, "--")):
            sorted_snr = np.sort(snr_db[mask])
            fraction = np.arange(1, sorted_snr.size + 1) / sorted_snr.size
            panel.plot(sorted_snr, fraction, color=color, ls=dash, lw=2, label=f"{label}, {region}")
    panel.set_xlabel("Received SNR (dB)")
    panel.set_ylabel("Fraction of positions")
    panel.set_title("SNR distribution over the street grid (four sectors unless noted)")
    panel.legend(frameon=False, fontsize=8, loc="upper left")
    style_axes(panel)


def build_coverage_figure(
    dataset: Dataset,
    meta: SceneMeta,
    config: SystemConfig,
    coverages: tuple[SweepCoverage, SweepCoverage],
    data_dir: Path,
    out_path: Path,
) -> None:
    coverage, single_ula = coverages
    image = plt.imread(data_dir / SCENE_IMAGE_FILENAME)
    figure, panels = plt.subplots(2, 3, figsize=(20, 9.5))
    regions = np.where(dataset.is_los, 0, 1)
    map_panel(panels[0, 0], meta, image, f"LOS (blue, {int(dataset.is_los.sum())}) / NLOS (orange)")
    map_scatter(panels[0, 0], dataset, np.array(CATEGORICAL[:2])[regions])
    map_panel(panels[0, 1], meta, image, f"Serving sector of the best sweep beam (N = {config.n_antennas})")
    sector_colors = np.array(CATEGORICAL[: config.n_sectors])
    map_scatter(panels[0, 1], dataset, sector_colors[coverage.best_beam // config.n_beams_per_sector])
    for sector_idx, (color, azimuth) in enumerate(
        zip(sector_colors, config.sector_boresight_azimuths, strict=True)
    ):
        panels[0, 1].plot(
            [], [], "s", color=color, label=f"sector {sector_idx}: {np.rad2deg(azimuth):.0f} deg"
        )
    panels[0, 1].legend(
        frameon=False,
        fontsize=9,
        markerscale=1.6,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        title="boresight",
        title_fontsize=9,
    )
    maps = (
        (
            panels[0, 2],
            coverage.sweep_snr_db,
            f"Received SNR with sweep at {config.tx_power_dbm:.0f} dBm (dB)",
            "Blues",
        ),
        (
            panels[1, 0],
            coverage.sweep_snr_db - coverage.isotropic_snr_db,
            "Sweep gain over one isotropic antenna (dB)",
            "Blues",
        ),
        (
            panels[1, 1],
            coverage.sweep_snr_db - single_ula.sweep_snr_db,
            "SNR change vs the v1 single ULA (dB)",
            "RdBu",
        ),
    )
    for panel, values, title, colormap in maps:
        map_panel(panel, meta, image, title)
        low, high = np.percentile(values, (1, 99))
        if colormap == "RdBu":
            low, high = -max(abs(low), abs(high)), max(abs(low), abs(high))
        figure.colorbar(
            map_scatter(panel, dataset, values, cmap=colormap, vmin=low, vmax=high), ax=panel, shrink=0.8
        )
    _snr_cdf_panel(panels[1, 2], dataset, coverage, single_ula)
    figure.tight_layout()
    save_figure(figure, out_path)


def _log_configuration(name: str, system: SystemConfig, coverage: SweepCoverage) -> None:
    sectors = coverage.best_beam // system.n_beams_per_sector
    shares = np.bincount(sectors, minlength=system.n_sectors) / sectors.size
    logger.info(
        f"experiment config={name} sweep_beams={system.n_beams} "
        f"sector_share_pct={[round(100 * float(share), 1) for share in shares]} "
        f"no_service_pct={100 * np.mean(coverage.sweep_snr_db < MIN_SERVICE_SNR_DB):.2f} "
        f"median_snr_db={np.median(coverage.sweep_snr_db):.2f} "
        f"p{SNR_LOW_PERCENTILE:.0f}_snr_db={np.percentile(coverage.sweep_snr_db, SNR_LOW_PERCENTILE):.2f}"
    )


def sector_experiment(
    dataset: Dataset,
    meta: SceneMeta,
    config: SystemConfig,
    coverages: tuple[SweepCoverage, SweepCoverage],
    data_dir: Path,
    out_path: Path,
) -> None:
    coverage, single_ula = coverages
    kept = tuple(
        azimuth
        for sector_idx, azimuth in enumerate(config.sector_boresight_azimuths)
        if sector_idx != EXPERIMENT_DROPPED_SECTOR
    )
    reduced = replace(config, sector_boresight_azimuths=kept)
    reduced_coverage = sweep_coverage(dataset, reduced)
    _log_configuration("four_sectors", config, coverage)
    _log_configuration("three_sectors", reduced, reduced_coverage)
    served = coverage.best_beam // config.n_beams_per_sector == EXPERIMENT_DROPPED_SECTOR
    loss_db = coverage.sweep_snr_db - reduced_coverage.sweep_snr_db
    versus_ula_db = reduced_coverage.sweep_snr_db - single_ula.sweep_snr_db
    newly_unserved = (reduced_coverage.sweep_snr_db < MIN_SERVICE_SNR_DB) & (
        coverage.sweep_snr_db >= MIN_SERVICE_SNR_DB
    )
    lost_service = served & newly_unserved
    logger.info(
        f"experiment dropped_sector={EXPERIMENT_DROPPED_SECTOR} served_positions={int(served.sum())} "
        f"loss_mean_db={loss_db[served].mean():.2f} loss_p95_db={np.percentile(loss_db[served], 95):.2f} "
        f"loss_max_db={loss_db[served].max():.2f} served_los={int((served & dataset.is_los).sum())} "
        f"served_snr_median_db={np.median(coverage.sweep_snr_db[served]):.1f} "
        f"lost_service={int(lost_service.sum())} "
        f"newly_unserved_total={int(np.sum(newly_unserved))} "
        f"three_sectors_worse_than_v1_pct={100 * np.mean(versus_ula_db < -WORSE_THRESHOLD_DB):.2f}"
    )
    image = plt.imread(data_dir / SCENE_IMAGE_FILENAME)
    figure, panel = plt.subplots(figsize=(9, 6))
    map_panel(
        panel, meta, image, f"SNR lost without sector {EXPERIMENT_DROPPED_SECTOR} by the users it serves (dB)"
    )
    map_scatter(panel, dataset, np.full(served.size, "#e4e3df"))
    points = dataset.positions_m[served]
    scatter = panel.scatter(
        points[:, 0], points[:, 1], c=loss_db[served], s=12, marker="s", lw=0, cmap="Oranges", vmin=0.0
    )
    figure.colorbar(scatter, ax=panel, shrink=0.8)
    figure.tight_layout()
    save_figure(figure, out_path)


def main() -> None:
    args = parse_args()
    matplotlib.use("Agg")
    config = load_config(args.config)
    rng = np.random.default_rng(args.seed)
    if args.sector_experiment is not None:
        dataset, meta = load_dataset(args.data_dir), load_meta(args.data_dir)
        coverages = (sweep_coverage(dataset, config), sweep_coverage(dataset, single_ula_config(config)))
        sector_experiment(dataset, meta, config, coverages, args.data_dir, args.sector_experiment)
        return
    args.out_dir.mkdir(parents=True, exist_ok=True)
    build_ber_figure(config, rng, args.seed, args.out_dir / "ber_synthetic.png")
    build_diagnostics_figure(config, rng, args.seed, args.out_dir / "ofdm_diagnostics.png")
    build_beam_pattern_figure(config, args.out_dir / "beam_patterns.png")
    build_gain_figure(config, rng, args.out_dir / "beamforming_gain.png")
    dataset, meta = load_dataset(args.data_dir), load_meta(args.data_dir)
    coverage = sweep_coverage(dataset, config)
    single_ula = sweep_coverage(dataset, single_ula_config(config))
    for region, mask in (("LOS", dataset.is_los), ("NLOS", ~dataset.is_los)):
        logger.info(
            f"coverage region={region} positions={int(mask.sum())} "
            f"median_isotropic_snr_db={np.median(coverage.isotropic_snr_db[mask]):.1f} "
            f"median_single_ula_sweep_snr_db={np.median(single_ula.sweep_snr_db[mask]):.1f} "
            f"median_sweep_snr_db={np.median(coverage.sweep_snr_db[mask]):.1f} "
            f"median_mrt_snr_db={np.median(coverage.mrt_snr_db[mask]):.1f}"
        )
    log_sector_comparison(dataset, config, coverage, single_ula, args.seed)
    build_coverage_figure(
        dataset, meta, config, (coverage, single_ula), args.data_dir, args.out_dir / "coverage_sweep.png"
    )
    build_dataset_links_figure(
        dataset, config, coverage.sweep_snr_db, args.seed, args.out_dir / "dataset_links.png"
    )
    predictor = load_predictor(args.data_dir)
    build_prediction_figure(
        dataset, config, args.data_dir / CHECKPOINT_FILENAME, args.out_dir / "beam_prediction.png"
    )
    build_ai_coverage_figure(
        dataset, meta, config, predictor, args.data_dir, args.out_dir / "coverage_ai.png"
    )
    build_position_ber_figure(dataset, config, predictor, args.seed, args.out_dir / "ber_positions.png")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    main()
