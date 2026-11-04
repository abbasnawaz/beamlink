import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from evaluate_beams import build_beam_pattern_figure, build_gain_figure
from evaluate_link import build_ber_figure, build_diagnostics_figure
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

from beamlink.array import ula_steering_vectors
from beamlink.beamforming import (
    codebook_beam_azimuths,
    dft_codebook,
    effective_channel,
    spread_over_subcarriers,
    sweep_beams,
    wideband_mrt_weights,
    wideband_power,
)
from beamlink.channel import paths_to_cfr, time_aligned_delays
from beamlink.config import (
    CHECKPOINT_FILENAME,
    SCENE_IMAGE_FILENAME,
    BeamMethod,
    CsiMode,
    Modulation,
    SimulationMode,
    SystemConfig,
    load_config,
)
from beamlink.data import ChannelPaths, Dataset, SceneMeta, load_dataset, load_meta, position_paths
from beamlink.link import LinkResult, run_link
from beamlink.linkbudget import linear_to_db, noise_variance_for_tx_power
from beamlink.predictor import load_predictor

logger = logging.getLogger("beamlink.offline")

COVERAGE_CHUNK_POSITIONS = 500
LINK_SLOTS = 4
LINK_METHODS = (BeamMethod.SWEEP, BeamMethod.MRT_WIDEBAND, BeamMethod.MRT_SUBCARRIER)


@dataclass(frozen=True, slots=True)
class SweepCoverage:
    best_beam: npt.NDArray[np.int64]
    single_antenna_snr_db: npt.NDArray[np.float32]
    sweep_snr_db: npt.NDArray[np.float32]
    mrt_snr_db: npt.NDArray[np.float32]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def _paths_slice(paths: ChannelPaths, rows: slice) -> ChannelPaths:
    return ChannelPaths(
        path_gain=paths.path_gain[rows],
        path_delay_s=paths.path_delay_s[rows],
        path_azimuth=paths.path_azimuth[rows],
        path_zenith=paths.path_zenith[rows],
        path_valid=paths.path_valid[rows],
    )


def _raw_cfr(paths: ChannelPaths, config: SystemConfig) -> npt.NDArray[np.complex64]:
    steering = ula_steering_vectors(
        paths.path_azimuth,
        paths.path_zenith,
        n_antennas=config.n_antennas,
        spacing_wavelengths=config.antenna_spacing_wavelengths,
        boresight_azimuth=config.bs_boresight_azimuth,
    )
    return paths_to_cfr(
        np.where(paths.path_valid, paths.path_gain, 0.0).astype(np.complex64),
        time_aligned_delays(paths.path_delay_s, paths.path_valid),
        steering,
        n_subcarriers=config.n_subcarriers,
        subcarrier_spacing_hz=config.subcarrier_spacing_hz,
    )


def _snr_db(wideband_power_value: npt.NDArray[np.float32], config: SystemConfig) -> npt.NDArray[np.float32]:
    noise_variance = noise_variance_for_tx_power(
        tx_power_dbm=config.tx_power_dbm,
        bandwidth_hz=config.bandwidth_hz,
        noise_figure_db=config.noise_figure_db,
    )
    return linear_to_db(wideband_power_value / config.n_subcarriers / noise_variance)


def sweep_coverage(dataset: Dataset, config: SystemConfig) -> SweepCoverage:
    codebook = dft_codebook(
        n_antennas=config.n_antennas,
        oversampling=config.codebook_oversampling,
        spacing_wavelengths=config.antenna_spacing_wavelengths,
        sector=config.codebook_sector,
    )
    parts: dict[str, list[npt.NDArray[np.generic]]] = {"best": [], "single": [], "sweep": [], "mrt": []}
    for start in range(0, dataset.positions_m.shape[0], COVERAGE_CHUNK_POSITIONS):
        cfr = _raw_cfr(_paths_slice(dataset.paths, slice(start, start + COVERAGE_CHUNK_POSITIONS)), config)
        beam_power = sweep_beams(cfr, codebook)
        mrt_weights = spread_over_subcarriers(wideband_mrt_weights(cfr), n_subcarriers=config.n_subcarriers)
        parts["best"].append(np.argmax(beam_power, axis=-1))
        parts["single"].append(np.sum(np.abs(cfr) ** 2, axis=(-2, -1)) / config.n_antennas)
        parts["sweep"].append(np.max(beam_power, axis=-1))
        parts["mrt"].append(wideband_power(effective_channel(cfr, mrt_weights)))
    return SweepCoverage(
        best_beam=np.concatenate(parts["best"]),
        single_antenna_snr_db=_snr_db(np.concatenate(parts["single"]), config),
        sweep_snr_db=_snr_db(np.concatenate(parts["sweep"]), config),
        mrt_snr_db=_snr_db(np.concatenate(parts["mrt"]), config),
    )


def _snr_cdf_panel(panel: Axes, dataset: Dataset, coverage: SweepCoverage) -> None:
    curves = (
        ("Single antenna", NEUTRAL, coverage.single_antenna_snr_db),
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
    panel.set_title("SNR distribution over the street grid")
    panel.legend(frameon=False, fontsize=8, loc="upper left")
    style_axes(panel)


def build_coverage_figure(
    dataset: Dataset,
    meta: SceneMeta,
    config: SystemConfig,
    coverage: SweepCoverage,
    data_dir: Path,
    out_path: Path,
) -> None:
    image = plt.imread(data_dir / SCENE_IMAGE_FILENAME)
    figure, panels = plt.subplots(2, 3, figsize=(20, 9.5))
    regions = np.where(dataset.is_los, 0, 1)
    map_panel(panels[0, 0], meta, image, f"LOS (blue, {int(dataset.is_los.sum())}) / NLOS (orange)")
    map_scatter(panels[0, 0], dataset, np.array(CATEGORICAL[:2])[regions])
    beam_azimuths = codebook_beam_azimuths(
        n_antennas=config.n_antennas,
        oversampling=config.codebook_oversampling,
        spacing_wavelengths=config.antenna_spacing_wavelengths,
        sector=config.codebook_sector,
        boresight_azimuth=config.bs_boresight_azimuth,
    )
    map_panel(panels[0, 1], meta, image, f"Best sweep beam (N = {config.n_antennas})")
    beam_colors = np.array(CATEGORICAL * (1 + beam_azimuths.size // len(CATEGORICAL)))[: beam_azimuths.size]
    map_scatter(panels[0, 1], dataset, beam_colors[coverage.best_beam])
    for beam_idx, (color, azimuth) in enumerate(zip(beam_colors, beam_azimuths, strict=True)):
        panels[0, 1].plot([], [], "s", color=color, label=f"{beam_idx}: {np.rad2deg(azimuth):.0f} deg")
    panels[0, 1].legend(
        frameon=False,
        fontsize=9,
        markerscale=1.6,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        title="beam: azimuth",
        title_fontsize=9,
    )
    maps = (
        (
            panels[0, 2],
            coverage.sweep_snr_db,
            f"Received SNR with sweep at {config.tx_power_dbm:.0f} dBm (dB)",
        ),
        (panels[1, 0], coverage.sweep_snr_db - coverage.single_antenna_snr_db, "Sweep beamforming gain (dB)"),
        (panels[1, 1], coverage.mrt_snr_db - coverage.sweep_snr_db, "Sweep loss vs wideband MRT (dB)"),
    )
    for panel, values, title in maps:
        map_panel(panel, meta, image, title)
        low, high = np.percentile(values, (1, 99))
        figure.colorbar(
            map_scatter(panel, dataset, values, cmap="Blues", vmin=low, vmax=high), ax=panel, shrink=0.8
        )
    _snr_cdf_panel(panels[1, 2], dataset, coverage)
    figure.tight_layout()
    save_figure(figure, out_path)


def _link_results(dataset: Dataset, config: SystemConfig, position_idx: int, seed: int) -> list[LinkResult]:
    paths = position_paths(dataset.paths, position_idx)
    return [
        run_link(
            paths,
            config,
            method=method,
            modulation=Modulation.QAM16,
            csi_mode=CsiMode.LS,
            simulation_mode=SimulationMode.FREQUENCY,
            tx_power_dbm=config.tx_power_dbm,
            seed=seed,
            n_slots=LINK_SLOTS,
        )
        for method in LINK_METHODS
    ]


def _median_position(snr_db: npt.NDArray[np.float32], mask: npt.NDArray[np.bool_]) -> int:
    candidates = np.flatnonzero(mask)
    return int(candidates[np.argsort(snr_db[candidates])[candidates.size // 2]])


def _log_link(region: str, position_idx: int, result: LinkResult) -> None:
    logger.info(
        f"link region={region} position={position_idx} method={result.method} snr_db={result.snr_db:.1f} "
        f"bf_gain_db={result.beamforming_gain_db:.2f} ber={result.ber:.2e} "
        f"spectral_efficiency={result.spectral_efficiency:.2f} power_loss_db={result.power_loss_db:.2f} "
        f"beam_idx={result.beam_idx} beams_tested={result.beams_tested}"
    )


def _link_row(
    panels: npt.NDArray[np.object_],
    dataset: Dataset,
    config: SystemConfig,
    region: str,
    position_idx: int,
    seed: int,
) -> None:
    results = _link_results(dataset, config, position_idx, seed)
    location = dataset.positions_m[position_idx]
    visible = results[0].pdp > 0
    delay_ns, pdp_db = results[0].pdp_delay_s[visible] * 1e9, linear_to_db(results[0].pdp[visible])
    panels[0].vlines(delay_ns, pdp_db.min() - 5, pdp_db, color=CATEGORICAL[0], lw=1.5)
    panels[0].plot(delay_ns, pdp_db, "o", color=CATEGORICAL[0], ms=5)
    panels[0].set_xlabel("Delay after first arrival (ns)")
    panels[0].set_ylabel("Path gain (dB)")
    panels[0].set_title(f"{region} position #{position_idx} at ({location[0]:.1f}, {location[1]:.1f}) m: PDP")
    frequency_megahertz = (
        (np.arange(config.n_subcarriers) - config.n_subcarriers // 2) * config.subcarrier_spacing_hz / 1e6
    )
    antenna_cfr = _raw_cfr(position_paths(dataset.paths, position_idx), config)[0]
    panels[1].plot(
        frequency_megahertz,
        linear_to_db(np.abs(antenna_cfr) ** 2),
        color=NEUTRAL,
        lw=1.5,
        label="Antenna 0 alone",
    )
    for result in results:
        panels[1].plot(
            frequency_megahertz,
            linear_to_db(np.abs(result.h_eff_true) ** 2),
            color=METHOD_COLORS[result.method],
            lw=2,
            label=METHOD_LABELS[result.method],
        )
        _log_link(region, position_idx, result)
    panels[1].set_xlabel("Baseband frequency (MHz)")
    panels[1].set_ylabel("|channel| (dB, includes path loss)")
    panels[1].set_title(f"{region}: single antenna vs beamformed h_eff (N = {config.n_antennas})")
    panels[1].legend(frameon=False, fontsize=8)
    for panel in panels:
        style_axes(panel)


def build_dataset_links_figure(
    dataset: Dataset, config: SystemConfig, coverage: SweepCoverage, seed: int, out_path: Path
) -> None:
    picks = {
        "LOS": _median_position(coverage.sweep_snr_db, dataset.is_los),
        "NLOS": _median_position(coverage.sweep_snr_db, ~dataset.is_los),
    }
    figure, panels = plt.subplots(2, 2, figsize=(14, 9))
    for row, (region, position_idx) in enumerate(picks.items()):
        _link_row(panels[row], dataset, config, region, position_idx, seed)
    figure.tight_layout()
    save_figure(figure, out_path)


def main() -> None:
    args = parse_args()
    matplotlib.use("Agg")
    config = load_config(args.config)
    rng = np.random.default_rng(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    build_ber_figure(config, rng, args.seed, args.out_dir / "ber_synthetic.png")
    build_diagnostics_figure(config, rng, args.seed, args.out_dir / "ofdm_diagnostics.png")
    build_beam_pattern_figure(config, args.out_dir / "beam_patterns.png")
    build_gain_figure(config, rng, args.out_dir / "beamforming_gain.png")
    dataset, meta = load_dataset(args.data_dir), load_meta(args.data_dir)
    coverage = sweep_coverage(dataset, config)
    for region, mask in (("LOS", dataset.is_los), ("NLOS", ~dataset.is_los)):
        logger.info(
            f"coverage region={region} positions={int(mask.sum())} "
            f"median_single_snr_db={np.median(coverage.single_antenna_snr_db[mask]):.1f} "
            f"median_sweep_snr_db={np.median(coverage.sweep_snr_db[mask]):.1f} "
            f"median_mrt_snr_db={np.median(coverage.mrt_snr_db[mask]):.1f}"
        )
    build_coverage_figure(dataset, meta, config, coverage, args.data_dir, args.out_dir / "coverage_sweep.png")
    build_dataset_links_figure(dataset, config, coverage, args.seed, args.out_dir / "dataset_links.png")
    predictor = load_predictor(args.data_dir)
    build_prediction_figure(
        dataset, args.data_dir / CHECKPOINT_FILENAME, args.out_dir / "beam_prediction.png"
    )
    build_ai_coverage_figure(
        dataset, meta, config, predictor, args.data_dir, args.out_dir / "coverage_ai.png"
    )
    build_position_ber_figure(dataset, config, predictor, args.seed, args.out_dir / "ber_positions.png")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    main()
