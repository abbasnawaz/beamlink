import argparse
import importlib
import logging
from dataclasses import replace
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import sionna.rt as rt

from beamlink.channel import paths_to_cfr, subcarrier_frequencies_hz
from beamlink.config import SIONNA_CFR_TOLERANCE, SIONNA_TILTED_CFR_TOLERANCE, SystemConfig, load_config
from beamlink.data import ChannelPaths, Dataset, load_dataset
from beamlink.link import array_response
from beamlink.linkbudget import linear_to_db

generate = importlib.import_module("01_generate_rt_dataset")
logger = logging.getLogger("beamlink.offline")

SIONNA_COLOR = "#2a78d6"
NUMPY_COLOR = "#eb6834"
TRUNCATED_COLOR = "#1baf7a"
NEUTRAL = "#52514e"
SAVE_DOTS_PER_INCH = 130


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--n-positions", type=int, default=10)
    parser.add_argument("--n-antennas", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path("results/sionna_validation.png"))
    return parser.parse_args()


def pick_positions(dataset: Dataset, n_positions: int, rng: np.random.Generator) -> npt.NDArray[np.int64]:
    los_idx = np.flatnonzero(dataset.is_los)
    nlos_idx = np.flatnonzero(~dataset.is_los)
    n_los = n_positions // 2
    chosen = np.concatenate(
        [rng.choice(los_idx, n_los, replace=False), rng.choice(nlos_idx, n_positions - n_los, replace=False)]
    )
    return chosen.astype(np.int64)


def sionna_cfr_at(
    scene: rt.Scene, position_m: npt.NDArray[np.float64], config: SystemConfig, seed: int
) -> npt.NDArray[np.complex128]:
    name = f"{generate.UE_NAME_PREFIX}0"
    scene.add(rt.Receiver(name, position=generate.point3f(position_m)))
    paths = generate.solve_paths(scene, config, seed)
    scene.remove(name)
    frequencies = rt.subcarrier_frequencies(config.n_subcarriers, config.subcarrier_spacing_hz)
    cfr = paths.cfr(frequencies=frequencies, normalize_delays=False, out_type="numpy")
    return np.asarray(cfr)[0, 0, 0, :, 0, :]


def all_single_antenna_paths(
    scene: rt.Scene, positions_m: npt.NDArray[np.float64], config: SystemConfig, seed: int
) -> ChannelPaths:
    traces = [generate.trace_positions(scene, point[np.newaxis], config, seed) for point in positions_m]
    width = max(trace.paths.path_valid.shape[1] for trace in traces)
    return generate.concatenate_traces(traces, width).paths


def sionna_sector_cfr(
    positions_m: npt.NDArray[np.float64], config: SystemConfig, seed: int
) -> npt.NDArray[np.complex128]:
    sectors = []
    for boresight_azimuth in config.sector_boresight_azimuths:
        scene = generate.load_street_scene(
            config,
            n_bs_antennas=config.n_antennas,
            pattern="tr38901",
            orientation=(boresight_azimuth, config.sector_downtilt, 0.0),
        )
        sectors.append(np.stack([sionna_cfr_at(scene, point, config, seed) for point in positions_m]))
    return np.concatenate(sectors, axis=-2)


def numpy_cfr(paths: ChannelPaths, config: SystemConfig) -> npt.NDArray[np.complex64]:
    gain = np.where(paths.path_valid, paths.path_gain, 0.0).astype(np.complex64)
    return paths_to_cfr(
        gain,
        paths.path_delay_s,
        array_response(paths.path_azimuth, paths.path_zenith, config),
        n_subcarriers=config.n_subcarriers,
        subcarrier_spacing_hz=config.subcarrier_spacing_hz,
    )


def strongest_sector_rows(cfr: npt.NDArray[np.complex128], n_antennas: int) -> slice:
    sector_power = np.sum(np.abs(cfr) ** 2, axis=-1).reshape(-1, n_antennas).sum(axis=-1)
    sector_idx = int(np.argmax(sector_power))
    return slice(sector_idx * n_antennas, (sector_idx + 1) * n_antennas)


def cfr_tolerance(config: SystemConfig) -> float:
    return SIONNA_TILTED_CFR_TOLERANCE if config.sector_downtilt != 0.0 else SIONNA_CFR_TOLERANCE


def relative_error(
    reference: npt.NDArray[np.complex128], candidate: npt.NDArray[np.complex64]
) -> npt.NDArray[np.float64]:
    difference = np.linalg.norm(reference - candidate, axis=(-2, -1))
    return difference / np.linalg.norm(reference, axis=(-2, -1))


def _magnitude_panel(
    panel: plt.Axes,
    frequency_megahertz: npt.NDArray[np.float64],
    responses: tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex64], npt.NDArray[np.complex64]],
    title: str,
    max_paths: int,
    n_antennas: int,
) -> None:
    sionna, all_paths, stored = responses
    rows = np.arange(sionna.shape[0])[strongest_sector_rows(sionna, n_antennas)]
    for antenna_idx, width in ((rows[0], 2.5), (rows[-1], 1.5)):
        label_suffix = f"antenna {antenna_idx}"
        panel.plot(
            frequency_megahertz,
            linear_to_db(np.abs(sionna[antenna_idx]) ** 2),
            color=SIONNA_COLOR,
            lw=width,
            label=f"Sionna RT, {label_suffix}",
        )
        panel.plot(
            frequency_megahertz,
            linear_to_db(np.abs(all_paths[antenna_idx]) ** 2),
            "--",
            color=NUMPY_COLOR,
            lw=width,
            label=f"NumPy, all paths, {label_suffix}",
        )
    panel.plot(
        frequency_megahertz,
        linear_to_db(np.abs(stored[rows[0]]) ** 2),
        ":",
        color=TRUNCATED_COLOR,
        lw=2,
        label=f"NumPy, stored {max_paths} strongest, antenna {rows[0]}",
    )
    panel.set_xlabel("Baseband frequency (MHz)")
    panel.set_ylabel("|H| (dB)")
    panel.set_title(title)
    panel.legend(frameon=False, fontsize=8)


def build_validation_figure(
    sionna: npt.NDArray[np.complex128],
    all_paths: npt.NDArray[np.complex64],
    stored: npt.NDArray[np.complex64],
    is_los: npt.NDArray[np.bool_],
    config: SystemConfig,
    out_path: Path,
) -> None:
    frequency_megahertz = (
        subcarrier_frequencies_hz(
            n_subcarriers=config.n_subcarriers, subcarrier_spacing_hz=config.subcarrier_spacing_hz
        )
        / 1e6
    )
    figure, panels = plt.subplots(2, 2, figsize=(14, 9.5))
    los_pick, nlos_pick = int(np.flatnonzero(is_los)[0]), int(np.flatnonzero(~is_los)[0])
    for panel, pick, kind in ((panels[0, 0], los_pick, "LOS"), (panels[0, 1], nlos_pick, "NLOS")):
        responses = (sionna[pick], all_paths[pick], stored[pick])
        _magnitude_panel(
            panel,
            frequency_megahertz,
            responses,
            f"{kind} position: |H| of the strongest sector",
            config.max_paths,
            config.n_antennas,
        )
    serving_rows = strongest_sector_rows(sionna[nlos_pick], config.n_antennas)
    phase_difference = np.angle(all_paths[nlos_pick, serving_rows] * np.conj(sionna[nlos_pick, serving_rows]))
    panels[1, 0].plot(frequency_megahertz, np.rad2deg(phase_difference.T), color=NUMPY_COLOR, lw=1, alpha=0.6)
    panels[1, 0].set_xlabel("Baseband frequency (MHz)")
    panels[1, 0].set_ylabel("Phase(NumPy) - phase(Sionna) (deg)")
    panels[1, 0].set_title(
        f"NLOS position: phase difference, {config.n_antennas} antennas of the serving sector"
    )
    convention = relative_error(sionna, all_paths)
    truncation = relative_error(sionna, stored)
    position_axis = np.arange(convention.size)
    panels[1, 1].bar(
        position_axis - 0.2, convention, width=0.4, color=NUMPY_COLOR, label="all paths (convention check)"
    )
    panels[1, 1].bar(
        position_axis + 0.2,
        truncation,
        width=0.4,
        color=TRUNCATED_COLOR,
        label=f"stored {config.max_paths} strongest",
    )
    panels[1, 1].axhline(cfr_tolerance(config), color=NEUTRAL, ls="--", lw=1.5, label="tolerance")
    panels[1, 1].set_yscale("log")
    panels[1, 1].set_xticks(position_axis, ["LOS" if los else "NLOS" for los in is_los], fontsize=8)
    panels[1, 1].set_ylabel("Relative CFR error ||H_numpy - H_sionna|| / ||H_sionna||")
    panels[1, 1].set_title("Relative error per validated position")
    panels[1, 1].legend(frameon=False, fontsize=8)
    for panel in panels.flat:
        panel.grid(True, which="both", color="#e4e3df", linewidth=0.6)
    figure.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=SAVE_DOTS_PER_INCH, bbox_inches="tight")
    plt.close(figure)
    logger.info(f"wrote {out_path}")


def validate_tilt(
    config: SystemConfig,
    dataset: Dataset,
    chosen: npt.NDArray[np.int64],
    traced_paths: ChannelPaths,
    seed: int,
) -> tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex64], npt.NDArray[np.complex64], bool]:
    positions_m = dataset.positions_m[chosen].astype(np.float64)
    reference = sionna_sector_cfr(positions_m, config, seed)
    all_paths = numpy_cfr(traced_paths, config)
    stored_paths = ChannelPaths(
        path_gain=dataset.paths.path_gain[chosen],
        path_delay_s=dataset.paths.path_delay_s[chosen],
        path_azimuth=dataset.paths.path_azimuth[chosen],
        path_zenith=dataset.paths.path_zenith[chosen],
        path_valid=dataset.paths.path_valid[chosen],
    )
    stored = numpy_cfr(stored_paths, config)
    convention, truncation = relative_error(reference, all_paths), relative_error(reference, stored)
    total_paths = traced_paths.path_valid.sum(axis=-1)
    downtilt_deg = np.rad2deg(config.sector_downtilt)
    for row, position_idx in enumerate(chosen):
        serving_rows = strongest_sector_rows(reference[row], config.n_antennas)
        serving_error = relative_error(reference[row, serving_rows], all_paths[row, serving_rows])
        logger.info(
            f"downtilt_deg={downtilt_deg:.1f} position={position_idx} "
            f"los={bool(dataset.is_los[position_idx])} paths_total={total_paths[row]} "
            f"error_all_paths={convention[row]:.2e} "
            f"error_serving_sector={serving_error:.2e} error_stored={truncation[row]:.2e}"
        )
    tolerance = cfr_tolerance(config)
    logger.info(
        f"n_antennas={config.n_antennas} n_sectors={config.n_sectors} downtilt_deg={downtilt_deg:.1f} "
        f"max_error_all_paths={convention.max():.2e} "
        f"max_error_stored_{config.max_paths}={truncation.max():.2e} tolerance={tolerance:.0e}"
    )
    return reference, all_paths, stored, bool(convention.max() <= tolerance)


def main() -> None:
    args = parse_args()
    matplotlib.use("Agg")
    config = load_config(args.config)
    config = replace(config, n_antennas=args.n_antennas or config.n_antennas)
    rng = np.random.default_rng(args.seed)
    dataset = load_dataset(args.data_dir)
    chosen = pick_positions(dataset, args.n_positions, rng)
    positions_m = dataset.positions_m[chosen].astype(np.float64)
    single_antenna_scene = generate.load_street_scene(config, n_bs_antennas=1)
    traced_paths = all_single_antenna_paths(single_antenna_scene, positions_m, config, args.seed)
    untilted = validate_tilt(replace(config, sector_downtilt=0.0), dataset, chosen, traced_paths, args.seed)
    passed = untilted[-1]
    reference, all_paths, stored = untilted[:3]
    if config.sector_downtilt != 0.0:
        reference, all_paths, stored, tilted_passed = validate_tilt(
            config, dataset, chosen, traced_paths, args.seed
        )
        passed = passed and tilted_passed
    build_validation_figure(reference, all_paths, stored, dataset.is_los[chosen], config, args.out)
    if not passed:
        logger.error("NumPy CFR does not match Sionna RT within tolerance")
        raise SystemExit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    main()
