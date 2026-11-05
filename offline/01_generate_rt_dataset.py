import argparse
import json
import logging
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import mitsuba as mi
import numpy as np
import numpy.typing as npt
import sionna.rt as rt
from matplotlib.patches import Rectangle

from beamlink.config import (
    DATASET_FILENAME,
    DATASET_SCHEMA_VERSION,
    META_FILENAME,
    RT_TRACE_YAW,
    SCENE_IMAGE_FILENAME,
    TOPDOWN_HALF_WIDTH_M,
    TOPDOWN_RESOLUTION_PX,
    SystemConfig,
    load_config,
)
from beamlink.data import ChannelPaths, Dataset, SceneMeta, dataset_arrays, meta_document

logger = logging.getLogger("beamlink.offline")

BS_DEVICE_NAME = "bs"
UE_NAME_PREFIX = "ue"
PROGRESS_LOG_POSITIONS = 500
TOPDOWN_DOTS_PER_INCH = 100
GROUND_COLOR = "#f4f3ef"
BUILDING_COLOR = "#c9c6bd"
EDGE_COLOR = "#52514e"


@dataclass(frozen=True, slots=True)
class TracedPaths:
    paths: ChannelPaths
    is_los_path: npt.NDArray[np.bool_]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    return parser.parse_args()


def point3f(values: npt.NDArray[np.float64]) -> mi.Point3f:
    return mi.Point3f(*(float(value) for value in values))


def bs_position_m(config: SystemConfig) -> npt.NDArray[np.float64]:
    return np.array([*config.bs_position_xy_m, config.bs_height_m], dtype=np.float64)


def load_street_scene(
    config: SystemConfig,
    n_bs_antennas: int,
    *,
    pattern: str = "iso",
    orientation: tuple[float, float, float] = (RT_TRACE_YAW, 0.0, 0.0),
) -> rt.Scene:
    scene = rt.load_scene(getattr(rt.scene, config.scene_name), merge_shapes=False)
    scene.frequency = config.carrier_frequency_hz
    scene.tx_array = rt.PlanarArray(
        num_rows=1,
        num_cols=n_bs_antennas,
        horizontal_spacing=config.antenna_spacing_wavelengths,
        pattern=pattern,
        polarization="V",
    )
    scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    position = bs_position_m(config)
    scene.add(
        rt.Transmitter(BS_DEVICE_NAME, position=point3f(position), orientation=mi.Point3f(*orientation))
    )
    return scene


def building_footprints_m(scene: rt.Scene, ue_height_m: float) -> npt.NDArray[np.float64]:
    boxes = [scene_object.mi_mesh.bbox() for scene_object in scene.objects.values()]
    footprints = np.array([[box.min.x, box.max.x, box.min.y, box.max.y, box.max.z] for box in boxes])
    return footprints[footprints[:, 4] > ue_height_m, :4]


def ue_grid_positions_m(
    config: SystemConfig, footprints_m: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    x_min, x_max, y_min, y_max = config.ue_grid_extent_m
    grid_x, grid_y = np.meshgrid(
        np.arange(x_min, x_max + config.grid_spacing_m / 2, config.grid_spacing_m),
        np.arange(y_min, y_max + config.grid_spacing_m / 2, config.grid_spacing_m),
    )
    points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)
    clearance = config.building_clearance_m
    inside_x = (points[:, np.newaxis, 0] > footprints_m[:, 0] - clearance) & (
        points[:, np.newaxis, 0] < footprints_m[:, 1] + clearance
    )
    inside_y = (points[:, np.newaxis, 1] > footprints_m[:, 2] - clearance) & (
        points[:, np.newaxis, 1] < footprints_m[:, 3] + clearance
    )
    walkable = ~np.any(inside_x & inside_y, axis=-1)
    heights = np.full((int(walkable.sum()), 1), config.ue_height_m)
    return np.concatenate([points[walkable], heights], axis=-1)


def solve_paths(scene: rt.Scene, config: SystemConfig, seed: int) -> rt.Paths:
    solver = rt.PathSolver(deterministic=True)
    return solver(
        scene,
        max_depth=config.max_reflection_depth,
        samples_per_src=config.rt_samples_per_source,
        max_num_paths_per_src=config.rt_samples_per_source,
        synthetic_array=True,
        diffraction=config.rt_diffraction,
        seed=seed,
    )


def single_antenna_paths(paths: rt.Paths, carrier_frequency_hz: float) -> TracedPaths:
    gain_real, gain_imag = paths.a
    passband_gain = (
        gain_real.numpy()[:, 0, 0, 0, :].astype(np.float64) + 1j * gain_imag.numpy()[:, 0, 0, 0, :]
    )
    delay_s = paths.tau.numpy()[:, 0, :].astype(np.float64)
    valid = paths.valid.numpy()[:, 0, :].astype(bool)
    baseband_gain = np.where(valid, passband_gain * np.exp(-2j * np.pi * carrier_frequency_hz * delay_s), 0.0)
    interactions = paths.interactions.numpy()[:, :, 0, :]
    is_los_path = valid & np.all(interactions == int(rt.InteractionType.NONE), axis=0)
    traced = ChannelPaths(
        path_gain=baseband_gain.astype(np.complex64),
        path_delay_s=np.where(valid, delay_s, 0.0).astype(np.float32),
        path_azimuth=np.where(valid, paths.phi_t.numpy()[:, 0, :], 0.0).astype(np.float32),
        path_zenith=np.where(valid, paths.theta_t.numpy()[:, 0, :], 0.0).astype(np.float32),
        path_valid=valid,
    )
    return TracedPaths(paths=traced, is_los_path=is_los_path)


def trace_positions(
    scene: rt.Scene, positions_m: npt.NDArray[np.float64], config: SystemConfig, seed: int
) -> TracedPaths:
    names = [f"{UE_NAME_PREFIX}{idx}" for idx in range(positions_m.shape[0])]
    scene.add(
        [rt.Receiver(name, position=point3f(point)) for name, point in zip(names, positions_m, strict=True)]
    )
    traced = single_antenna_paths(solve_paths(scene, config, seed), config.carrier_frequency_hz)
    for name in names:
        scene.remove(name)
    return traced


def _pad_paths(traced: TracedPaths, n_paths: int) -> TracedPaths:
    missing = n_paths - traced.paths.path_valid.shape[1]
    if missing <= 0:
        return traced
    pad = ((0, 0), (0, missing))
    paths = traced.paths
    return TracedPaths(
        paths=ChannelPaths(
            path_gain=np.pad(paths.path_gain, pad),
            path_delay_s=np.pad(paths.path_delay_s, pad),
            path_azimuth=np.pad(paths.path_azimuth, pad),
            path_zenith=np.pad(paths.path_zenith, pad),
            path_valid=np.pad(paths.path_valid, pad),
        ),
        is_los_path=np.pad(traced.is_los_path, pad),
    )


def strongest_paths(traced: TracedPaths, max_paths: int) -> TracedPaths:
    traced = _pad_paths(traced, max_paths)
    power = np.where(traced.paths.path_valid, np.abs(traced.paths.path_gain) ** 2, -np.inf)
    order = np.argsort(-power, axis=-1, kind="stable")[:, :max_paths]
    paths = traced.paths
    return TracedPaths(
        paths=ChannelPaths(
            path_gain=np.take_along_axis(paths.path_gain, order, axis=-1),
            path_delay_s=np.take_along_axis(paths.path_delay_s, order, axis=-1),
            path_azimuth=np.take_along_axis(paths.path_azimuth, order, axis=-1),
            path_zenith=np.take_along_axis(paths.path_zenith, order, axis=-1),
            path_valid=np.take_along_axis(paths.path_valid, order, axis=-1),
        ),
        is_los_path=np.take_along_axis(traced.is_los_path, order, axis=-1),
    )


def concatenate_traces(batches: list[TracedPaths], n_paths: int) -> TracedPaths:
    padded = [_pad_paths(batch, n_paths) for batch in batches]
    return TracedPaths(
        paths=ChannelPaths(
            path_gain=np.concatenate([batch.paths.path_gain for batch in padded]),
            path_delay_s=np.concatenate([batch.paths.path_delay_s for batch in padded]),
            path_azimuth=np.concatenate([batch.paths.path_azimuth for batch in padded]),
            path_zenith=np.concatenate([batch.paths.path_zenith for batch in padded]),
            path_valid=np.concatenate([batch.paths.path_valid for batch in padded]),
        ),
        is_los_path=np.concatenate([batch.is_los_path for batch in padded]),
    )


def trace_grid(
    scene: rt.Scene, positions_m: npt.NDArray[np.float64], config: SystemConfig, seed: int
) -> TracedPaths:
    batch_starts = range(0, positions_m.shape[0], config.rt_receivers_per_batch)
    batches = []
    for start in batch_starts:
        batch = positions_m[start : start + config.rt_receivers_per_batch]
        batches.append(strongest_paths(trace_positions(scene, batch, config, seed), config.max_paths))
        if start % PROGRESS_LOG_POSITIONS < config.rt_receivers_per_batch:
            logger.info(f"traced positions={start + batch.shape[0]}/{positions_m.shape[0]}")
    return concatenate_traces(batches, config.max_paths)


def draw_topdown(
    footprints_m: npt.NDArray[np.float64], path: Path
) -> tuple[tuple[float, ...], tuple[int, ...]]:
    width_px, height_px = TOPDOWN_RESOLUTION_PX
    half_height_m = TOPDOWN_HALF_WIDTH_M * height_px / width_px
    extent = (-TOPDOWN_HALF_WIDTH_M, TOPDOWN_HALF_WIDTH_M, -half_height_m, half_height_m)
    figure = plt.figure(figsize=(width_px / TOPDOWN_DOTS_PER_INCH, height_px / TOPDOWN_DOTS_PER_INCH))
    panel = figure.add_axes((0.0, 0.0, 1.0, 1.0))
    panel.set_facecolor(GROUND_COLOR)
    for x_min, x_max, y_min, y_max in footprints_m:
        panel.add_patch(
            Rectangle(
                (x_min, y_min), x_max - x_min, y_max - y_min, facecolor=BUILDING_COLOR, edgecolor=EDGE_COLOR
            )
        )
    panel.set_xlim(extent[0], extent[1])
    panel.set_ylim(extent[2], extent[3])
    panel.set_axis_off()
    figure.patch.set_facecolor(GROUND_COLOR)
    figure.savefig(path, dpi=TOPDOWN_DOTS_PER_INCH, metadata={"Software": None})
    plt.close(figure)
    return extent, TOPDOWN_RESOLUTION_PX


def scene_meta(
    config: SystemConfig,
    footprints_m: npt.NDArray[np.float64],
    no_coverage_m: npt.NDArray[np.float64],
    image: tuple[tuple[float, ...], tuple[int, ...]],
) -> SceneMeta:
    return SceneMeta(
        schema_version=DATASET_SCHEMA_VERSION,
        scene_name=config.scene_name,
        carrier_frequency_hz=config.carrier_frequency_hz,
        bs_position_m=tuple(float(value) for value in bs_position_m(config)),
        grid_extent_m=config.ue_grid_extent_m,
        grid_spacing_m=config.grid_spacing_m,
        image_extent_m=image[0],
        image_size_px=image[1],
        building_footprints_m=tuple(tuple(float(value) for value in box) for box in footprints_m),
        no_coverage_positions_m=tuple(
            tuple(round(float(value), 3) for value in point) for point in no_coverage_m
        ),
        library_versions={
            package: version(package) for package in ("sionna-rt", "mitsuba", "drjit", "numpy")
        },
    )


def main() -> None:
    args = parse_args()
    matplotlib.use("Agg")
    config = load_config(args.config)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    scene = load_street_scene(config, n_bs_antennas=1)
    footprints_m = building_footprints_m(scene, config.ue_height_m)
    positions_m = ue_grid_positions_m(config, footprints_m)
    logger.info(
        f"scene={config.scene_name} buildings={footprints_m.shape[0]} ue_positions={positions_m.shape[0]}"
    )
    traced = trace_grid(scene, positions_m, config, args.seed)
    covered = traced.paths.path_valid.any(axis=-1)
    dataset = Dataset(
        positions_m=positions_m[covered].astype(np.float32),
        paths=ChannelPaths(
            path_gain=traced.paths.path_gain[covered],
            path_delay_s=traced.paths.path_delay_s[covered],
            path_azimuth=traced.paths.path_azimuth[covered],
            path_zenith=traced.paths.path_zenith[covered],
            path_valid=traced.paths.path_valid[covered],
        ),
        is_los=traced.is_los_path[covered].any(axis=-1),
    )
    dataset_path = args.out_dir / DATASET_FILENAME
    np.savez_compressed(dataset_path, **dataset_arrays(dataset))
    image = draw_topdown(footprints_m, args.out_dir / SCENE_IMAGE_FILENAME)
    meta = scene_meta(config, footprints_m, positions_m[~covered], image)
    (args.out_dir / META_FILENAME).write_text(json.dumps(meta_document(meta), indent=2), encoding="utf-8")
    logger.info(
        f"wrote {dataset_path} positions={int(covered.sum())} no_coverage={int((~covered).sum())} "
        f"los={int(dataset.is_los.sum())} size_mb={dataset_path.stat().st_size / 2**20:.2f}"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    main()
