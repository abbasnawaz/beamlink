import dataclasses
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from beamlink.config import (
    DATASET_FILENAME,
    DATASET_SCHEMA_VERSION,
    META_FILENAME,
    N_WIDE_BEAMS,
    SCENE_IMAGE_FILENAME,
    FeatureSet,
)
from beamlink.exceptions import ArtifactError

logger = logging.getLogger("beamlink")

_PATH_ARRAY_DTYPES = {
    "path_gain_real": np.float32,
    "path_gain_imag": np.float32,
    "path_delay_s": np.float32,
    "path_azimuth": np.float32,
    "path_zenith": np.float32,
    "path_valid": np.bool_,
}


@dataclass(frozen=True, slots=True)
class ChannelPaths:
    path_gain: npt.NDArray[np.complex64]
    path_delay_s: npt.NDArray[np.float32]
    path_azimuth: npt.NDArray[np.float32]
    path_zenith: npt.NDArray[np.float32]
    path_valid: npt.NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class Labels:
    beam_power_db: dict[int, npt.NDArray[np.float32]]
    best_beam: dict[int, npt.NDArray[np.int16]]
    wide_beam_power_db: npt.NDArray[np.float32]
    split: npt.NDArray[np.int8]


@dataclass(frozen=True, slots=True)
class Dataset:
    positions_m: npt.NDArray[np.float32]
    paths: ChannelPaths
    is_los: npt.NDArray[np.bool_]
    labels: Labels | None = None


@dataclass(frozen=True, slots=True)
class SceneMeta:
    schema_version: int
    scene_name: str
    carrier_frequency_hz: float
    bs_position_m: tuple[float, ...]
    bs_boresight_azimuth: float
    grid_extent_m: tuple[float, ...]
    grid_spacing_m: float
    image_extent_m: tuple[float, ...]
    image_size_px: tuple[int, ...]
    building_footprints_m: tuple[tuple[float, ...], ...]
    no_coverage_positions_m: tuple[tuple[float, ...], ...]
    library_versions: dict[str, str]


def synthetic_paths(
    *,
    rng: np.random.Generator,
    n_positions: int,
    n_paths: int,
    max_delay_s: float,
    delay_decay_s: float,
    azimuth_span: tuple[float, float],
    zenith_spread: float,
    los_power_ratio: float | None,
) -> ChannelPaths:
    shape = (n_positions, n_paths)
    later_delays_s = rng.uniform(0.0, max_delay_s, size=(n_positions, n_paths - 1))
    delay_s = np.sort(np.concatenate([np.zeros((n_positions, 1)), later_delays_s], axis=-1), axis=-1)
    mean_power = np.exp(-delay_s / delay_decay_s)
    gain = np.sqrt(mean_power / 2.0) * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
    if los_power_ratio is not None:
        scattered_power = np.sum(np.abs(gain[:, 1:]) ** 2, axis=-1)
        los_phase = rng.uniform(0.0, 2.0 * np.pi, size=n_positions)
        gain[:, 0] = np.sqrt(los_power_ratio * scattered_power) * np.exp(1j * los_phase)
    gain = gain / np.sqrt(np.sum(np.abs(gain) ** 2, axis=-1, keepdims=True))
    azimuth = rng.uniform(azimuth_span[0], azimuth_span[1], size=shape)
    zenith = np.pi / 2.0 + rng.uniform(-zenith_spread, zenith_spread, size=shape)
    return ChannelPaths(
        path_gain=gain.astype(np.complex64),
        path_delay_s=delay_s.astype(np.float32),
        path_azimuth=azimuth.astype(np.float32),
        path_zenith=zenith.astype(np.float32),
        path_valid=np.ones(shape, dtype=bool),
    )


def position_paths(paths: ChannelPaths, position_idx: int) -> ChannelPaths:
    return ChannelPaths(
        path_gain=paths.path_gain[position_idx],
        path_delay_s=paths.path_delay_s[position_idx],
        path_azimuth=paths.path_azimuth[position_idx],
        path_zenith=paths.path_zenith[position_idx],
        path_valid=paths.path_valid[position_idx],
    )


def dataset_arrays(dataset: Dataset) -> dict[str, npt.NDArray[np.generic]]:
    paths = dataset.paths
    return {
        "positions_m": dataset.positions_m.astype(np.float32),
        "path_gain_real": paths.path_gain.real.astype(np.float32),
        "path_gain_imag": paths.path_gain.imag.astype(np.float32),
        "path_delay_s": paths.path_delay_s.astype(np.float32),
        "path_azimuth": paths.path_azimuth.astype(np.float32),
        "path_zenith": paths.path_zenith.astype(np.float32),
        "path_valid": paths.path_valid.astype(np.bool_),
        "is_los": dataset.is_los.astype(np.bool_),
        **_label_arrays(dataset.labels),
    }


def _label_arrays(labels: Labels | None) -> dict[str, npt.NDArray[np.generic]]:
    if labels is None:
        return {}
    arrays: dict[str, npt.NDArray[np.generic]] = {
        "wide_beam_power_db": labels.wide_beam_power_db.astype(np.float32),
        "split": labels.split.astype(np.int8),
    }
    for n_antennas, beam_power_db in labels.beam_power_db.items():
        arrays[f"beam_power_db_n{n_antennas}"] = beam_power_db.astype(np.float32)
        arrays[f"best_beam_n{n_antennas}"] = labels.best_beam[n_antennas].astype(np.int16)
    return arrays


def meta_document(meta: SceneMeta) -> dict[str, object]:
    return dataclasses.asdict(meta)


def _require_array(
    arrays: dict[str, npt.NDArray[np.generic]], path: Path, key: str, dtype: type, shape: tuple[int, ...]
) -> npt.NDArray[np.generic]:
    if key not in arrays:
        raise ArtifactError(str(path), f"missing key {key!r}")
    array = arrays[key]
    if array.dtype != dtype:
        raise ArtifactError(str(path), f"{key} has dtype {array.dtype}, expected {np.dtype(dtype)}")
    if array.shape != shape:
        raise ArtifactError(str(path), f"{key} has shape {array.shape}, expected {shape}")
    return array


def dataset_from_arrays(arrays: dict[str, npt.NDArray[np.generic]], path: Path) -> Dataset:
    if "positions_m" not in arrays or arrays["positions_m"].ndim != 2:
        raise ArtifactError(str(path), "positions_m must exist with shape (n_positions, 3)")
    n_positions = arrays["positions_m"].shape[0]
    if "path_valid" not in arrays or arrays["path_valid"].ndim != 2:
        raise ArtifactError(str(path), "path_valid must exist with shape (n_positions, n_paths)")
    path_shape = (n_positions, arrays["path_valid"].shape[1])
    positions_m = _require_array(arrays, path, "positions_m", np.float32, (n_positions, 3))
    checked = {
        key: _require_array(arrays, path, key, dtype, path_shape) for key, dtype in _PATH_ARRAY_DTYPES.items()
    }
    is_los = _require_array(arrays, path, "is_los", np.bool_, (n_positions,))
    paths = ChannelPaths(
        path_gain=(checked["path_gain_real"] + 1j * checked["path_gain_imag"]).astype(np.complex64),
        path_delay_s=checked["path_delay_s"],
        path_azimuth=checked["path_azimuth"],
        path_zenith=checked["path_zenith"],
        path_valid=checked["path_valid"],
    )
    labels = _labels_from_arrays(arrays, path, n_positions)
    return Dataset(positions_m=positions_m, paths=paths, is_los=is_los, labels=labels)


def _labels_from_arrays(
    arrays: dict[str, npt.NDArray[np.generic]], path: Path, n_positions: int
) -> Labels | None:
    antenna_counts = sorted(
        int(match.group(1)) for key in arrays if (match := re.fullmatch(r"beam_power_db_n(\d+)", key))
    )
    if not antenna_counts and "wide_beam_power_db" not in arrays and "split" not in arrays:
        return None
    if not antenna_counts:
        raise ArtifactError(str(path), "labels present but no beam_power_db_n{N} key")
    beam_power_db, best_beam = {}, {}
    for n_antennas in antenna_counts:
        power_key = f"beam_power_db_n{n_antennas}"
        n_beams = arrays[power_key].shape[-1] if arrays[power_key].ndim == 2 else 0
        beam_power_db[n_antennas] = _require_array(
            arrays, path, power_key, np.float32, (n_positions, n_beams)
        )
        best_beam[n_antennas] = _require_array(
            arrays, path, f"best_beam_n{n_antennas}", np.int16, (n_positions,)
        )
    return Labels(
        beam_power_db=beam_power_db,
        best_beam=best_beam,
        wide_beam_power_db=_require_array(
            arrays, path, "wide_beam_power_db", np.float32, (n_positions, N_WIDE_BEAMS)
        ),
        split=_require_array(arrays, path, "split", np.int8, (n_positions,)),
    )


def load_dataset(data_dir: Path) -> Dataset:
    path = data_dir / DATASET_FILENAME
    if not path.is_file():
        raise ArtifactError(str(path), "file not found")
    with np.load(path) as archive:
        arrays = {key: archive[key] for key in archive.files}
    dataset = dataset_from_arrays(arrays, path)
    size_mb = path.stat().st_size / 2**20
    logger.info(f"loaded {path} positions={dataset.positions_m.shape[0]} size_mb={size_mb:.1f}")
    return dataset


def meta_from_document(document: dict[str, object], path: Path) -> SceneMeta:
    expected = {field.name for field in dataclasses.fields(SceneMeta)}
    if set(document) != expected:
        missing, unknown = sorted(expected - set(document)), sorted(set(document) - expected)
        raise ArtifactError(str(path), f"missing keys {missing}, unknown keys {unknown}")
    if document["schema_version"] != DATASET_SCHEMA_VERSION:
        raise ArtifactError(
            str(path), f"schema_version {document['schema_version']}, expected {DATASET_SCHEMA_VERSION}"
        )
    return SceneMeta(
        schema_version=int(document["schema_version"]),
        scene_name=str(document["scene_name"]),
        carrier_frequency_hz=float(document["carrier_frequency_hz"]),
        bs_position_m=tuple(float(value) for value in document["bs_position_m"]),
        bs_boresight_azimuth=float(document["bs_boresight_azimuth"]),
        grid_extent_m=tuple(float(value) for value in document["grid_extent_m"]),
        grid_spacing_m=float(document["grid_spacing_m"]),
        image_extent_m=tuple(float(value) for value in document["image_extent_m"]),
        image_size_px=tuple(int(value) for value in document["image_size_px"]),
        building_footprints_m=tuple(
            tuple(float(value) for value in box) for box in document["building_footprints_m"]
        ),
        no_coverage_positions_m=tuple(
            tuple(float(value) for value in point) for point in document["no_coverage_positions_m"]
        ),
        library_versions={str(key): str(value) for key, value in dict(document["library_versions"]).items()},
    )


def load_meta(data_dir: Path) -> SceneMeta:
    path = data_dir / META_FILENAME
    if not path.is_file():
        raise ArtifactError(str(path), "file not found")
    meta = meta_from_document(json.loads(path.read_text(encoding="utf-8")), path)
    logger.info(f"loaded {path} scene={meta.scene_name} size_kb={path.stat().st_size / 1024:.1f}")
    return meta


def feature_names(feature_set: FeatureSet) -> tuple[str, ...]:
    position = ("position_x_m", "position_y_m")
    if feature_set is FeatureSet.POSITION:
        return position
    return position + tuple(f"wide_beam_{beam_idx}_power_db" for beam_idx in range(N_WIDE_BEAMS))


def predictor_features(dataset: Dataset, feature_set: FeatureSet) -> npt.NDArray[np.float32]:
    position = dataset.positions_m[:, :2]
    if feature_set is FeatureSet.POSITION:
        return position.astype(np.float32)
    if dataset.labels is None:
        raise ArtifactError(
            DATASET_FILENAME, "wide-beam features need labels; run offline/02_build_labels.py"
        )
    return np.concatenate([position, dataset.labels.wide_beam_power_db], axis=-1).astype(np.float32)


def n_probe_beams(feature_set: FeatureSet) -> int:
    return N_WIDE_BEAMS if feature_set is FeatureSet.POSITION_WIDE_BEAMS else 0


def nearest_position_idx(positions_m: npt.NDArray[np.float32], x_m: float, y_m: float) -> int:
    squared_distance = (positions_m[:, 0] - x_m) ** 2 + (positions_m[:, 1] - y_m) ** 2
    return int(np.argmin(squared_distance))


def load_scene_image(data_dir: Path) -> bytes:
    path = data_dir / SCENE_IMAGE_FILENAME
    if not path.is_file():
        raise ArtifactError(str(path), "file not found")
    image = path.read_bytes()
    logger.info(f"loaded {path} size_kb={len(image) / 1024:.1f}")
    return image
