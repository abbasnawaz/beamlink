import json
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from beamlink.config import (
    DATASET_FILENAME,
    DATASET_SCHEMA_VERSION,
    META_FILENAME,
    WIDE_BEAM_ELEMENTS,
    FeatureSet,
    load_config,
)
from beamlink.data import (
    ChannelPaths,
    Dataset,
    Labels,
    SceneMeta,
    dataset_arrays,
    feature_names,
    labels_match_config,
    load_dataset,
    load_meta,
    meta_document,
    n_probe_beams,
    position_paths,
    predictor_features,
)
from beamlink.exceptions import ArtifactError

N_POSITIONS = 5
N_PATHS = 4
LABEL_ANTENNA_COUNTS = (8, 16)
SECTOR_BORESIGHTS = (0.0, np.pi / 2, np.pi, 3 * np.pi / 2)
N_SECTORS = len(SECTOR_BORESIGHTS)
N_WIDE_BEAMS = N_SECTORS * WIDE_BEAM_ELEMENTS
CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def _dataset(seed: int) -> Dataset:
    rng = np.random.default_rng(seed)
    shape = (N_POSITIONS, N_PATHS)
    gain = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex64)
    paths = ChannelPaths(
        path_gain=gain,
        path_delay_s=rng.uniform(0, 1e-6, shape).astype(np.float32),
        path_azimuth=rng.uniform(-np.pi, np.pi, shape).astype(np.float32),
        path_zenith=rng.uniform(0, np.pi, shape).astype(np.float32),
        path_valid=rng.random(shape) > 0.3,
    )
    return Dataset(
        positions_m=rng.uniform(-50, 50, (N_POSITIONS, 3)).astype(np.float32),
        paths=paths,
        is_los=rng.random(N_POSITIONS) > 0.5,
    )


def _meta() -> SceneMeta:
    return SceneMeta(
        schema_version=DATASET_SCHEMA_VERSION,
        scene_name="simple_street_canyon",
        carrier_frequency_hz=28e9,
        bs_position_m=(0.0, -7.6, 12.0),
        grid_extent_m=(-90.0, 90.0, -58.0, 58.0),
        grid_spacing_m=1.5,
        image_extent_m=(-100.0, 100.0, -65.0, 65.0),
        image_size_px=(1200, 780),
        building_footprints_m=((-15.0, 16.0, -36.5, -8.6),),
        no_coverage_positions_m=((1.0, 2.0, 1.5),),
        library_versions={"sionna-rt": "2.2.0"},
    )


def _write_arrays(directory: Path, arrays: dict[str, npt.NDArray[np.generic]]) -> None:
    np.savez_compressed(directory / DATASET_FILENAME, **arrays)


def test_dataset_round_trip_preserves_every_array(tmp_path: Path) -> None:
    original = _dataset(0)
    _write_arrays(tmp_path, dataset_arrays(original))
    loaded = load_dataset(tmp_path)
    assert np.array_equal(loaded.positions_m, original.positions_m)
    assert np.array_equal(loaded.is_los, original.is_los)
    assert loaded.paths.path_gain.dtype == np.complex64
    assert np.array_equal(loaded.paths.path_gain, original.paths.path_gain)
    assert np.array_equal(loaded.paths.path_valid, original.paths.path_valid)


def test_missing_key_is_rejected(tmp_path: Path) -> None:
    arrays = dataset_arrays(_dataset(1))
    del arrays["path_zenith"]
    _write_arrays(tmp_path, arrays)
    with pytest.raises(ArtifactError, match="path_zenith"):
        load_dataset(tmp_path)


def test_wrong_dtype_is_rejected(tmp_path: Path) -> None:
    arrays = dataset_arrays(_dataset(2))
    arrays["path_delay_s"] = arrays["path_delay_s"].astype(np.float64)
    _write_arrays(tmp_path, arrays)
    with pytest.raises(ArtifactError, match="dtype"):
        load_dataset(tmp_path)


def test_wrong_shape_is_rejected(tmp_path: Path) -> None:
    arrays = dataset_arrays(_dataset(3))
    arrays["is_los"] = arrays["is_los"][:-1]
    _write_arrays(tmp_path, arrays)
    with pytest.raises(ArtifactError, match="shape"):
        load_dataset(tmp_path)


def test_missing_dataset_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError, match="not found"):
        load_dataset(tmp_path)


def test_meta_round_trip(tmp_path: Path) -> None:
    (tmp_path / META_FILENAME).write_text(json.dumps(meta_document(_meta())))
    assert load_meta(tmp_path) == _meta()


def test_meta_with_wrong_schema_version_is_rejected(tmp_path: Path) -> None:
    document = meta_document(_meta())
    document["schema_version"] = DATASET_SCHEMA_VERSION + 1
    (tmp_path / META_FILENAME).write_text(json.dumps(document))
    with pytest.raises(ArtifactError, match="schema_version"):
        load_meta(tmp_path)


def test_meta_with_unknown_key_is_rejected(tmp_path: Path) -> None:
    document = meta_document(_meta())
    document["extra"] = 1
    (tmp_path / META_FILENAME).write_text(json.dumps(document))
    with pytest.raises(ArtifactError, match="unknown"):
        load_meta(tmp_path)


def test_position_paths_selects_one_row() -> None:
    dataset = _dataset(4)
    single = position_paths(dataset.paths, 2)
    assert single.path_gain.shape == (N_PATHS,)
    assert np.array_equal(single.path_delay_s, dataset.paths.path_delay_s[2])


def _labelled(seed: int) -> Dataset:
    rng = np.random.default_rng(seed)
    base = _dataset(seed)
    power = {
        count: rng.uniform(-150, -80, (N_POSITIONS, N_SECTORS * count)).astype(np.float32)
        for count in LABEL_ANTENNA_COUNTS
    }
    labels = Labels(
        beam_power_db=power,
        best_beam={
            count: np.argmax(power[count], axis=-1).astype(np.int16) for count in LABEL_ANTENNA_COUNTS
        },
        wide_beam_power_db=rng.uniform(-150, -80, (N_POSITIONS, N_WIDE_BEAMS)).astype(np.float32),
        split=rng.integers(0, 3, N_POSITIONS).astype(np.int8),
        sector_boresight_azimuths=np.asarray(SECTOR_BORESIGHTS, dtype=np.float32),
        sector_downtilt=float(np.float32(np.deg2rad(10.0))),
    )
    return Dataset(positions_m=base.positions_m, paths=base.paths, is_los=base.is_los, labels=labels)


def test_labels_round_trip(tmp_path: Path) -> None:
    original = _labelled(5)
    _write_arrays(tmp_path, dataset_arrays(original))
    loaded = load_dataset(tmp_path)
    assert loaded.labels is not None
    assert sorted(loaded.labels.beam_power_db) == list(LABEL_ANTENNA_COUNTS)
    for count in LABEL_ANTENNA_COUNTS:
        assert np.array_equal(loaded.labels.beam_power_db[count], original.labels.beam_power_db[count])
        assert loaded.labels.best_beam[count].dtype == np.int16
    assert np.array_equal(loaded.labels.split, original.labels.split)
    assert np.array_equal(loaded.labels.sector_boresight_azimuths, original.labels.sector_boresight_azimuths)
    assert loaded.labels.sector_downtilt == pytest.approx(original.labels.sector_downtilt)


def test_unlabelled_dataset_loads_without_labels(tmp_path: Path) -> None:
    _write_arrays(tmp_path, dataset_arrays(_dataset(6)))
    assert load_dataset(tmp_path).labels is None


def test_partial_labels_are_rejected(tmp_path: Path) -> None:
    arrays = dataset_arrays(_labelled(7))
    del arrays["best_beam_n16"]
    _write_arrays(tmp_path, arrays)
    with pytest.raises(ArtifactError, match="best_beam_n16"):
        load_dataset(tmp_path)


def test_predictor_features_follow_the_feature_set() -> None:
    dataset = _labelled(8)
    for feature_set in FeatureSet:
        features = predictor_features(dataset, feature_set)
        assert features.shape == (N_POSITIONS, len(feature_names(feature_set, N_WIDE_BEAMS)))
        assert features.dtype == np.float32
    assert n_probe_beams(FeatureSet.POSITION, N_WIDE_BEAMS) == 0
    assert n_probe_beams(FeatureSet.POSITION_WIDE_BEAMS, N_WIDE_BEAMS) == N_WIDE_BEAMS


def test_wide_beam_features_need_labels() -> None:
    with pytest.raises(ArtifactError, match="labels"):
        predictor_features(_dataset(9), FeatureSet.POSITION_WIDE_BEAMS)


def test_beam_count_that_does_not_fit_the_sectors_is_rejected(tmp_path: Path) -> None:
    arrays = dataset_arrays(_labelled(10))
    arrays["beam_power_db_n8"] = arrays["beam_power_db_n8"][:, :-1]
    _write_arrays(tmp_path, arrays)
    with pytest.raises(ArtifactError, match="beam_power_db_n8"):
        load_dataset(tmp_path)


def test_labels_from_another_sector_layout_do_not_match_the_config() -> None:
    labels = _labelled(11).labels
    config = load_config(CONFIG_PATH)
    assert labels_match_config(labels, config)
    rotated = Labels(
        beam_power_db=labels.beam_power_db,
        best_beam=labels.best_beam,
        wide_beam_power_db=labels.wide_beam_power_db,
        split=labels.split,
        sector_boresight_azimuths=labels.sector_boresight_azimuths + np.float32(0.1),
        sector_downtilt=labels.sector_downtilt,
    )
    assert not labels_match_config(rotated, config)
