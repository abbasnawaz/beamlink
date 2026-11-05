import argparse
import logging
from dataclasses import replace
from pathlib import Path

import numpy as np
import numpy.typing as npt

from beamlink.beamforming import sweep_beams
from beamlink.channel import paths_to_cfr, time_aligned_delays
from beamlink.config import (
    DATASET_FILENAME,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VALIDATION,
    WIDE_BEAM_ELEMENTS,
    SystemConfig,
    load_config,
)
from beamlink.data import ChannelPaths, Dataset, Labels, dataset_arrays, load_dataset
from beamlink.link import array_response, system_codebook
from beamlink.linkbudget import linear_to_db

logger = logging.getLogger("beamlink.offline")

CHUNK_POSITIONS = 500


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    return parser.parse_args()


def paths_slice(paths: ChannelPaths, rows: slice) -> ChannelPaths:
    return ChannelPaths(
        path_gain=paths.path_gain[rows],
        path_delay_s=paths.path_delay_s[rows],
        path_azimuth=paths.path_azimuth[rows],
        path_zenith=paths.path_zenith[rows],
        path_valid=paths.path_valid[rows],
    )


def position_cfr(paths: ChannelPaths, config: SystemConfig) -> npt.NDArray[np.complex64]:
    return paths_to_cfr(
        np.where(paths.path_valid, paths.path_gain, 0.0).astype(np.complex64),
        time_aligned_delays(paths.path_delay_s, paths.path_valid),
        array_response(paths.path_azimuth, paths.path_zenith, config),
        n_subcarriers=config.n_subcarriers,
        subcarrier_spacing_hz=config.subcarrier_spacing_hz,
    )


def beam_power_db(paths: ChannelPaths, config: SystemConfig) -> npt.NDArray[np.float32]:
    codebook = system_codebook(config)
    chunks = [
        sweep_beams(position_cfr(paths_slice(paths, slice(start, start + CHUNK_POSITIONS)), config), codebook)
        for start in range(0, paths.path_gain.shape[0], CHUNK_POSITIONS)
    ]
    return linear_to_db(np.concatenate(chunks) / config.n_subcarriers)


def spatial_split(
    positions_m: npt.NDArray[np.float32], config: SystemConfig, rng: np.random.Generator
) -> npt.NDArray[np.int8]:
    x_min, _, y_min, _ = config.ue_grid_extent_m
    block_x = np.floor((positions_m[:, 0] - x_min) / config.split_block_m).astype(np.int64)
    block_y = np.floor((positions_m[:, 1] - y_min) / config.split_block_m).astype(np.int64)
    blocks, block_of_position = np.unique(np.stack([block_x, block_y], axis=-1), axis=0, return_inverse=True)
    shuffled = rng.permutation(blocks.shape[0])
    train_fraction, validation_fraction, _ = config.split_fractions
    n_train = int(round(train_fraction * blocks.shape[0]))
    n_validation = int(round(validation_fraction * blocks.shape[0]))
    split_of_block = np.full(blocks.shape[0], SPLIT_TEST, dtype=np.int8)
    split_of_block[shuffled[:n_train]] = SPLIT_TRAIN
    split_of_block[shuffled[n_train : n_train + n_validation]] = SPLIT_VALIDATION
    return split_of_block[block_of_position.ravel()]


def build_labels(dataset: Dataset, config: SystemConfig, rng: np.random.Generator) -> Labels:
    power_by_antennas: dict[int, npt.NDArray[np.float32]] = {}
    best_by_antennas: dict[int, npt.NDArray[np.int16]] = {}
    for n_antennas in config.label_antenna_counts:
        power_by_antennas[n_antennas] = beam_power_db(dataset.paths, replace(config, n_antennas=n_antennas))
        best_by_antennas[n_antennas] = np.argmax(power_by_antennas[n_antennas], axis=-1).astype(np.int16)
    wide_config = replace(
        config, n_antennas=WIDE_BEAM_ELEMENTS, codebook_oversampling=1, codebook_sector=None
    )
    return Labels(
        beam_power_db=power_by_antennas,
        best_beam=best_by_antennas,
        wide_beam_power_db=beam_power_db(dataset.paths, wide_config),
        split=spatial_split(dataset.positions_m, config, rng),
        sector_boresight_azimuths=np.asarray(config.sector_boresight_azimuths, dtype=np.float32),
        sector_downtilt=config.sector_downtilt,
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    rng = np.random.default_rng(args.seed)
    dataset = load_dataset(args.data_dir)
    labels = build_labels(dataset, config, rng)
    labelled = Dataset(
        positions_m=dataset.positions_m, paths=dataset.paths, is_los=dataset.is_los, labels=labels
    )
    path = args.data_dir / DATASET_FILENAME
    np.savez_compressed(path, **dataset_arrays(labelled))
    for n_antennas, best in labels.best_beam.items():
        usage = np.bincount(best, minlength=labels.beam_power_db[n_antennas].shape[1])
        logger.info(f"labels n_antennas={n_antennas} beam_usage={usage.tolist()}")
    split_counts = np.bincount(labels.split, minlength=3)
    logger.info(
        f"wrote {path} train={split_counts[SPLIT_TRAIN]} validation={split_counts[SPLIT_VALIDATION]} "
        f"test={split_counts[SPLIT_TEST]} size_mb={path.stat().st_size / 2**20:.2f}"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    main()
