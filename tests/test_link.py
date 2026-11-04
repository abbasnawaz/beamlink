import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from beamlink.config import (
    MAX_PLOTTED_SYMBOLS,
    SYNTHETIC_AZIMUTH_SPAN,
    SYNTHETIC_DELAY_DECAY_S,
    SYNTHETIC_ZENITH_SPREAD,
    THERMAL_NOISE_DBM_PER_HZ,
    BeamMethod,
    CsiMode,
    Modulation,
    SimulationMode,
    SystemConfig,
    load_config,
)
from beamlink.data import ChannelPaths, load_dataset, position_paths, synthetic_paths
from beamlink.exceptions import ConfigError, ShapeError
from beamlink.link import LinkResult, run_link
from beamlink.linkbudget import (
    db_to_linear,
    dbm_to_watt,
    linear_to_db,
    noise_power_watt,
    noise_variance_for_snr,
    watt_to_dbm,
)
from beamlink.metrics import awgn_ber_theory, fading_ber_theory

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
AWGN_SLOTS = 30
COMPARISON_SLOTS = 20
BER_SIGMAS = 5.0
FADING_THEORY_RELATIVE_TOLERANCE = 0.25
SHORT_PREFIX_PENALTY = 5.0
LONG_PREFIX_LENGTH = 40
SHORT_PREFIX_LENGTH = 1
LONG_DELAY_SPREAD_S = 500e-9
LONG_DELAY_DECAY_S = 250e-9
RUN_LINK_BUDGET_S = 1.0
N_PATHS = 8
PATH_GAIN_DB = -100.0
TX_POWER_DBM = 30.0
POWER_STEP_DB = 10.0
SNR_TOLERANCE_DB = 1e-3
N_DATASET_POSITIONS = 4
DEFAULT_NOISE_FLOOR_DBM = -92.126


def _config(**changes: object) -> SystemConfig:
    return replace(load_config(CONFIG_PATH), **changes)


def _awgn_paths() -> ChannelPaths:
    return ChannelPaths(
        path_gain=np.ones(1, dtype=np.complex64),
        path_delay_s=np.zeros(1, dtype=np.float32),
        path_azimuth=np.zeros(1, dtype=np.float32),
        path_zenith=np.full(1, np.pi / 2, dtype=np.float32),
        path_valid=np.ones(1, dtype=bool),
    )


def _multipath(seed: int, max_delay_s: float, delay_decay_s: float) -> ChannelPaths:
    paths = synthetic_paths(
        rng=np.random.default_rng(seed),
        n_positions=1,
        n_paths=N_PATHS,
        max_delay_s=max_delay_s,
        delay_decay_s=delay_decay_s,
        azimuth_span=SYNTHETIC_AZIMUTH_SPAN,
        zenith_spread=SYNTHETIC_ZENITH_SPREAD,
        los_power_ratio=None,
    )
    return position_paths(paths, 0)


def _run(
    paths: ChannelPaths,
    config: SystemConfig,
    *,
    method: BeamMethod = BeamMethod.SWEEP,
    modulation: Modulation = Modulation.QAM16,
    csi_mode: CsiMode = CsiMode.PERFECT,
    simulation_mode: SimulationMode = SimulationMode.FREQUENCY,
    snr_db: float | None = 10.0,
    tx_power_dbm: float | None = None,
    seed: int = 0,
    n_slots: int = 1,
    candidate_beams: npt.NDArray[np.int64] | None = None,
) -> LinkResult:
    return run_link(
        paths,
        config,
        method=method,
        modulation=modulation,
        csi_mode=csi_mode,
        simulation_mode=simulation_mode,
        snr_db=snr_db,
        tx_power_dbm=tx_power_dbm,
        seed=seed,
        n_slots=n_slots,
        candidate_beams=candidate_beams,
    )


def _ber_sigma(probability: float, n_bits: int) -> float:
    return float(np.sqrt(probability * (1.0 - probability) / n_bits))


def test_awgn_ber_matches_theory() -> None:
    config = _config(n_antennas=1)
    for modulation, snr_db in (
        (Modulation.QPSK, 2.0),
        (Modulation.QPSK, 6.0),
        (Modulation.QAM16, 10.0),
        (Modulation.QAM64, 16.0),
    ):
        result = _run(_awgn_paths(), config, modulation=modulation, snr_db=snr_db, seed=3, n_slots=AWGN_SLOTS)
        theory = float(awgn_ber_theory(db_to_linear(snr_db), modulation=modulation))
        assert abs(result.ber - theory) < BER_SIGMAS * _ber_sigma(theory, result.n_bits)


def test_frequency_and_time_modes_give_the_same_ber_with_a_long_prefix() -> None:
    config = _config(cyclic_prefix_length=LONG_PREFIX_LENGTH)
    paths = _multipath(1, LONG_DELAY_SPREAD_S, SYNTHETIC_DELAY_DECAY_S)
    results = [
        _run(paths, config, simulation_mode=mode, snr_db=6.0, seed=index, n_slots=COMPARISON_SLOTS)
        for index, mode in enumerate(SimulationMode)
    ]
    sigma = _ber_sigma(results[0].ber, results[0].n_bits)
    assert abs(results[0].ber - results[1].ber) < 2 * BER_SIGMAS * sigma


def test_short_prefix_adds_intersymbol_interference() -> None:
    paths = _multipath(2, LONG_DELAY_SPREAD_S, LONG_DELAY_DECAY_S)
    snr_db = 30.0
    long_prefix = _run(
        paths,
        _config(cyclic_prefix_length=LONG_PREFIX_LENGTH),
        simulation_mode=SimulationMode.TIME,
        snr_db=snr_db,
        n_slots=4,
    )
    short_prefix = _run(
        paths,
        _config(cyclic_prefix_length=SHORT_PREFIX_LENGTH),
        simulation_mode=SimulationMode.TIME,
        snr_db=snr_db,
        n_slots=4,
    )
    theory = fading_ber_theory(
        long_prefix.h_eff_true, noise_variance=noise_variance_for_snr(snr_db), modulation=Modulation.QAM16
    )
    assert long_prefix.ber == pytest.approx(theory, rel=FADING_THEORY_RELATIVE_TOLERANCE, abs=1e-4)
    assert short_prefix.ber > SHORT_PREFIX_PENALTY * max(long_prefix.ber, 1e-4)


def test_same_arguments_give_identical_results() -> None:
    paths = _multipath(3, LONG_DELAY_SPREAD_S, SYNTHETIC_DELAY_DECAY_S)
    first = _run(paths, _config(), csi_mode=CsiMode.LS, simulation_mode=SimulationMode.TIME, seed=9)
    second = _run(paths, _config(), csi_mode=CsiMode.LS, simulation_mode=SimulationMode.TIME, seed=9)
    assert first.ber == second.ber
    assert np.array_equal(first.symbols_after_equalization, second.symbols_after_equalization)


def test_methods_see_the_same_bits_and_noise() -> None:
    paths = _multipath(4, LONG_DELAY_SPREAD_S, SYNTHETIC_DELAY_DECAY_S)
    config = _config()
    sweep = _run(paths, config, csi_mode=CsiMode.LS, seed=5)
    candidates = np.array([sweep.beam_idx, (sweep.beam_idx + 1) % config.n_antennas], dtype=np.int64)
    top_k = _run(
        paths, config, method=BeamMethod.AI_TOP_K, csi_mode=CsiMode.LS, seed=5, candidate_beams=candidates
    )
    assert top_k.beams_tested == 2
    assert top_k.beam_idx == sweep.beam_idx
    assert top_k.ber == sweep.ber
    assert top_k.power_loss_db == pytest.approx(0.0, abs=1e-5)


def test_mrt_methods_never_lose_power_against_the_sweep() -> None:
    paths = _multipath(5, LONG_DELAY_SPREAD_S, SYNTHETIC_DELAY_DECAY_S)
    for method in (BeamMethod.MRT_WIDEBAND, BeamMethod.MRT_SUBCARRIER):
        result = _run(paths, _config(), method=method)
        assert result.power_loss_db <= 1e-4
        assert result.beams_tested == 0


def test_run_link_meets_the_latency_budget() -> None:
    paths = _multipath(6, LONG_DELAY_SPREAD_S, SYNTHETIC_DELAY_DECAY_S)
    config = _config(n_antennas=32)
    for simulation_mode in SimulationMode:
        for method in (BeamMethod.SWEEP, BeamMethod.MRT_WIDEBAND, BeamMethod.MRT_SUBCARRIER):
            start_s = time.perf_counter()
            _run(paths, config, method=method, csi_mode=CsiMode.LS, simulation_mode=simulation_mode)
            assert time.perf_counter() - start_s < RUN_LINK_BUDGET_S


def test_constellations_are_subsampled_for_plotting() -> None:
    result = _run(_multipath(7, LONG_DELAY_SPREAD_S, SYNTHETIC_DELAY_DECAY_S), _config(), n_slots=2)
    assert result.symbols_before_equalization.size <= MAX_PLOTTED_SYMBOLS
    assert result.symbols_after_equalization.size <= MAX_PLOTTED_SYMBOLS


def test_ai_top_k_without_candidates_is_rejected() -> None:
    with pytest.raises(ConfigError):
        _run(_awgn_paths(), _config(), method=BeamMethod.AI_TOP_K)


def test_mismatched_path_shapes_are_rejected() -> None:
    paths = _awgn_paths()
    broken = replace(paths, path_delay_s=np.zeros(2, dtype=np.float32))
    with pytest.raises(ShapeError):
        _run(broken, _config())


def _single_path(gain_db: float) -> ChannelPaths:
    paths = _awgn_paths()
    amplitude = np.sqrt(db_to_linear(gain_db))
    return replace(paths, path_gain=np.full(1, amplitude, dtype=np.complex64))


def test_noise_floor_is_thermal_noise_plus_bandwidth_plus_noise_figure() -> None:
    config = _config()
    expected_dbm = (
        THERMAL_NOISE_DBM_PER_HZ + float(linear_to_db(config.bandwidth_hz)) + config.noise_figure_db
    )
    noise_dbm = float(
        watt_to_dbm(
            noise_power_watt(bandwidth_hz=config.bandwidth_hz, noise_figure_db=config.noise_figure_db)
        )
    )
    assert noise_dbm == pytest.approx(expected_dbm, abs=SNR_TOLERANCE_DB)
    assert noise_dbm == pytest.approx(DEFAULT_NOISE_FLOOR_DBM, abs=1e-2)


def test_dbm_and_watt_conversions_round_trip() -> None:
    assert float(dbm_to_watt(30.0)) == pytest.approx(1.0)
    assert float(watt_to_dbm(dbm_to_watt(-17.5))) == pytest.approx(-17.5, abs=1e-4)


def test_tx_power_mode_snr_follows_the_link_budget() -> None:
    config = _config(n_antennas=1)
    result = _run(_single_path(PATH_GAIN_DB), config, snr_db=None, tx_power_dbm=TX_POWER_DBM)
    noise_dbm = float(
        watt_to_dbm(
            noise_power_watt(bandwidth_hz=config.bandwidth_hz, noise_figure_db=config.noise_figure_db)
        )
    )
    assert result.snr_db == pytest.approx(TX_POWER_DBM + PATH_GAIN_DB - noise_dbm, abs=SNR_TOLERANCE_DB)


def test_more_tx_power_gives_proportionally_more_snr() -> None:
    paths = _multipath(8, LONG_DELAY_SPREAD_S, SYNTHETIC_DELAY_DECAY_S)
    scaled = replace(
        paths, path_gain=(paths.path_gain * np.sqrt(db_to_linear(PATH_GAIN_DB))).astype(np.complex64)
    )
    low = _run(scaled, _config(), snr_db=None, tx_power_dbm=TX_POWER_DBM)
    high = _run(scaled, _config(), snr_db=None, tx_power_dbm=TX_POWER_DBM + POWER_STEP_DB)
    assert high.snr_db - low.snr_db == pytest.approx(POWER_STEP_DB, abs=SNR_TOLERANCE_DB)
    assert high.beam_idx == low.beam_idx


def test_exactly_one_power_mode_is_required() -> None:
    with pytest.raises(ConfigError):
        _run(_awgn_paths(), _config(), snr_db=None, tx_power_dbm=None)
    with pytest.raises(ConfigError):
        _run(_awgn_paths(), _config(), snr_db=10.0, tx_power_dbm=TX_POWER_DBM)


def test_run_link_on_ray_traced_positions_meets_the_latency_budget() -> None:
    if not (DATA_DIR / "dataset.npz").is_file():
        pytest.skip("data/dataset.npz has not been generated")
    dataset = load_dataset(DATA_DIR)
    config = _config(n_antennas=32)
    los_idx = np.flatnonzero(dataset.is_los)[: N_DATASET_POSITIONS // 2]
    nlos_idx = np.flatnonzero(~dataset.is_los)[: N_DATASET_POSITIONS // 2]
    for position_idx in np.concatenate([los_idx, nlos_idx]):
        paths = position_paths(dataset.paths, int(position_idx))
        for simulation_mode in SimulationMode:
            start_s = time.perf_counter()
            result = _run(
                paths,
                config,
                csi_mode=CsiMode.LS,
                simulation_mode=simulation_mode,
                snr_db=None,
                tx_power_dbm=config.tx_power_dbm,
            )
            assert time.perf_counter() - start_s < RUN_LINK_BUDGET_S
            assert np.isfinite(result.snr_db)
            assert result.beams_tested == config.n_antennas
