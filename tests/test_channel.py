import numpy as np
import numpy.typing as npt

from beamlink.array import ula_steering_vectors
from beamlink.channel import (
    cir_to_cfr,
    n_cir_taps,
    paths_to_cfr,
    paths_to_cir,
    rms_delay_spread_s,
    subcarrier_frequencies_hz,
    time_aligned_delays,
    unit_power_gain,
)
from beamlink.config import (
    SYNTHETIC_AZIMUTH_SPAN,
    SYNTHETIC_DELAY_DECAY_S,
    SYNTHETIC_MAX_DELAY_S,
    SYNTHETIC_N_PATHS,
    SYNTHETIC_ZENITH_SPREAD,
)
from beamlink.data import ChannelPaths, synthetic_paths

N_SUBCARRIERS = 256
SUBCARRIER_SPACING_HZ = 120e3
SAMPLE_RATE_HZ = N_SUBCARRIERS * SUBCARRIER_SPACING_HZ
N_PRECURSOR_TAPS = 4
N_POSTCURSOR_TAPS = 4
SINC_TRUNCATION_TOLERANCE = 0.2


def _paths(seed: int, n_positions: int = 1) -> ChannelPaths:
    return synthetic_paths(
        rng=np.random.default_rng(seed),
        n_positions=n_positions,
        n_paths=SYNTHETIC_N_PATHS,
        max_delay_s=SYNTHETIC_MAX_DELAY_S,
        delay_decay_s=SYNTHETIC_DELAY_DECAY_S,
        azimuth_span=SYNTHETIC_AZIMUTH_SPAN,
        zenith_spread=SYNTHETIC_ZENITH_SPREAD,
        los_power_ratio=None,
    )


def _steering(paths: ChannelPaths, n_antennas: int) -> npt.NDArray[np.complex64]:
    return ula_steering_vectors(
        paths.path_azimuth,
        paths.path_zenith,
        n_antennas=n_antennas,
        spacing_wavelengths=0.5,
        boresight_azimuth=0.0,
    )


def test_cfr_has_canonical_shape_for_batched_positions() -> None:
    paths = _paths(0, n_positions=5)
    cfr = paths_to_cfr(
        paths.path_gain,
        paths.path_delay_s,
        _steering(paths, 8),
        n_subcarriers=N_SUBCARRIERS,
        subcarrier_spacing_hz=SUBCARRIER_SPACING_HZ,
    )
    assert cfr.shape == (5, 8, N_SUBCARRIERS)
    assert cfr.dtype == np.complex64


def test_sinc_cir_reproduces_path_cfr_up_to_timing_backoff() -> None:
    paths = _paths(1)
    single_antenna = np.ones((1, 1, SYNTHETIC_N_PATHS), dtype=np.complex64)
    cfr = paths_to_cfr(
        paths.path_gain,
        paths.path_delay_s,
        single_antenna,
        n_subcarriers=N_SUBCARRIERS,
        subcarrier_spacing_hz=SUBCARRIER_SPACING_HZ,
    )
    n_taps = n_cir_taps(
        paths.path_delay_s,
        sample_rate_hz=SAMPLE_RATE_HZ,
        n_precursor_taps=N_PRECURSOR_TAPS,
        n_postcursor_taps=N_POSTCURSOR_TAPS,
    )
    cir = paths_to_cir(
        paths.path_gain,
        paths.path_delay_s,
        single_antenna,
        sample_rate_hz=SAMPLE_RATE_HZ,
        n_precursor_taps=N_PRECURSOR_TAPS,
        n_taps=n_taps,
    )
    frequencies_hz = subcarrier_frequencies_hz(
        n_subcarriers=N_SUBCARRIERS, subcarrier_spacing_hz=SUBCARRIER_SPACING_HZ
    )
    backoff_phase = np.exp(-2j * np.pi * frequencies_hz * N_PRECURSOR_TAPS / SAMPLE_RATE_HZ)
    expected = cfr * backoff_phase
    relative_error = np.linalg.norm(cir_to_cfr(cir, n_subcarriers=N_SUBCARRIERS) - expected)
    assert relative_error / np.linalg.norm(expected) < SINC_TRUNCATION_TOLERANCE


def test_time_alignment_puts_first_valid_path_at_zero() -> None:
    delay_s = np.array([[300e-9, 120e-9, 50e-9]], dtype=np.float32)
    valid = np.array([[True, True, False]])
    aligned = time_aligned_delays(delay_s, valid)
    assert np.allclose(aligned, [[180e-9, 0.0, 0.0]])


def test_rms_delay_spread_of_two_equal_paths_is_half_their_spacing() -> None:
    gain = np.ones(2, dtype=np.complex64)
    delay_s = np.array([0.0, 100e-9], dtype=np.float32)
    assert np.isclose(rms_delay_spread_s(gain, delay_s), 50e-9)


def test_unit_power_gain_normalizes_mean_cfr_power() -> None:
    paths = _paths(2)
    steering = _steering(paths, 8)
    grid_axes = {"n_subcarriers": N_SUBCARRIERS, "subcarrier_spacing_hz": SUBCARRIER_SPACING_HZ}
    cfr = paths_to_cfr(paths.path_gain, paths.path_delay_s, steering, **grid_axes)
    scaled_gain = unit_power_gain(paths.path_gain, cfr)
    scaled_cfr = paths_to_cfr(scaled_gain, paths.path_delay_s, steering, **grid_axes)
    assert np.isclose(np.mean(np.abs(scaled_cfr) ** 2), 1.0, rtol=1e-4)
