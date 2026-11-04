import numpy as np

from beamlink.array import ula_steering_vectors
from beamlink.channel import paths_to_cfr
from beamlink.config import (
    SYNTHETIC_AZIMUTH_SPAN,
    SYNTHETIC_DELAY_DECAY_S,
    SYNTHETIC_MAX_DELAY_S,
    SYNTHETIC_N_PATHS,
    SYNTHETIC_ZENITH_SPREAD,
)
from beamlink.data import synthetic_paths
from beamlink.estimation import interpolate_pilots, linear_interpolation_matrix, ls_estimate
from beamlink.ofdm import apply_frequency_channel, pilot_mask, pilot_subcarrier_idx, pilot_values

N_OFDM_SYMBOLS = 14
N_SUBCARRIERS = 256
PILOT_SPACING = 4
NOISE_VARIANCE = 0.1
N_REALIZATIONS = 4000
BIAS_SIGMAS = 5.0
VARIANCE_RELATIVE_TOLERANCE = 0.05


def test_linear_interpolation_is_exact_for_a_linear_channel() -> None:
    pilots = pilot_subcarrier_idx(n_subcarriers=N_SUBCARRIERS, pilot_spacing=PILOT_SPACING, pilot_offset=0)
    interpolation = linear_interpolation_matrix(pilots, n_subcarriers=N_SUBCARRIERS)
    subcarrier = np.arange(N_SUBCARRIERS, dtype=np.float32)
    linear_channel = ((0.3 + 0.1j) * subcarrier + 2.0).astype(np.complex64)
    interpolated = interpolate_pilots(linear_channel[pilots], interpolation)
    assert np.allclose(interpolated, linear_channel, rtol=1e-5, atol=1e-4)


def test_ls_estimator_is_unbiased() -> None:
    rng = np.random.default_rng(5)
    paths = synthetic_paths(
        rng=rng,
        n_positions=1,
        n_paths=SYNTHETIC_N_PATHS,
        max_delay_s=SYNTHETIC_MAX_DELAY_S,
        delay_decay_s=SYNTHETIC_DELAY_DECAY_S,
        azimuth_span=SYNTHETIC_AZIMUTH_SPAN,
        zenith_spread=SYNTHETIC_ZENITH_SPREAD,
        los_power_ratio=None,
    )
    steering = ula_steering_vectors(
        paths.path_azimuth, paths.path_zenith, n_antennas=1, spacing_wavelengths=0.5, boresight_azimuth=0.0
    )
    cfr = paths_to_cfr(
        paths.path_gain,
        paths.path_delay_s,
        steering,
        n_subcarriers=N_SUBCARRIERS,
        subcarrier_spacing_hz=120e3,
    )[0, 0]
    mask = pilot_mask(
        n_ofdm_symbols=N_OFDM_SYMBOLS,
        n_subcarriers=N_SUBCARRIERS,
        pilot_symbol_indices=(0,),
        pilot_spacing=PILOT_SPACING,
        pilot_offset=0,
    )
    known_pilots = pilot_values(mask, rng=rng)
    grid = np.zeros((N_REALIZATIONS, N_OFDM_SYMBOLS, N_SUBCARRIERS), dtype=np.complex64)
    grid[:, mask] = known_pilots
    received = apply_frequency_channel(grid, cfr, noise_variance=NOISE_VARIANCE, rng=rng)
    pilots = pilot_subcarrier_idx(n_subcarriers=N_SUBCARRIERS, pilot_spacing=PILOT_SPACING, pilot_offset=0)
    estimation_error = ls_estimate(received, known_pilots, mask) - cfr[pilots]
    bias_limit = BIAS_SIGMAS * np.sqrt(NOISE_VARIANCE / N_REALIZATIONS)
    assert np.max(np.abs(estimation_error.mean(axis=0))) < bias_limit
    assert np.isclose(
        np.mean(np.abs(estimation_error) ** 2), NOISE_VARIANCE, rtol=VARIANCE_RELATIVE_TOLERANCE
    )
