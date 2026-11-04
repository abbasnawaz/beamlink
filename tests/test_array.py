import numpy as np

from beamlink.array import (
    beam_pattern,
    steered_beam_weights,
    ula_element_positions,
    ula_steering_vectors,
)

HALF_WAVELENGTH = 0.5
PATTERN_GRID_POINTS = 3601
PEAK_TOLERANCE = np.deg2rad(0.06)


def test_steering_vectors_have_unit_modulus_and_canonical_shape() -> None:
    azimuth = np.array([0.3, -1.0, 0.7], dtype=np.float32)
    steering = ula_steering_vectors(
        azimuth, np.pi / 2, n_antennas=8, spacing_wavelengths=HALF_WAVELENGTH, boresight_azimuth=0.0
    )
    assert steering.shape == (8, 3)
    assert steering.dtype == np.complex64
    assert np.allclose(np.abs(steering), 1.0)


def test_broadside_steering_is_all_in_phase() -> None:
    steering = ula_steering_vectors(
        0.4, np.pi / 2, n_antennas=8, spacing_wavelengths=HALF_WAVELENGTH, boresight_azimuth=0.4
    )
    assert np.allclose(steering[:, 0], 1.0, atol=1e-6)


def test_steering_matches_plane_wave_geometry() -> None:
    n_antennas, boresight, azimuth, zenith = 6, 0.4, 0.9, 1.3
    wavelength_m = 1.0
    positions_m = ula_element_positions(
        n_antennas=n_antennas, spacing_m=HALF_WAVELENGTH * wavelength_m, boresight_azimuth=boresight
    )
    direction = np.array(
        [np.sin(zenith) * np.cos(azimuth), np.sin(zenith) * np.sin(azimuth), np.cos(zenith)], dtype=np.float32
    )
    expected = np.exp(1j * 2 * np.pi / wavelength_m * positions_m @ direction)
    steering = ula_steering_vectors(
        azimuth,
        zenith,
        n_antennas=n_antennas,
        spacing_wavelengths=HALF_WAVELENGTH,
        boresight_azimuth=boresight,
    )
    assert np.allclose(steering[:, 0], expected, atol=1e-5)


def test_steered_beam_pattern_peaks_at_target_with_full_array_gain() -> None:
    n_antennas = 16
    grid = np.linspace(-np.pi / 2, np.pi / 2, PATTERN_GRID_POINTS, dtype=np.float32)
    for target_deg in (-50.0, -20.0, 0.0, 15.0, 45.0):
        target = float(np.deg2rad(target_deg))
        weights = steered_beam_weights(
            target,
            np.pi / 2,
            n_antennas=n_antennas,
            spacing_wavelengths=HALF_WAVELENGTH,
            boresight_azimuth=0.0,
        )
        pattern = beam_pattern(
            weights, grid, zenith=np.pi / 2, spacing_wavelengths=HALF_WAVELENGTH, boresight_azimuth=0.0
        )
        assert abs(grid[np.argmax(pattern)] - target) < PEAK_TOLERANCE
        assert np.isclose(pattern.max(), n_antennas, rtol=1e-4)
