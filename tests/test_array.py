import numpy as np
import pytest

from beamlink.array import (
    beam_pattern,
    element_field_pattern,
    sectored_array_response,
    steered_beam_weights,
    ula_element_positions,
    ula_steering_vectors,
)
from beamlink.linkbudget import linear_to_db

HALF_WAVELENGTH = 0.5
PATTERN_GRID_POINTS = 3601
PEAK_TOLERANCE = np.deg2rad(0.06)
PEAK_GAIN_DBI = 8.0
BEAMWIDTH = np.deg2rad(65.0)
FRONT_TO_BACK_DB = 30.0
DOWNTILT = np.deg2rad(10.0)
SECTOR_BORESIGHTS = (0.0, np.pi / 2, np.pi, 3 * np.pi / 2)
GAIN_TOLERANCE_DB = 1e-3


def _element_gain_dbi(azimuth: float, zenith: float, boresight_azimuth: float, downtilt: float) -> float:
    field = element_field_pattern(
        azimuth,
        zenith,
        boresight_azimuth=boresight_azimuth,
        downtilt=downtilt,
        peak_gain_dbi=PEAK_GAIN_DBI,
        beamwidth=BEAMWIDTH,
        front_to_back_db=FRONT_TO_BACK_DB,
    )
    return float(linear_to_db(field**2))


def _sectored(azimuth: np.ndarray | float, zenith: np.ndarray | float, n_antennas: int) -> np.ndarray:
    return sectored_array_response(
        azimuth,
        zenith,
        n_antennas=n_antennas,
        spacing_wavelengths=HALF_WAVELENGTH,
        sector_boresight_azimuths=SECTOR_BORESIGHTS,
        downtilt=DOWNTILT,
        peak_gain_dbi=PEAK_GAIN_DBI,
        beamwidth=BEAMWIDTH,
        front_to_back_db=FRONT_TO_BACK_DB,
    )


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
        steering = ula_steering_vectors(
            grid, np.pi / 2, n_antennas=n_antennas, spacing_wavelengths=HALF_WAVELENGTH, boresight_azimuth=0.0
        )
        pattern = beam_pattern(weights, steering)
        assert abs(grid[np.argmax(pattern)] - target) < PEAK_TOLERANCE
        assert np.isclose(pattern.max(), n_antennas, rtol=1e-4)


def test_element_peaks_on_boresight_and_loses_3_db_at_half_beamwidth() -> None:
    for boresight_azimuth in SECTOR_BORESIGHTS:
        assert _element_gain_dbi(boresight_azimuth, np.pi / 2, boresight_azimuth, 0.0) == pytest.approx(
            PEAK_GAIN_DBI, abs=GAIN_TOLERANCE_DB
        )
        for offset in (-BEAMWIDTH / 2, BEAMWIDTH / 2):
            gain_dbi = _element_gain_dbi(boresight_azimuth + offset, np.pi / 2, boresight_azimuth, 0.0)
            assert gain_dbi == pytest.approx(PEAK_GAIN_DBI - 3.0, abs=GAIN_TOLERANCE_DB)


def test_element_is_clipped_at_the_front_to_back_ratio_behind_the_panel() -> None:
    for behind in (np.pi, 0.8 * np.pi, -0.7 * np.pi):
        gain_dbi = _element_gain_dbi(behind, np.pi / 2, 0.0, 0.0)
        assert gain_dbi == pytest.approx(PEAK_GAIN_DBI - FRONT_TO_BACK_DB, abs=GAIN_TOLERANCE_DB)


def test_downtilt_moves_the_element_peak_below_the_horizon() -> None:
    tilted_peak = _element_gain_dbi(np.pi / 2, np.pi / 2 + DOWNTILT, np.pi / 2, DOWNTILT)
    assert tilted_peak == pytest.approx(PEAK_GAIN_DBI, abs=GAIN_TOLERANCE_DB)
    assert _element_gain_dbi(np.pi / 2, np.pi / 2, np.pi / 2, DOWNTILT) < PEAK_GAIN_DBI


def test_sectored_response_stacks_each_panel_with_its_element_field() -> None:
    n_antennas = 4
    azimuth = np.array([0.2, 1.9, -2.8], dtype=np.float32)
    zenith = np.full(3, np.pi / 2 + 0.05, dtype=np.float32)
    response = _sectored(azimuth, zenith, n_antennas)
    assert response.shape == (len(SECTOR_BORESIGHTS) * n_antennas, 3)
    assert response.dtype == np.complex64
    for sector_idx, boresight_azimuth in enumerate(SECTOR_BORESIGHTS):
        block = response[sector_idx * n_antennas : (sector_idx + 1) * n_antennas]
        steering = ula_steering_vectors(
            azimuth,
            zenith,
            n_antennas=n_antennas,
            spacing_wavelengths=HALF_WAVELENGTH,
            boresight_azimuth=boresight_azimuth,
        )
        field = element_field_pattern(
            azimuth,
            zenith,
            boresight_azimuth=boresight_azimuth,
            downtilt=DOWNTILT,
            peak_gain_dbi=PEAK_GAIN_DBI,
            beamwidth=BEAMWIDTH,
            front_to_back_db=FRONT_TO_BACK_DB,
        )
        assert np.allclose(block, steering * field, atol=1e-6)
