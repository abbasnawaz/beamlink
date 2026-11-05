import numpy as np
import numpy.typing as npt
import pytest

from beamlink.array import beam_pattern, ula_steering_vectors
from beamlink.beamforming import (
    beam_sector,
    beamforming_gain,
    codebook_beam_azimuths,
    dft_codebook,
    effective_channel,
    sector_power_shares,
    sectored_codebook,
    serving_sector,
    spread_over_subcarriers,
    strongest_beam_idx,
    subcarrier_mrt_weights,
    sweep_beams,
    wideband_mrt_weights,
    wideband_power,
)
from beamlink.channel import paths_to_cfr
from beamlink.config import (
    SYNTHETIC_AZIMUTH_SPAN,
    SYNTHETIC_DELAY_DECAY_S,
    SYNTHETIC_MAX_DELAY_S,
    SYNTHETIC_N_PATHS,
    SYNTHETIC_ZENITH_SPREAD,
)
from beamlink.data import synthetic_paths

HALF_WAVELENGTH = 0.5
N_SUBCARRIERS = 256
SUBCARRIER_SPACING_HZ = 120e3
ANTENNA_COUNTS = (4, 8, 16, 32)
N_RANDOM_CHANNELS = 20
PATTERN_GRID_POINTS = 18001
PEAK_TOLERANCE = np.deg2rad(0.05)
END_FIRE_COSINE = 0.999
POWER_RELATIVE_TOLERANCE = 1e-4


def _codebook(
    n_antennas: int, oversampling: int = 1, sector: tuple[float, ...] | None = None
) -> npt.NDArray[np.complex64]:
    return dft_codebook(
        n_antennas=n_antennas, oversampling=oversampling, spacing_wavelengths=HALF_WAVELENGTH, sector=sector
    )


def _beam_azimuths(
    n_antennas: int, oversampling: int = 1, sector: tuple[float, ...] | None = None
) -> npt.NDArray[np.float32]:
    return codebook_beam_azimuths(
        n_antennas=n_antennas,
        oversampling=oversampling,
        spacing_wavelengths=HALF_WAVELENGTH,
        sector=sector,
        boresight_azimuth=0.0,
    )


def _random_cfr(seed: int, n_antennas: int, n_positions: int) -> npt.NDArray[np.complex64]:
    paths = synthetic_paths(
        rng=np.random.default_rng(seed),
        n_positions=n_positions,
        n_paths=SYNTHETIC_N_PATHS,
        max_delay_s=SYNTHETIC_MAX_DELAY_S,
        delay_decay_s=SYNTHETIC_DELAY_DECAY_S,
        azimuth_span=SYNTHETIC_AZIMUTH_SPAN,
        zenith_spread=SYNTHETIC_ZENITH_SPREAD,
        los_power_ratio=None,
    )
    steering = ula_steering_vectors(
        paths.path_azimuth,
        paths.path_zenith,
        n_antennas=n_antennas,
        spacing_wavelengths=HALF_WAVELENGTH,
        boresight_azimuth=0.0,
    )
    return paths_to_cfr(
        paths.path_gain,
        paths.path_delay_s,
        steering,
        n_subcarriers=N_SUBCARRIERS,
        subcarrier_spacing_hz=SUBCARRIER_SPACING_HZ,
    )


def _single_path_cfr(azimuth: float, n_antennas: int) -> npt.NDArray[np.complex64]:
    steering = ula_steering_vectors(
        np.array([azimuth], dtype=np.float32),
        np.pi / 2,
        n_antennas=n_antennas,
        spacing_wavelengths=HALF_WAVELENGTH,
        boresight_azimuth=0.0,
    )
    return paths_to_cfr(
        np.ones(1, dtype=np.complex64),
        np.array([50e-9], dtype=np.float32),
        steering,
        n_subcarriers=N_SUBCARRIERS,
        subcarrier_spacing_hz=SUBCARRIER_SPACING_HZ,
    )


def test_dft_codebook_is_orthonormal() -> None:
    for n_antennas in ANTENNA_COUNTS:
        codebook = _codebook(n_antennas)
        assert codebook.shape == (n_antennas, n_antennas)
        assert np.allclose(codebook @ codebook.conj().T, np.eye(n_antennas), atol=1e-5)


def test_dft_beam_pattern_peaks_at_its_azimuth() -> None:
    grid = np.linspace(-np.pi / 2, np.pi / 2, PATTERN_GRID_POINTS, dtype=np.float32)
    for n_antennas, oversampling in ((8, 1), (8, 4), (16, 2)):
        codebook = _codebook(n_antennas, oversampling)
        azimuths = _beam_azimuths(n_antennas, oversampling)
        steering = ula_steering_vectors(
            grid, np.pi / 2, n_antennas=n_antennas, spacing_wavelengths=HALF_WAVELENGTH, boresight_azimuth=0.0
        )
        patterns = beam_pattern(codebook, steering)
        peaks = grid[np.argmax(patterns, axis=-1)]
        inside = np.abs(np.sin(azimuths)) < END_FIRE_COSINE
        assert np.all(np.abs(peaks[inside] - azimuths[inside]) < PEAK_TOLERANCE)
        assert np.allclose(patterns.max(axis=-1), n_antennas, rtol=POWER_RELATIVE_TOLERANCE)


def test_oversampled_and_sector_codebooks_have_unit_norm_beams() -> None:
    codebook = _codebook(8, oversampling=4)
    assert codebook.shape == (32, 8)
    assert np.allclose(np.linalg.norm(codebook, axis=-1), 1.0, atol=1e-6)
    sector = (-np.pi / 3, np.pi / 3)
    azimuths = _beam_azimuths(8, oversampling=2, sector=sector)
    assert azimuths.shape == (16,)
    assert np.isclose(azimuths[0], sector[0]) and np.isclose(azimuths[-1], sector[1])


def test_sweep_matches_direct_power_computation() -> None:
    cfr = _random_cfr(0, 8, n_positions=1)[0]
    codebook = _codebook(8, oversampling=2)
    direct = np.sum(np.abs(np.einsum("bn,nk->bk", codebook, cfr)) ** 2, axis=-1)
    assert np.allclose(sweep_beams(cfr, codebook), direct, rtol=POWER_RELATIVE_TOLERANCE)


def test_sweep_is_batched_over_positions() -> None:
    cfr = _random_cfr(1, 8, n_positions=5)
    codebook = _codebook(8)
    batched = sweep_beams(cfr, codebook)
    assert batched.shape == (5, 8)
    assert np.allclose(batched[3], sweep_beams(cfr[3], codebook))


def test_wideband_mrt_beats_every_codebook_beam_and_subcarrier_mrt_beats_both() -> None:
    for n_antennas in ANTENNA_COUNTS:
        cfr = _random_cfr(n_antennas, n_antennas, n_positions=N_RANDOM_CHANNELS)
        weights = wideband_mrt_weights(cfr)
        assert np.allclose(np.linalg.norm(weights, axis=-1), 1.0, atol=1e-5)
        mrt_power = wideband_power(
            effective_channel(cfr, spread_over_subcarriers(weights, n_subcarriers=N_SUBCARRIERS))
        )
        codebook_power = sweep_beams(cfr, _codebook(n_antennas, oversampling=4))
        assert np.all(codebook_power <= mrt_power[:, np.newaxis] * (1 + POWER_RELATIVE_TOLERANCE))
        subcarrier_power = wideband_power(effective_channel(cfr, subcarrier_mrt_weights(cfr)))
        assert np.all(subcarrier_power >= mrt_power * (1 - POWER_RELATIVE_TOLERANCE))


def test_single_path_gives_full_array_gain_and_nearest_beam() -> None:
    n_antennas = 16
    target = float(np.deg2rad(20.0))
    cfr = _single_path_cfr(target, n_antennas)
    weights = spread_over_subcarriers(wideband_mrt_weights(cfr), n_subcarriers=N_SUBCARRIERS)
    mrt_gain = beamforming_gain(wideband_power(effective_channel(cfr, weights)), cfr)
    assert np.isclose(mrt_gain, n_antennas, rtol=POWER_RELATIVE_TOLERANCE)
    subcarrier_gain = beamforming_gain(
        wideband_power(effective_channel(cfr, subcarrier_mrt_weights(cfr))), cfr
    )
    assert np.isclose(subcarrier_gain, n_antennas, rtol=POWER_RELATIVE_TOLERANCE)
    beam_power = sweep_beams(cfr, _codebook(n_antennas))
    best = strongest_beam_idx(beam_power, candidate_idx=np.arange(n_antennas))
    assert best == np.argmin(np.abs(np.sin(_beam_azimuths(n_antennas)) - np.sin(target)))


def test_subcarrier_mrt_effective_channel_is_antenna_norm() -> None:
    cfr = _random_cfr(2, 8, n_positions=1)[0]
    h_eff = effective_channel(cfr, subcarrier_mrt_weights(cfr))
    assert np.allclose(h_eff, np.linalg.norm(cfr, axis=0), atol=1e-5)


def test_strongest_beam_only_considers_candidates() -> None:
    beam_power = np.array([1.0, 9.0, 3.0, 5.0], dtype=np.float32)
    assert strongest_beam_idx(beam_power, candidate_idx=np.array([0, 2, 3])) == 3
    assert strongest_beam_idx(beam_power, candidate_idx=np.arange(4)) == 1


def test_sectored_codebook_beams_are_unit_norm_and_live_in_their_own_sector() -> None:
    n_antennas, n_sectors, oversampling = 8, 4, 2
    codebook = sectored_codebook(
        n_antennas=n_antennas,
        n_sectors=n_sectors,
        oversampling=oversampling,
        spacing_wavelengths=HALF_WAVELENGTH,
        sector=None,
    )
    n_beams_per_sector = oversampling * n_antennas
    assert codebook.shape == (n_sectors * n_beams_per_sector, n_sectors * n_antennas)
    assert np.allclose(np.linalg.norm(codebook, axis=-1), 1.0, atol=1e-6)
    for beam_idx, weights in enumerate(codebook):
        sector_idx, beam_in_sector = beam_sector(beam_idx, n_beams_per_sector=n_beams_per_sector)
        own_rows = slice(sector_idx * n_antennas, (sector_idx + 1) * n_antennas)
        assert np.count_nonzero(np.delete(weights, np.arange(n_sectors * n_antennas)[own_rows])) == 0
        assert np.allclose(weights[own_rows], _codebook(n_antennas, oversampling)[beam_in_sector])
        assert serving_sector(weights, n_sectors=n_sectors) == sector_idx


def test_beam_sector_maps_global_indices_to_sector_and_local_beam() -> None:
    assert beam_sector(0, n_beams_per_sector=8) == (0, 0)
    assert beam_sector(13, n_beams_per_sector=8) == (1, 5)
    assert beam_sector(31, n_beams_per_sector=8) == (3, 7)


def test_serving_sector_of_per_subcarrier_weights_follows_the_energy() -> None:
    weights = np.zeros((3 * 4, N_SUBCARRIERS), dtype=np.complex64)
    weights[8:12] = 0.5
    weights[0:4, :10] = 0.1
    assert serving_sector(weights, n_sectors=3) == 2


def test_sector_power_shares_sum_to_one_and_are_whole_for_codebook_beams() -> None:
    n_antennas, n_sectors = 8, 4
    codebook = sectored_codebook(
        n_antennas=n_antennas,
        n_sectors=n_sectors,
        oversampling=2,
        spacing_wavelengths=HALF_WAVELENGTH,
        sector=None,
    )
    for beam_idx in (0, 21, 40, 63):
        shares = sector_power_shares(codebook[beam_idx], n_sectors=n_sectors)
        assert shares[beam_idx // (2 * n_antennas)] == pytest.approx(1.0)
    weights = np.zeros((n_sectors * n_antennas, N_SUBCARRIERS), dtype=np.complex64)
    weights[:n_antennas] = 0.5
    weights[2 * n_antennas : 3 * n_antennas] = 1.0
    shares = sector_power_shares(weights, n_sectors=n_sectors)
    assert shares.sum() == pytest.approx(1.0)
    assert shares[2] == pytest.approx(0.8)
    assert shares[0] == pytest.approx(0.2)
