import numpy as np
import numpy.typing as npt

from beamlink.array import ula_response

FloatArray = npt.NDArray[np.float32]
ComplexArray = npt.NDArray[np.complex64]
IndexArray = npt.NDArray[np.int64]


def codebook_direction_cosines(
    *,
    n_antennas: int,
    oversampling: int,
    spacing_wavelengths: float,
    sector: tuple[float, ...] | None,
) -> FloatArray:
    n_beams = oversampling * n_antennas
    if sector is None:
        spatial_frequency = np.arange(n_beams, dtype=np.float32) / np.float32(n_beams) - np.float32(0.5)
        return spatial_frequency / np.float32(spacing_wavelengths)
    sector_start, sector_end = np.sin(sector)
    return np.linspace(sector_start, sector_end, n_beams, dtype=np.float32)


def codebook_beam_azimuths(
    *,
    n_antennas: int,
    oversampling: int,
    spacing_wavelengths: float,
    sector: tuple[float, ...] | None,
    boresight_azimuth: float,
) -> FloatArray:
    cosines = codebook_direction_cosines(
        n_antennas=n_antennas,
        oversampling=oversampling,
        spacing_wavelengths=spacing_wavelengths,
        sector=sector,
    )
    return (np.float32(boresight_azimuth) + np.arcsin(np.clip(cosines, -1.0, 1.0))).astype(np.float32)


def dft_codebook(
    *,
    n_antennas: int,
    oversampling: int,
    spacing_wavelengths: float,
    sector: tuple[float, ...] | None,
) -> ComplexArray:
    cosines = codebook_direction_cosines(
        n_antennas=n_antennas,
        oversampling=oversampling,
        spacing_wavelengths=spacing_wavelengths,
        sector=sector,
    )
    response = ula_response(cosines, n_antennas=n_antennas, spacing_wavelengths=spacing_wavelengths)
    return (np.conj(response).T / np.sqrt(np.float32(n_antennas))).astype(np.complex64)


def spread_over_subcarriers(beam_weights: ComplexArray, *, n_subcarriers: int) -> ComplexArray:
    return np.repeat(beam_weights[..., np.newaxis], n_subcarriers, axis=-1)


def effective_channel(cfr: ComplexArray, subcarrier_weights: ComplexArray) -> ComplexArray:
    return np.einsum("...nk,...nk->...k", cfr, subcarrier_weights).astype(np.complex64)


def codebook_effective_channels(cfr: ComplexArray, codebook: ComplexArray) -> ComplexArray:
    return np.einsum("...nk,bn->...bk", cfr, codebook).astype(np.complex64)


def wideband_power(effective_cfr: ComplexArray) -> FloatArray:
    return np.sum(np.abs(effective_cfr) ** 2, axis=-1).astype(np.float32)


def sweep_beams(cfr: ComplexArray, codebook: ComplexArray) -> FloatArray:
    return wideband_power(codebook_effective_channels(cfr, codebook))


def strongest_beam_idx(beam_power: FloatArray, *, candidate_idx: IndexArray) -> int:
    return int(candidate_idx[np.argmax(beam_power[candidate_idx])])


def wideband_mrt_weights(cfr: ComplexArray) -> ComplexArray:
    spatial_covariance = np.einsum("...nk,...mk->...nm", np.conj(cfr), cfr)
    _, eigenvectors = np.linalg.eigh(spatial_covariance)
    principal = eigenvectors[..., :, -1]
    phase_reference = np.exp(-1j * np.angle(principal[..., :1]))
    return (principal * phase_reference).astype(np.complex64)


def subcarrier_mrt_weights(cfr: ComplexArray) -> ComplexArray:
    antenna_norm = np.linalg.norm(cfr, axis=-2, keepdims=True)
    return (np.conj(cfr) / np.maximum(antenna_norm, np.finfo(np.float32).tiny)).astype(np.complex64)


def beamforming_gain(beam_power: FloatArray | float, cfr: ComplexArray) -> FloatArray:
    single_antenna_power = np.sum(np.abs(cfr) ** 2, axis=(-2, -1)) / np.float32(cfr.shape[-2])
    return (np.asarray(beam_power) / single_antenna_power).astype(np.float32)


def precode_grid(grid: ComplexArray, subcarrier_weights: ComplexArray) -> ComplexArray:
    return np.einsum("...sk,nk->...nsk", grid, subcarrier_weights).astype(np.complex64)
