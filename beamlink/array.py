import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]
ComplexArray = npt.NDArray[np.complex64]


def _centered_antenna_offsets(n_antennas: int) -> FloatArray:
    return np.arange(n_antennas, dtype=np.float32) - np.float32((n_antennas - 1) / 2.0)


def ula_element_positions(*, n_antennas: int, spacing_m: float, boresight_azimuth: float) -> FloatArray:
    array_axis = np.array([-np.sin(boresight_azimuth), np.cos(boresight_azimuth), 0.0], dtype=np.float32)
    return np.einsum("n,c->nc", _centered_antenna_offsets(n_antennas) * np.float32(spacing_m), array_axis)


def direction_cosine(
    azimuth: FloatArray | float,
    zenith: FloatArray | float,
    *,
    boresight_azimuth: float,
) -> FloatArray:
    azimuth = np.asarray(azimuth, dtype=np.float32)
    zenith = np.asarray(zenith, dtype=np.float32)
    return (np.sin(zenith) * np.sin(azimuth - np.float32(boresight_azimuth))).astype(np.float32)


def ula_response(
    direction_cosines: FloatArray,
    *,
    n_antennas: int,
    spacing_wavelengths: float,
) -> ComplexArray:
    phase_per_antenna = np.float32(2.0 * np.pi * spacing_wavelengths) * direction_cosines
    phase = np.einsum("n,...p->...np", _centered_antenna_offsets(n_antennas), phase_per_antenna)
    return np.exp(1j * phase).astype(np.complex64)


def ula_steering_vectors(
    azimuth: FloatArray | float,
    zenith: FloatArray | float,
    *,
    n_antennas: int,
    spacing_wavelengths: float,
    boresight_azimuth: float,
) -> ComplexArray:
    cosines = direction_cosine(azimuth, zenith, boresight_azimuth=boresight_azimuth)
    return ula_response(
        np.atleast_1d(cosines), n_antennas=n_antennas, spacing_wavelengths=spacing_wavelengths
    )


def steered_beam_weights(
    azimuth: float,
    zenith: float,
    *,
    n_antennas: int,
    spacing_wavelengths: float,
    boresight_azimuth: float,
) -> ComplexArray:
    steering = ula_steering_vectors(
        azimuth,
        zenith,
        n_antennas=n_antennas,
        spacing_wavelengths=spacing_wavelengths,
        boresight_azimuth=boresight_azimuth,
    )
    return (np.conj(steering[:, 0]) / np.sqrt(np.float32(n_antennas))).astype(np.complex64)


def beam_pattern(
    beam_weights: ComplexArray,
    azimuth_grid: FloatArray,
    *,
    zenith: float,
    spacing_wavelengths: float,
    boresight_azimuth: float,
) -> FloatArray:
    steering = ula_steering_vectors(
        azimuth_grid,
        zenith,
        n_antennas=beam_weights.shape[-1],
        spacing_wavelengths=spacing_wavelengths,
        boresight_azimuth=boresight_azimuth,
    )
    return (np.abs(np.einsum("na,...n->...a", steering, beam_weights)) ** 2).astype(np.float32)
