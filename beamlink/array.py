import numpy as np
import numpy.typing as npt

from beamlink.config import ELEMENT_ROLLOFF_DB
from beamlink.linkbudget import db_to_linear

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


def panel_local_angles(
    azimuth: FloatArray | float,
    zenith: FloatArray | float,
    *,
    boresight_azimuth: float,
    downtilt: float,
) -> tuple[FloatArray, FloatArray]:
    azimuth = np.asarray(azimuth, dtype=np.float32)
    zenith = np.asarray(zenith, dtype=np.float32)
    toward_x = np.sin(zenith) * np.cos(azimuth)
    toward_y = np.sin(zenith) * np.sin(azimuth)
    toward_z = np.cos(zenith)
    cos_yaw, sin_yaw = np.float32(np.cos(boresight_azimuth)), np.float32(np.sin(boresight_azimuth))
    cos_tilt, sin_tilt = np.float32(np.cos(downtilt)), np.float32(np.sin(downtilt))
    yawed_x = cos_yaw * toward_x + sin_yaw * toward_y
    local_y = -sin_yaw * toward_x + cos_yaw * toward_y
    local_x = cos_tilt * yawed_x - sin_tilt * toward_z
    local_z = sin_tilt * yawed_x + cos_tilt * toward_z
    local_azimuth = np.arctan2(local_y, local_x).astype(np.float32)
    local_zenith = np.arccos(np.clip(local_z, -1.0, 1.0)).astype(np.float32)
    return local_azimuth, local_zenith


def element_field_pattern(
    azimuth: FloatArray | float,
    zenith: FloatArray | float,
    *,
    boresight_azimuth: float,
    downtilt: float,
    peak_gain_dbi: float,
    beamwidth: float,
    front_to_back_db: float,
) -> FloatArray:
    local_azimuth, local_zenith = panel_local_angles(
        azimuth, zenith, boresight_azimuth=boresight_azimuth, downtilt=downtilt
    )
    horizontal_db = -np.minimum(ELEMENT_ROLLOFF_DB * (local_azimuth / beamwidth) ** 2, front_to_back_db)
    vertical_db = -np.minimum(
        ELEMENT_ROLLOFF_DB * ((local_zenith - np.pi / 2.0) / beamwidth) ** 2, front_to_back_db
    )
    gain_db = peak_gain_dbi - np.minimum(-(horizontal_db + vertical_db), front_to_back_db)
    return np.sqrt(db_to_linear(gain_db)).astype(np.float32)


def sectored_array_response(
    azimuth: FloatArray | float,
    zenith: FloatArray | float,
    *,
    n_antennas: int,
    spacing_wavelengths: float,
    sector_boresight_azimuths: tuple[float, ...],
    downtilt: float,
    peak_gain_dbi: float,
    beamwidth: float,
    front_to_back_db: float,
) -> ComplexArray:
    azimuth, zenith = np.broadcast_arrays(
        np.atleast_1d(np.asarray(azimuth, dtype=np.float32)), np.asarray(zenith, dtype=np.float32)
    )
    sector_blocks = []
    for boresight_azimuth in sector_boresight_azimuths:
        steering = ula_steering_vectors(
            azimuth,
            zenith,
            n_antennas=n_antennas,
            spacing_wavelengths=spacing_wavelengths,
            boresight_azimuth=boresight_azimuth,
        )
        element_field = element_field_pattern(
            azimuth,
            zenith,
            boresight_azimuth=boresight_azimuth,
            downtilt=downtilt,
            peak_gain_dbi=peak_gain_dbi,
            beamwidth=beamwidth,
            front_to_back_db=front_to_back_db,
        )
        sector_blocks.append(steering * element_field[..., np.newaxis, :])
    return np.concatenate(sector_blocks, axis=-2).astype(np.complex64)


def beam_pattern(beam_weights: ComplexArray, array_response: ComplexArray) -> FloatArray:
    return (np.abs(np.einsum("na,...n->...a", array_response, beam_weights)) ** 2).astype(np.float32)
