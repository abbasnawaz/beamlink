import numpy as np
import numpy.typing as npt

from beamlink.config import DBM_PER_DBW, THERMAL_NOISE_DBM_PER_HZ

FloatArray = npt.NDArray[np.float32]


def db_to_linear(value_db: FloatArray | float) -> FloatArray:
    return np.power(np.float32(10.0), np.asarray(value_db, dtype=np.float32) / np.float32(10.0))


def linear_to_db(value: FloatArray | float) -> FloatArray:
    linear = np.asarray(value, dtype=np.float32)
    return np.float32(10.0) * np.log10(np.maximum(linear, np.finfo(np.float32).tiny))


def dbm_to_watt(power_dbm: FloatArray | float) -> FloatArray:
    return db_to_linear(np.asarray(power_dbm, dtype=np.float32) - np.float32(DBM_PER_DBW))


def watt_to_dbm(power_watt: FloatArray | float) -> FloatArray:
    return linear_to_db(power_watt) + np.float32(DBM_PER_DBW)


def noise_variance_for_snr(snr_db: float) -> float:
    return float(1.0 / db_to_linear(snr_db))


def noise_power_watt(*, bandwidth_hz: float, noise_figure_db: float) -> float:
    noise_dbm = THERMAL_NOISE_DBM_PER_HZ + float(linear_to_db(bandwidth_hz)) + noise_figure_db
    return float(dbm_to_watt(noise_dbm))


def received_snr_db(
    channel_gain_db: FloatArray | float, *, tx_power_dbm: float, bandwidth_hz: float, noise_figure_db: float
) -> FloatArray:
    noise_dbm = float(
        watt_to_dbm(noise_power_watt(bandwidth_hz=bandwidth_hz, noise_figure_db=noise_figure_db))
    )
    return (np.asarray(channel_gain_db, dtype=np.float32) + np.float32(tx_power_dbm - noise_dbm)).astype(
        np.float32
    )


def noise_variance_for_tx_power(*, tx_power_dbm: float, bandwidth_hz: float, noise_figure_db: float) -> float:
    noise_watt = noise_power_watt(bandwidth_hz=bandwidth_hz, noise_figure_db=noise_figure_db)
    return noise_watt / float(dbm_to_watt(tx_power_dbm))
