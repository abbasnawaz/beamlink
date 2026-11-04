import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]
ComplexArray = npt.NDArray[np.complex64]


def subcarrier_frequencies_hz(*, n_subcarriers: int, subcarrier_spacing_hz: float) -> FloatArray:
    subcarrier_offset = np.arange(n_subcarriers, dtype=np.float32) - np.float32(n_subcarriers // 2)
    return subcarrier_offset * np.float32(subcarrier_spacing_hz)


def time_aligned_delays(path_delay_s: FloatArray, path_valid: npt.NDArray[np.bool_]) -> FloatArray:
    first_arrival_s = np.min(np.where(path_valid, path_delay_s, np.inf), axis=-1, keepdims=True)
    return np.where(path_valid, path_delay_s - first_arrival_s, 0.0).astype(np.float32)


def paths_to_cfr(
    path_gain: ComplexArray,
    path_delay_s: FloatArray,
    steering: ComplexArray,
    *,
    n_subcarriers: int,
    subcarrier_spacing_hz: float,
) -> ComplexArray:
    frequencies_hz = subcarrier_frequencies_hz(
        n_subcarriers=n_subcarriers, subcarrier_spacing_hz=subcarrier_spacing_hz
    )
    delay_phase = np.exp(-2j * np.pi * np.einsum("...p,k->...pk", path_delay_s, frequencies_hz))
    return np.einsum("...p,...np,...pk->...nk", path_gain, steering, delay_phase).astype(np.complex64)


def n_cir_taps(
    path_delay_s: FloatArray,
    *,
    sample_rate_hz: float,
    n_precursor_taps: int,
    n_postcursor_taps: int,
) -> int:
    max_delay_samples = float(np.max(path_delay_s)) * sample_rate_hz
    return n_precursor_taps + int(np.ceil(max_delay_samples)) + n_postcursor_taps + 1


def paths_to_cir(
    path_gain: ComplexArray,
    path_delay_s: FloatArray,
    steering: ComplexArray,
    *,
    sample_rate_hz: float,
    n_precursor_taps: int,
    n_taps: int,
) -> ComplexArray:
    tap_idx = np.arange(n_taps, dtype=np.float32)
    delay_samples = path_delay_s * np.float32(sample_rate_hz)
    sinc_kernel = np.sinc(tap_idx - np.float32(n_precursor_taps) - delay_samples[..., np.newaxis])
    return np.einsum("...p,...np,...pm->...nm", path_gain, steering, sinc_kernel).astype(np.complex64)


def cir_to_cfr(cir: ComplexArray, *, n_subcarriers: int) -> ComplexArray:
    return np.fft.fftshift(np.fft.fft(cir, n=n_subcarriers, axis=-1), axes=-1).astype(np.complex64)


def unit_power_gain(path_gain: ComplexArray, cfr: ComplexArray) -> ComplexArray:
    mean_power = np.mean(np.abs(cfr) ** 2, axis=(-2, -1))
    return (path_gain / np.sqrt(mean_power)[..., np.newaxis]).astype(np.complex64)


def power_delay_profile(path_gain: ComplexArray, path_delay_s: FloatArray) -> tuple[FloatArray, FloatArray]:
    order = np.argsort(path_delay_s, axis=-1)
    sorted_delay_s = np.take_along_axis(path_delay_s, order, axis=-1)
    sorted_power = np.take_along_axis(np.abs(path_gain) ** 2, order, axis=-1)
    return sorted_delay_s.astype(np.float32), sorted_power.astype(np.float32)


def rms_delay_spread_s(path_gain: ComplexArray, path_delay_s: FloatArray) -> FloatArray:
    path_power = np.abs(path_gain) ** 2
    weights = path_power / np.sum(path_power, axis=-1, keepdims=True)
    mean_delay_s = np.sum(weights * path_delay_s, axis=-1)
    mean_square_delay = np.sum(weights * path_delay_s**2, axis=-1)
    return np.sqrt(np.maximum(mean_square_delay - mean_delay_s**2, 0.0)).astype(np.float32)
