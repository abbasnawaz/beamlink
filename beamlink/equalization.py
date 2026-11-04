import numpy as np
import numpy.typing as npt

ComplexArray = npt.NDArray[np.complex64]
FloatArray = npt.NDArray[np.float32]


def _channel_per_resource_element(received_grid: ComplexArray, cfr: ComplexArray) -> ComplexArray:
    return np.broadcast_to(cfr[..., np.newaxis, :], received_grid.shape)


def zf_equalize(received_grid: ComplexArray, cfr: ComplexArray) -> tuple[ComplexArray, FloatArray]:
    channel = _channel_per_resource_element(received_grid, cfr)
    channel_power = np.maximum(np.abs(channel) ** 2, np.finfo(np.float32).tiny)
    equalized = received_grid * np.conj(channel) / channel_power
    return equalized.astype(np.complex64), (1.0 / channel_power).astype(np.float32)


def mmse_equalize(
    received_grid: ComplexArray, cfr: ComplexArray, *, noise_variance: float
) -> tuple[ComplexArray, FloatArray]:
    channel = _channel_per_resource_element(received_grid, cfr)
    channel_power = np.abs(channel) ** 2
    mmse_estimate = np.conj(channel) * received_grid / (channel_power + noise_variance)
    bias = np.maximum(channel_power / (channel_power + noise_variance), np.finfo(np.float32).tiny)
    unbiased = mmse_estimate / bias
    post_noise_variance = noise_variance / np.maximum(channel_power, np.finfo(np.float32).tiny)
    return unbiased.astype(np.complex64), post_noise_variance.astype(np.float32)
