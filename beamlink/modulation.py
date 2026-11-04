import numpy as np
import numpy.typing as npt

from beamlink.config import BITS_PER_SYMBOL, Modulation


def bits_per_symbol(modulation: Modulation) -> int:
    return BITS_PER_SYMBOL[modulation]


def _levels_per_axis(modulation: Modulation) -> int:
    return 2 ** (bits_per_symbol(modulation) // 2)


def _energy_scale(modulation: Modulation) -> np.float32:
    constellation_size = 2 ** bits_per_symbol(modulation)
    return np.float32(np.sqrt(2.0 * (constellation_size - 1) / 3.0))


def _binary_to_gray(level_idx: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    return level_idx ^ (level_idx >> 1)


def constellation(modulation: Modulation) -> npt.NDArray[np.complex64]:
    bits_per_axis = bits_per_symbol(modulation) // 2
    levels = _levels_per_axis(modulation)
    level_idx = np.arange(levels, dtype=np.int64)
    amplitude_by_gray_code = np.empty(levels, dtype=np.float32)
    amplitude_by_gray_code[_binary_to_gray(level_idx)] = 2 * level_idx - (levels - 1)
    labels = np.arange(2 ** bits_per_symbol(modulation), dtype=np.int64)
    in_phase = amplitude_by_gray_code[labels >> bits_per_axis]
    quadrature = amplitude_by_gray_code[labels & (levels - 1)]
    return ((in_phase + 1j * quadrature) / _energy_scale(modulation)).astype(np.complex64)


def _bits_to_labels(bits: npt.NDArray[np.uint8], bits_per_label: int) -> npt.NDArray[np.int64]:
    grouped = bits.reshape(*bits.shape[:-1], -1, bits_per_label).astype(np.int64)
    weights = 1 << np.arange(bits_per_label - 1, -1, -1, dtype=np.int64)
    return np.einsum("...sb,b->...s", grouped, weights)


def _labels_to_bits(labels: npt.NDArray[np.int64], bits_per_label: int) -> npt.NDArray[np.uint8]:
    shifts = np.arange(bits_per_label - 1, -1, -1, dtype=np.int64)
    bits = (labels[..., np.newaxis] >> shifts) & 1
    return bits.reshape(*labels.shape[:-1], -1).astype(np.uint8)


def map_bits(bits: npt.NDArray[np.uint8], *, modulation: Modulation) -> npt.NDArray[np.complex64]:
    return constellation(modulation)[_bits_to_labels(bits, bits_per_symbol(modulation))]


def _nearest_gray_code(axis_values: npt.NDArray[np.float32], levels: int) -> npt.NDArray[np.int64]:
    level_idx = np.clip(np.rint((axis_values + (levels - 1)) / 2.0), 0, levels - 1).astype(np.int64)
    return _binary_to_gray(level_idx)


def demap_hard(symbols: npt.NDArray[np.complex64], *, modulation: Modulation) -> npt.NDArray[np.uint8]:
    bits_per_axis = bits_per_symbol(modulation) // 2
    levels = _levels_per_axis(modulation)
    scaled = symbols * _energy_scale(modulation)
    in_phase_code = _nearest_gray_code(scaled.real, levels)
    quadrature_code = _nearest_gray_code(scaled.imag, levels)
    labels = (in_phase_code << bits_per_axis) | quadrature_code
    return _labels_to_bits(labels, bits_per_symbol(modulation))


def max_log_likelihood_ratios(
    symbols: npt.NDArray[np.complex64],
    *,
    noise_variance: npt.NDArray[np.float32] | float,
    modulation: Modulation,
) -> npt.NDArray[np.float32]:
    points = constellation(modulation)
    n_bits = bits_per_symbol(modulation)
    point_bits = _labels_to_bits(np.arange(points.size, dtype=np.int64)[:, np.newaxis], n_bits).astype(bool)
    squared_distance = np.abs(symbols[..., np.newaxis] - points) ** 2
    distance_per_bit = squared_distance[..., np.newaxis, :]
    nearest_with_zero = np.where(~point_bits.T, distance_per_bit, np.inf).min(axis=-1)
    nearest_with_one = np.where(point_bits.T, distance_per_bit, np.inf).min(axis=-1)
    ratios = (nearest_with_one - nearest_with_zero) / np.asarray(noise_variance)[..., np.newaxis]
    return ratios.reshape(*symbols.shape[:-1], -1).astype(np.float32)
