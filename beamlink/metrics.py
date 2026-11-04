import numpy as np
import numpy.typing as npt
from scipy.special import erfc

from beamlink.config import BITS_PER_SYMBOL, Modulation
from beamlink.linkbudget import linear_to_db

FloatArray = npt.NDArray[np.float32]
ComplexArray = npt.NDArray[np.complex64]


def bit_error_rate(transmitted_bits: npt.NDArray[np.uint8], received_bits: npt.NDArray[np.uint8]) -> float:
    return float(np.count_nonzero(transmitted_bits != received_bits)) / transmitted_bits.size


def awgn_ber_theory(snr: FloatArray | float, *, modulation: Modulation) -> FloatArray:
    n_bits = BITS_PER_SYMBOL[modulation]
    constellation_size = 2**n_bits
    levels = int(round(np.sqrt(constellation_size)))
    bit_position = np.arange(1, n_bits // 2 + 1, dtype=np.int64)[:, np.newaxis]
    boundary_idx = np.arange(levels, dtype=np.int64)[np.newaxis, :]
    is_active = boundary_idx < (1.0 - 2.0 ** (-bit_position)) * levels
    scaled = boundary_idx * 2 ** (bit_position - 1)
    sign = (-1.0) ** (scaled // levels)
    weight = 2.0 ** (bit_position - 1) - np.floor(scaled / levels + 0.5)
    coefficient = np.where(is_active, sign * weight, 0.0) / levels
    snr_array = np.asarray(snr, dtype=np.float64)[..., np.newaxis, np.newaxis]
    argument = (2 * boundary_idx + 1) * np.sqrt(3.0 * snr_array / (2.0 * (constellation_size - 1)))
    per_bit_position = np.sum(coefficient * erfc(argument), axis=-1)
    return np.mean(per_bit_position, axis=-1).astype(np.float32)


def fading_ber_theory(effective_cfr: ComplexArray, *, noise_variance: float, modulation: Modulation) -> float:
    per_subcarrier_snr = np.abs(effective_cfr) ** 2 / noise_variance
    return float(np.mean(awgn_ber_theory(per_subcarrier_snr, modulation=modulation)))


def effective_snr(effective_cfr: ComplexArray, *, noise_variance: float) -> float:
    return float(np.mean(np.abs(effective_cfr) ** 2) / noise_variance)


def spectral_efficiency(snr: float) -> float:
    return float(np.log2(1.0 + snr))


def power_loss_db(best_power: FloatArray | float, chosen_power: FloatArray | float) -> FloatArray:
    return linear_to_db(np.asarray(best_power) / np.maximum(chosen_power, np.finfo(np.float32).tiny))


def top_k_accuracy(candidate_beams: npt.NDArray[np.int64], best_beam: npt.NDArray[np.int64]) -> float:
    return float(np.mean(np.any(candidate_beams == best_beam[..., np.newaxis], axis=-1)))


def best_candidate_power_db(
    beam_power_db: npt.NDArray[np.float32], candidate_beams: npt.NDArray[np.int64]
) -> npt.NDArray[np.float32]:
    return np.max(np.take_along_axis(beam_power_db, candidate_beams, axis=-1), axis=-1).astype(np.float32)


def top_k_power_loss_db(
    beam_power_db: npt.NDArray[np.float32], candidate_beams: npt.NDArray[np.int64]
) -> npt.NDArray[np.float32]:
    best_candidate_db = best_candidate_power_db(beam_power_db, candidate_beams)
    return (np.max(beam_power_db, axis=-1) - best_candidate_db).astype(np.float32)
