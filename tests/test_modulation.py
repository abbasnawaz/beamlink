import numpy as np

from beamlink.config import Modulation
from beamlink.modulation import (
    bits_per_symbol,
    constellation,
    demap_hard,
    map_bits,
    max_log_likelihood_ratios,
)

N_SYMBOLS = 1800
NOISE_SCALE = 0.05


def test_qam_average_energy_is_one() -> None:
    for modulation in Modulation:
        assert np.isclose(np.mean(np.abs(constellation(modulation)) ** 2), 1.0, atol=1e-6)


def test_qam_round_trip_is_lossless() -> None:
    rng = np.random.default_rng(0)
    for modulation in Modulation:
        bits = rng.integers(0, 2, size=(3, bits_per_symbol(modulation) * N_SYMBOLS), dtype=np.uint8)
        symbols = map_bits(bits, modulation=modulation)
        assert symbols.dtype == np.complex64
        assert np.array_equal(demap_hard(symbols, modulation=modulation), bits)


def test_gray_neighbours_differ_by_one_bit() -> None:
    for modulation in Modulation:
        points = constellation(modulation)
        distance = np.abs(points[:, np.newaxis] - points[np.newaxis, :])
        minimum = np.min(distance[distance > 1e-6])
        first_neighbour, second_neighbour = np.nonzero(np.isclose(distance, minimum, rtol=1e-4))
        differing = first_neighbour ^ second_neighbour
        bit_values = (differing[:, np.newaxis] >> np.arange(bits_per_symbol(modulation))) & 1
        assert np.all(bit_values.sum(axis=-1) == 1)


def test_likelihood_ratio_sign_agrees_with_hard_decision() -> None:
    rng = np.random.default_rng(1)
    for modulation in Modulation:
        bits = rng.integers(0, 2, size=bits_per_symbol(modulation) * N_SYMBOLS, dtype=np.uint8)
        noise = NOISE_SCALE * (rng.standard_normal(N_SYMBOLS) + 1j * rng.standard_normal(N_SYMBOLS))
        received = (map_bits(bits, modulation=modulation) + noise).astype(np.complex64)
        ratios = max_log_likelihood_ratios(received, noise_variance=2 * NOISE_SCALE**2, modulation=modulation)
        assert np.array_equal((ratios < 0).astype(np.uint8), demap_hard(received, modulation=modulation))
