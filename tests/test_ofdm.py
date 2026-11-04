import numpy as np
import numpy.typing as npt

from beamlink.config import Modulation
from beamlink.modulation import bits_per_symbol, demap_hard, map_bits
from beamlink.ofdm import (
    apply_frequency_channel,
    apply_time_channel,
    extract_data_symbols,
    map_resource_grid,
    ofdm_demodulate,
    ofdm_modulate,
    pilot_mask,
    pilot_values,
)

N_OFDM_SYMBOLS = 14
N_SUBCARRIERS = 256
CYCLIC_PREFIX_LENGTH = 18
PILOT_SPACING = 4
N_ANTENNAS = 4
N_TAPS = 12
EXACT_TOLERANCE = 1e-4


def _mask() -> npt.NDArray[np.bool_]:
    return pilot_mask(
        n_ofdm_symbols=N_OFDM_SYMBOLS,
        n_subcarriers=N_SUBCARRIERS,
        pilot_symbol_indices=(0,),
        pilot_spacing=PILOT_SPACING,
        pilot_offset=0,
    )


def _grid(
    seed: int, modulation: Modulation, leading_shape: tuple[int, ...] = ()
) -> npt.NDArray[np.complex64]:
    rng = np.random.default_rng(seed)
    mask = _mask()
    n_bits = int(np.count_nonzero(~mask)) * bits_per_symbol(modulation)
    bits = rng.integers(0, 2, size=(*leading_shape, n_bits), dtype=np.uint8)
    return map_resource_grid(map_bits(bits, modulation=modulation), pilot_values(mask, rng=rng), mask)


def _demodulate(signal: npt.NDArray[np.complex64], cyclic_prefix_length: int) -> npt.NDArray[np.complex64]:
    return ofdm_demodulate(
        signal,
        n_ofdm_symbols=N_OFDM_SYMBOLS,
        n_subcarriers=N_SUBCARRIERS,
        cyclic_prefix_length=cyclic_prefix_length,
    )


def test_resource_grid_layout() -> None:
    mask = _mask()
    assert mask.shape == (N_OFDM_SYMBOLS, N_SUBCARRIERS)
    assert np.count_nonzero(mask) == N_SUBCARRIERS // PILOT_SPACING
    assert np.all(mask[1:] == 0)
    assert np.allclose(np.abs(pilot_values(mask, rng=np.random.default_rng(0))), 1.0)


def test_ofdm_loopback_without_channel_or_noise_has_no_bit_errors() -> None:
    rng = np.random.default_rng(1)
    mask = _mask()
    for modulation in Modulation:
        n_bits = int(np.count_nonzero(~mask)) * bits_per_symbol(modulation)
        bits = rng.integers(0, 2, size=n_bits, dtype=np.uint8)
        grid = map_resource_grid(map_bits(bits, modulation=modulation), pilot_values(mask, rng=rng), mask)
        received = _demodulate(
            ofdm_modulate(grid, cyclic_prefix_length=CYCLIC_PREFIX_LENGTH), CYCLIC_PREFIX_LENGTH
        )
        assert np.allclose(received, grid, atol=EXACT_TOLERANCE)
        assert np.array_equal(demap_hard(extract_data_symbols(received, mask), modulation=modulation), bits)


def test_unitary_ofdm_preserves_power() -> None:
    signal = ofdm_modulate(_grid(2, Modulation.QPSK), cyclic_prefix_length=0)
    assert np.isclose(np.mean(np.abs(signal) ** 2), 1.0, rtol=1e-4)


def test_frequency_and_time_modes_agree_when_prefix_covers_cir() -> None:
    rng = np.random.default_rng(3)
    cir = (rng.standard_normal((N_ANTENNAS, N_TAPS)) + 1j * rng.standard_normal((N_ANTENNAS, N_TAPS))).astype(
        np.complex64
    )
    weights = (
        rng.standard_normal((N_ANTENNAS, N_SUBCARRIERS))
        + 1j * rng.standard_normal((N_ANTENNAS, N_SUBCARRIERS))
    ).astype(np.complex64)
    grid = _grid(4, Modulation.QAM16)
    cyclic_prefix_length = N_TAPS - 1
    antenna_grids = np.einsum("sk,nk->nsk", grid, weights)
    antenna_signals = ofdm_modulate(antenna_grids, cyclic_prefix_length=cyclic_prefix_length)
    time_received = _demodulate(
        apply_time_channel(antenna_signals, cir, noise_variance=0.0, rng=rng), cyclic_prefix_length
    )
    cfr = np.fft.fftshift(np.fft.fft(cir, n=N_SUBCARRIERS, axis=-1), axes=-1)
    h_eff = np.sum(cfr * weights, axis=0).astype(np.complex64)
    freq_received = apply_frequency_channel(grid, h_eff, noise_variance=0.0, rng=rng)
    assert np.allclose(time_received, freq_received, atol=EXACT_TOLERANCE * np.abs(freq_received).max())
