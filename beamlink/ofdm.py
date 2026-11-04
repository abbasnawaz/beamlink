import numpy as np
import numpy.typing as npt

from beamlink.config import Modulation
from beamlink.modulation import constellation

ComplexArray = npt.NDArray[np.complex64]
BoolArray = npt.NDArray[np.bool_]


def pilot_subcarrier_idx(
    *, n_subcarriers: int, pilot_spacing: int, pilot_offset: int
) -> npt.NDArray[np.int64]:
    return np.arange(pilot_offset, n_subcarriers, pilot_spacing, dtype=np.int64)


def pilot_mask(
    *,
    n_ofdm_symbols: int,
    n_subcarriers: int,
    pilot_symbol_indices: tuple[int, ...],
    pilot_spacing: int,
    pilot_offset: int,
) -> BoolArray:
    is_pilot_symbol = np.isin(np.arange(n_ofdm_symbols), pilot_symbol_indices)
    is_pilot_subcarrier = np.zeros(n_subcarriers, dtype=bool)
    is_pilot_subcarrier[
        pilot_subcarrier_idx(
            n_subcarriers=n_subcarriers, pilot_spacing=pilot_spacing, pilot_offset=pilot_offset
        )
    ] = True
    return np.logical_and.outer(is_pilot_symbol, is_pilot_subcarrier)


def pilot_values(mask: BoolArray, *, rng: np.random.Generator) -> ComplexArray:
    qpsk_points = constellation(Modulation.QPSK)
    return qpsk_points[rng.integers(0, qpsk_points.size, size=int(np.count_nonzero(mask)))]


def map_resource_grid(
    data_symbols: ComplexArray, known_pilots: ComplexArray, mask: BoolArray
) -> ComplexArray:
    grid = np.zeros((*data_symbols.shape[:-1], *mask.shape), dtype=np.complex64)
    grid[..., mask] = known_pilots
    grid[..., ~mask] = data_symbols
    return grid


def extract_data_symbols(grid: ComplexArray, mask: BoolArray) -> ComplexArray:
    return grid[..., ~mask]


def ofdm_modulate(grid: ComplexArray, *, cyclic_prefix_length: int) -> ComplexArray:
    n_subcarriers = grid.shape[-1]
    unitary_scale = np.sqrt(np.float32(n_subcarriers))
    symbols_in_time = np.fft.ifft(np.fft.ifftshift(grid, axes=-1), axis=-1) * unitary_scale
    cyclic_prefix = symbols_in_time[..., n_subcarriers - cyclic_prefix_length :]
    with_prefix = np.concatenate([cyclic_prefix, symbols_in_time], axis=-1)
    return with_prefix.reshape(*grid.shape[:-2], -1).astype(np.complex64)


def ofdm_demodulate(
    signal: ComplexArray,
    *,
    n_ofdm_symbols: int,
    n_subcarriers: int,
    cyclic_prefix_length: int,
) -> ComplexArray:
    samples_per_symbol = n_subcarriers + cyclic_prefix_length
    frames = signal[..., : n_ofdm_symbols * samples_per_symbol]
    frames = frames.reshape(*signal.shape[:-1], n_ofdm_symbols, samples_per_symbol)
    symbol_bodies = frames[..., cyclic_prefix_length:]
    unitary_scale = np.sqrt(np.float32(n_subcarriers))
    return (np.fft.fftshift(np.fft.fft(symbol_bodies, axis=-1), axes=-1) / unitary_scale).astype(np.complex64)


def complex_gaussian_noise(
    shape: tuple[int, ...], *, noise_variance: float, rng: np.random.Generator
) -> ComplexArray:
    real_part = rng.standard_normal(shape, dtype=np.float32)
    imaginary_part = rng.standard_normal(shape, dtype=np.float32)
    return (np.sqrt(np.float32(noise_variance / 2.0)) * (real_part + 1j * imaginary_part)).astype(
        np.complex64
    )


def apply_frequency_channel(
    grid: ComplexArray,
    effective_cfr: ComplexArray,
    *,
    noise_variance: float,
    rng: np.random.Generator,
) -> ComplexArray:
    faded = effective_cfr[..., np.newaxis, :] * grid
    return faded + complex_gaussian_noise(faded.shape, noise_variance=noise_variance, rng=rng)


def _linear_convolution(signal: ComplexArray, cir: ComplexArray) -> ComplexArray:
    n_output = signal.shape[-1] + cir.shape[-1] - 1
    n_fft = 1 << int(np.ceil(np.log2(n_output)))
    spectrum = np.fft.fft(signal, n_fft, axis=-1) * np.fft.fft(cir, n_fft, axis=-1)
    return np.fft.ifft(spectrum, axis=-1)[..., :n_output]


def apply_time_channel(
    antenna_signals: ComplexArray,
    cir: ComplexArray,
    *,
    noise_variance: float,
    rng: np.random.Generator,
) -> ComplexArray:
    n_samples = antenna_signals.shape[-1]
    per_antenna = _linear_convolution(antenna_signals, cir)[..., :n_samples]
    received = np.sum(per_antenna, axis=-2).astype(np.complex64)
    return received + complex_gaussian_noise(received.shape, noise_variance=noise_variance, rng=rng)
