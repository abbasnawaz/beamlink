import numpy as np
import numpy.typing as npt

from beamlink.equalization import mmse_equalize, zf_equalize

N_OFDM_SYMBOLS = 14
N_SUBCARRIERS = 256
TINY_NOISE_VARIANCE = 1e-9
TOLERANCE = 1e-3


def _channel_and_grid(seed: int) -> tuple[npt.NDArray[np.complex64], npt.NDArray[np.complex64]]:
    rng = np.random.default_rng(seed)
    cfr = (rng.standard_normal(N_SUBCARRIERS) + 1j * rng.standard_normal(N_SUBCARRIERS)).astype(np.complex64)
    shape = (N_OFDM_SYMBOLS, N_SUBCARRIERS)
    grid = np.exp(1j * rng.uniform(0, 2 * np.pi, shape)).astype(np.complex64)
    return cfr, grid


def test_zf_inverts_a_noiseless_channel() -> None:
    cfr, grid = _channel_and_grid(0)
    equalized, post_noise = zf_equalize(cfr[np.newaxis, :] * grid, cfr)
    assert np.allclose(equalized, grid, atol=TOLERANCE)
    assert np.allclose(post_noise[0], 1.0 / np.abs(cfr) ** 2, rtol=1e-4)


def test_unbiased_mmse_inverts_a_noiseless_channel() -> None:
    cfr, grid = _channel_and_grid(1)
    equalized, _ = mmse_equalize(cfr[np.newaxis, :] * grid, cfr, noise_variance=TINY_NOISE_VARIANCE)
    assert np.allclose(equalized, grid, atol=TOLERANCE)
