import numpy as np
import numpy.typing as npt

ComplexArray = npt.NDArray[np.complex64]
FloatArray = npt.NDArray[np.float32]


def linear_interpolation_matrix(
    pilot_subcarriers: npt.NDArray[np.int64], *, n_subcarriers: int
) -> FloatArray:
    pilot_position = pilot_subcarriers.astype(np.float32)
    subcarrier_position = np.arange(n_subcarriers, dtype=np.float32)
    left_pilot = np.clip(
        np.searchsorted(pilot_position, subcarrier_position, side="right") - 1, 0, pilot_position.size - 2
    )
    pilot_gap = pilot_position[left_pilot + 1] - pilot_position[left_pilot]
    fraction = (subcarrier_position - pilot_position[left_pilot]) / pilot_gap
    weights = np.zeros((n_subcarriers, pilot_position.size), dtype=np.float32)
    subcarrier_idx = np.arange(n_subcarriers)
    weights[subcarrier_idx, left_pilot] = 1.0 - fraction
    weights[subcarrier_idx, left_pilot + 1] = fraction
    return weights


def ls_estimate(
    received_grid: ComplexArray,
    known_pilots: ComplexArray,
    pilot_mask: npt.NDArray[np.bool_],
) -> ComplexArray:
    n_pilot_symbols = int(np.count_nonzero(pilot_mask.any(axis=-1)))
    per_pilot = received_grid[..., pilot_mask] / known_pilots
    per_symbol = per_pilot.reshape(*per_pilot.shape[:-1], n_pilot_symbols, -1)
    return np.mean(per_symbol, axis=-2).astype(np.complex64)


def interpolate_pilots(pilot_estimates: ComplexArray, interpolation_matrix: FloatArray) -> ComplexArray:
    return np.einsum("...p,kp->...k", pilot_estimates, interpolation_matrix).astype(np.complex64)
