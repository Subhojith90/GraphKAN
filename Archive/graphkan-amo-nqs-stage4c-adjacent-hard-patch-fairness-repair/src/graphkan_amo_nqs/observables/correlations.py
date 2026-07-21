from __future__ import annotations

import numpy as np

from .density import probabilities_from_state, density_from_probabilities


def two_point_from_probabilities(probabilities: np.ndarray, states: np.ndarray) -> np.ndarray:
    s = np.asarray(states, dtype=float)
    return np.einsum("b,bi,bj->ij", probabilities, s, s, optimize=True)


def connected_from_probabilities(probabilities: np.ndarray, states: np.ndarray) -> np.ndarray:
    rho = density_from_probabilities(probabilities, states)
    nn = two_point_from_probabilities(probabilities, states)
    return nn - np.outer(rho, rho)


def connected_correlations_from_state(vector: np.ndarray, states: np.ndarray) -> np.ndarray:
    return connected_from_probabilities(probabilities_from_state(vector), states)


def correlation_mae(a: np.ndarray, b: np.ndarray, mask_diagonal: bool = True) -> float:
    x = np.asarray(a, dtype=float); y = np.asarray(b, dtype=float)
    mask = np.ones_like(x, dtype=bool)
    if mask_diagonal:
        np.fill_diagonal(mask, False)
    return float(np.mean(np.abs(x[mask] - y[mask])))
