from __future__ import annotations

import numpy as np


def pairwise_distances(positions: np.ndarray) -> np.ndarray:
    x = np.asarray(positions, dtype=float)
    diff = x[:, None, :] - x[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=-1))
    return d


def upper_triangular_distances(positions: np.ndarray) -> np.ndarray:
    d = pairwise_distances(positions)
    iu = np.triu_indices_from(d, k=1)
    return d[iu]


def interaction_matrix(positions: np.ndarray, c6: float, epsilon: float = 1e-12) -> np.ndarray:
    d = pairwise_distances(positions)
    v = np.zeros_like(d, dtype=float)
    mask = d > epsilon
    v[mask] = float(c6) / np.power(d[mask], 6)
    np.fill_diagonal(v, 0.0)
    return v
