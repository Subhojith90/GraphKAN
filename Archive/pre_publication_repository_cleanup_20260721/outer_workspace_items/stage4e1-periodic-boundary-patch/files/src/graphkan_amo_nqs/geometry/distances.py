from __future__ import annotations

import numpy as np


def pairwise_distances(
    positions: np.ndarray,
    *,
    periodic_length: float | None = None,
) -> np.ndarray:
    """Pairwise distances, optionally using a 1D minimum-image convention.

    Periodic support is deliberately explicit.  A periodic geometry carries a
    physical ring length in metadata; raw Cartesian positions alone do not
    imply periodicity.
    """
    x = np.asarray(positions, dtype=float)
    diff = x[:, None, :] - x[None, :, :]
    if periodic_length is not None:
        length = float(periodic_length)
        if not np.isfinite(length) or length <= 0:
            raise ValueError("periodic_length must be finite and positive")
        # The current periodic-chain implementation is a one-dimensional ring
        # embedded along the first coordinate.  Other coordinates remain
        # ordinary Cartesian offsets.
        dx = np.abs(diff[..., 0])
        diff[..., 0] = np.minimum(dx, length - dx)
    return np.sqrt(np.sum(diff * diff, axis=-1))


def geometry_distance_matrix(geometry) -> np.ndarray:
    """Return the canonical distance matrix for a Geometry instance."""
    metadata = getattr(geometry, "metadata", {}) or {}
    bc = str(metadata.get("boundary_condition", "open")).lower()
    periodic_length = metadata.get("periodic_length") if bc == "periodic" else None
    return pairwise_distances(geometry.positions, periodic_length=periodic_length)


def upper_triangular_distances(
    positions: np.ndarray,
    *,
    periodic_length: float | None = None,
) -> np.ndarray:
    d = pairwise_distances(positions, periodic_length=periodic_length)
    iu = np.triu_indices_from(d, k=1)
    return d[iu]


def interaction_matrix(
    positions: np.ndarray,
    c6: float,
    epsilon: float = 1e-12,
    *,
    distance_matrix: np.ndarray | None = None,
    periodic_length: float | None = None,
) -> np.ndarray:
    d = (
        np.asarray(distance_matrix, dtype=float)
        if distance_matrix is not None
        else pairwise_distances(positions, periodic_length=periodic_length)
    )
    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be square")
    v = np.zeros_like(d, dtype=float)
    mask = d > epsilon
    v[mask] = float(c6) / np.power(d[mask], 6)
    np.fill_diagonal(v, 0.0)
    return v
