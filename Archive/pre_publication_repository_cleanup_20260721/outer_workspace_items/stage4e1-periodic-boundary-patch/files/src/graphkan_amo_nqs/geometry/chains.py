from __future__ import annotations

import numpy as np

from .base import Geometry, ordered_edges_from_distances, with_jitter
from .distances import pairwise_distances


def make_chain_geometry(n_sites: int, spacing: float = 1.0, jitter: float = 0.0, seed: int | None = None,
                        edge_radius: float | None = None, k_nearest: int | None = None) -> Geometry:
    positions = np.stack([np.arange(n_sites, dtype=float) * spacing, np.zeros(n_sites)], axis=1)
    positions = with_jitter(positions, jitter, seed)
    edges = ordered_edges_from_distances(positions, radius=edge_radius or 1.01 * spacing, k=k_nearest)
    return Geometry("chain", positions, edges, spacing=spacing, seed=seed,
                    metadata={"jitter": jitter, "boundary_condition": "open"})


def make_periodic_chain_geometry(
    n_sites: int,
    spacing: float = 1.0,
    *,
    seed: int | None = None,
) -> Geometry:
    """Construct a 1D periodic chain with minimum-image distances.

    The positions remain on a line for compatibility with existing model code;
    periodicity is carried by explicit metadata and must be consumed through
    ``geometry_distance_matrix`` rather than inferred from positions.
    """
    if int(n_sites) < 3:
        raise ValueError("periodic chain requires at least three sites")
    n_sites = int(n_sites)
    spacing = float(spacing)
    length = n_sites * spacing
    positions = np.stack([np.arange(n_sites, dtype=float) * spacing, np.zeros(n_sites)], axis=1)
    d = pairwise_distances(positions, periodic_length=length)
    edges = []
    for i in range(n_sites):
        for j in range(i + 1, n_sites):
            if np.isclose(d[i, j], spacing):
                edges.append((i, j))
    return Geometry(
        "periodic_chain",
        positions,
        tuple(edges),
        spacing=spacing,
        seed=seed,
        metadata={
            "boundary_condition": "periodic",
            "periodic_length": length,
            "endpoint_features_enabled": False,
        },
    )
