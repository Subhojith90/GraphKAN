from __future__ import annotations

import numpy as np

from .base import Geometry, ordered_edges_from_distances, with_jitter


def make_chain_geometry(n_sites: int, spacing: float = 1.0, jitter: float = 0.0, seed: int | None = None,
                        edge_radius: float | None = None, k_nearest: int | None = None) -> Geometry:
    positions = np.stack([np.arange(n_sites, dtype=float) * spacing, np.zeros(n_sites)], axis=1)
    positions = with_jitter(positions, jitter, seed)
    edges = ordered_edges_from_distances(positions, radius=edge_radius or 1.01 * spacing, k=k_nearest)
    return Geometry("chain", positions, edges, spacing=spacing, seed=seed, metadata={"jitter": jitter})
