from __future__ import annotations

import numpy as np

from .base import Geometry, ordered_edges_from_distances, with_jitter


def make_kagome_fragment(cells_x: int, cells_y: int, spacing: float = 1.0, jitter: float = 0.0,
                         seed: int | None = None, edge_radius: float | None = None,
                         k_nearest: int | None = None) -> Geometry:
    # Three-site basis in a triangular Bravais cell. This is a finite fragment, not a periodic lattice.
    a1 = np.array([spacing, 0.0])
    a2 = np.array([0.5 * spacing, np.sqrt(3) * spacing / 2])
    basis = [np.array([0.0, 0.0]), 0.5 * a1, 0.5 * a2]
    positions = []
    for iy in range(cells_y):
        for ix in range(cells_x):
            origin = ix * a1 + iy * a2
            for b in basis:
                positions.append(origin + b)
    pos = with_jitter(np.array(positions, dtype=float), jitter, seed)
    edges = ordered_edges_from_distances(pos, radius=edge_radius or 0.56 * spacing, k=k_nearest)
    return Geometry("kagome", pos, edges, spacing=spacing, seed=seed, metadata={"cells_x": cells_x, "cells_y": cells_y, "jitter": jitter})
