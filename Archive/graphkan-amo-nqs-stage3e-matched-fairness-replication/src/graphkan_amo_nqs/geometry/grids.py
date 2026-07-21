from __future__ import annotations

import numpy as np

from .base import Geometry, ordered_edges_from_distances, with_jitter


def make_square_grid_geometry(rows: int, cols: int, spacing: float = 1.0, jitter: float = 0.0,
                              seed: int | None = None, edge_radius: float | None = None,
                              k_nearest: int | None = None) -> Geometry:
    positions = []
    for y in range(rows):
        for x in range(cols):
            positions.append((x * spacing, y * spacing))
    pos = with_jitter(np.array(positions, dtype=float), jitter, seed)
    edges = ordered_edges_from_distances(pos, radius=edge_radius or 1.01 * spacing, k=k_nearest)
    return Geometry("square", pos, edges, spacing=spacing, seed=seed, metadata={"rows": rows, "cols": cols, "jitter": jitter})
