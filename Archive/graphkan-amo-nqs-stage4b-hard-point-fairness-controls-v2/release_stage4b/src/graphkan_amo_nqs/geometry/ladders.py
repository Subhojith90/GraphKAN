from __future__ import annotations

import numpy as np

from .base import Geometry, ordered_edges_from_distances, with_jitter


def make_ladder_geometry(length: int, width: int = 2, spacing: float = 1.0, jitter: float = 0.0,
                         seed: int | None = None, edge_radius: float | None = None,
                         k_nearest: int | None = None) -> Geometry:
    positions = []
    for x in range(length):
        for y in range(width):
            positions.append((x * spacing, y * spacing))
    pos = with_jitter(np.array(positions, dtype=float), jitter, seed)
    edges = ordered_edges_from_distances(pos, radius=edge_radius or 1.01 * spacing, k=k_nearest)
    return Geometry("ladder", pos, edges, spacing=spacing, seed=seed, metadata={"length": length, "width": width, "jitter": jitter})
