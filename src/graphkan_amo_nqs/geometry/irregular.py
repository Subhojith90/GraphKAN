from __future__ import annotations

import numpy as np

from .base import Geometry, ordered_edges_from_distances


def make_irregular_geometry(n_sites: int, box_size: float = 4.0, min_distance: float = 0.35,
                            seed: int | None = None, edge_radius: float | None = None,
                            k_nearest: int | None = 3, max_tries: int = 10000) -> Geometry:
    rng = np.random.default_rng(seed)
    points: list[np.ndarray] = []
    tries = 0
    while len(points) < n_sites and tries < max_tries:
        tries += 1
        p = rng.uniform(0.0, box_size, size=2)
        if not points or min(np.linalg.norm(p - q) for q in points) >= min_distance:
            points.append(p)
    if len(points) != n_sites:
        raise RuntimeError(f"Could not place {n_sites} irregular points with min_distance={min_distance}")
    pos = np.array(points, dtype=float)
    edges = ordered_edges_from_distances(pos, radius=edge_radius, k=k_nearest)
    return Geometry("irregular", pos, edges, spacing=1.0, seed=seed, metadata={"box_size": box_size, "min_distance": min_distance})
