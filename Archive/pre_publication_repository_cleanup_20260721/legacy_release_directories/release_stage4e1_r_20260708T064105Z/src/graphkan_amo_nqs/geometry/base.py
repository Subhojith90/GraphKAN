from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Geometry:
    family: str
    positions: np.ndarray
    edges: tuple[tuple[int, int], ...]
    spacing: float = 1.0
    seed: int | None = None
    generator_version: str = "0.1.0"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_sites(self) -> int:
        return int(self.positions.shape[0])

    @property
    def id_fields(self) -> dict[str, Any]:
        from .hashing import distance_matrix_hash
        from .distances import geometry_distance_matrix
        return {
            "family": self.family,
            "n_sites": self.n_sites,
            "spacing": self.spacing,
            "seed": self.seed,
            "distance_matrix_hash": distance_matrix_hash(distance_matrix=geometry_distance_matrix(self)),
            **self.metadata,
        }

    def instance_prefix(self) -> str:
        from .hashing import geometry_id
        return geometry_id(self)


def ordered_edges_from_distances(positions: np.ndarray, radius: float | None = None, k: int | None = None) -> tuple[tuple[int, int], ...]:
    from .distances import pairwise_distances
    d = pairwise_distances(positions)
    n = len(positions)
    edge_set: set[tuple[int, int]] = set()
    if radius is not None:
        for i in range(n):
            for j in range(i + 1, n):
                if d[i, j] <= radius and d[i, j] > 0:
                    edge_set.add((i, j))
    if k is not None and k > 0:
        for i in range(n):
            order = np.argsort(d[i])
            added = 0
            for j in order:
                if i == j:
                    continue
                a, b = sorted((int(i), int(j)))
                edge_set.add((a, b))
                added += 1
                if added >= k:
                    break
    if radius is None and k is None:
        for i in range(n):
            for j in range(i + 1, n):
                edge_set.add((i, j))
    return tuple(sorted(edge_set))


def with_jitter(positions: np.ndarray, jitter: float, seed: int | None) -> np.ndarray:
    if jitter <= 0:
        return np.asarray(positions, dtype=float)
    rng = np.random.default_rng(seed)
    return np.asarray(positions, dtype=float) + rng.normal(scale=jitter, size=np.asarray(positions).shape)
