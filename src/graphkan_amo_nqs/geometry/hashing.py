from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


def stable_hash(data: Any) -> str:
    blob = json.dumps(data, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def distance_matrix_hash(
    positions: np.ndarray | None = None,
    ndigits: int = 12,
    *,
    distance_matrix: np.ndarray | None = None,
    periodic_length: float | None = None,
) -> str:
    from .distances import pairwise_distances
    if distance_matrix is None:
        if positions is None:
            raise ValueError("positions or distance_matrix is required")
        distance_matrix = pairwise_distances(positions, periodic_length=periodic_length)
    d = np.round(np.asarray(distance_matrix, dtype=float), ndigits).tolist()
    return stable_hash(d)


def geometry_id(geometry) -> str:
    payload = geometry.id_fields
    return f"{geometry.family}_N{geometry.n_sites}_{stable_hash(payload)}"
