from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


def stable_hash(data: Any) -> str:
    blob = json.dumps(data, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def distance_matrix_hash(positions: np.ndarray, ndigits: int = 12) -> str:
    from .distances import pairwise_distances
    d = np.round(pairwise_distances(positions), ndigits).tolist()
    return stable_hash(d)


def geometry_id(geometry) -> str:
    payload = geometry.id_fields
    return f"{geometry.family}_N{geometry.n_sites}_{stable_hash(payload)}"
