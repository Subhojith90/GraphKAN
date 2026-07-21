from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .hashing import distance_matrix_hash


def assign_split(index: int, fractions=(0.6, 0.2, 0.2)) -> str:
    frac = (index * 0.61803398875) % 1.0
    if frac < fractions[0]:
        return "train"
    if frac < fractions[0] + fractions[1]:
        return "validation"
    return "test"


def detect_split_leakage(records: Iterable[dict]) -> list[str]:
    seen = defaultdict(set)
    for rec in records:
        key = rec.get("distance_matrix_hash") or rec.get("geometry_hash") or rec.get("instance_id")
        split = rec.get("split")
        seen[key].add(split)
    return [k for k, splits in seen.items() if len(splits) > 1]
