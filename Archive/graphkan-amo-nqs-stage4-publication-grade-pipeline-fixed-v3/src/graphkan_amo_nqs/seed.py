from __future__ import annotations

import hashlib
import os
import random
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SeedLineage:
    root_seed: int
    label: str
    derived_seed: int


def stable_int_hash(text: str, modulo: int = 2**32 - 1) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % modulo


def derive_seed(root_seed: int, *labels: object) -> int:
    return stable_int_hash("::".join([str(root_seed), *map(str, labels)]))


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))
