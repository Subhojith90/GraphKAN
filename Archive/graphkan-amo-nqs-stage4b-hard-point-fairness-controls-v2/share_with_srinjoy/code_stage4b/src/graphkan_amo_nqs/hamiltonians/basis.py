from __future__ import annotations

import numpy as np


def enumerate_binary_states(n_sites: int) -> np.ndarray:
    nums = np.arange(2**n_sites, dtype=np.uint64)
    bits = ((nums[:, None] >> np.arange(n_sites, dtype=np.uint64)) & 1).astype(np.int8)
    return bits


def bits_to_int(bits: np.ndarray) -> int:
    b = np.asarray(bits, dtype=np.uint64)
    return int(np.sum(b << np.arange(len(b), dtype=np.uint64)))


def int_to_bits(index: int, n_sites: int) -> np.ndarray:
    return (((np.uint64(index) >> np.arange(n_sites, dtype=np.uint64)) & 1)).astype(np.int8)


def parity_signs(states: np.ndarray) -> np.ndarray:
    return np.where(np.sum(states, axis=1) % 2 == 0, 1.0, -1.0)


def sign_vector(states: np.ndarray, sign_rule: str = "none") -> np.ndarray:
    if sign_rule in (None, "none", "positive"):
        return np.ones(states.shape[0], dtype=float)
    if sign_rule == "parity":
        return parity_signs(states)
    raise ValueError(f"Unknown sign_rule: {sign_rule}")
