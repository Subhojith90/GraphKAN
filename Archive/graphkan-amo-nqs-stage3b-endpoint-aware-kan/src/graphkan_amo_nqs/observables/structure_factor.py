from __future__ import annotations

import numpy as np


def chain_structure_factor(connected: np.ndarray, positions: np.ndarray, k_values: np.ndarray) -> np.ndarray:
    x = np.asarray(positions)[:, 0]
    out = []
    for k in k_values:
        phase = np.cos(k * (x[:, None] - x[None, :]))
        out.append(np.sum(phase * connected) / len(x))
    return np.asarray(out, dtype=float)
