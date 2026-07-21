from __future__ import annotations

import numpy as np


def fidelity(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=complex); y = np.asarray(b, dtype=complex)
    nx = np.vdot(x, x).real; ny = np.vdot(y, y).real
    if nx <= 0 or ny <= 0:
        return 0.0
    return float(abs(np.vdot(x, y)) ** 2 / (nx * ny))
