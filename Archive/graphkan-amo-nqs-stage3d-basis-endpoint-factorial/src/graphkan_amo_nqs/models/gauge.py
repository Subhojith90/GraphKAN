from __future__ import annotations

import numpy as np


def gauge_fix_curve(r: np.ndarray, values: np.ndarray, weights: np.ndarray | None = None, align_sign: bool = True) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    v = np.asarray(values, dtype=float).copy()
    if weights is None:
        weights = np.ones_like(v) / len(v)
    else:
        weights = np.asarray(weights, dtype=float)
        weights = weights / (np.sum(weights) + 1e-12)
    v = v - float(np.sum(weights * v))
    amp = float(np.sqrt(np.sum(weights * v * v)))
    if amp > 1e-12:
        v = v / amp
    if align_sign and len(v) >= 2:
        # Convention: blockade repulsion channel should be more negative at shortest distance.
        if v[np.argmin(r)] > v[np.argmax(r)]:
            v = -v
    return v
