from __future__ import annotations

import itertools

import numpy as np


def curve_alignment(curves: list[np.ndarray]) -> float:
    if len(curves) < 2:
        return float("nan")
    vals = []
    for a, b in itertools.combinations(curves, 2):
        a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            vals.append(0.0)
        else:
            vals.append(float(np.corrcoef(a, b)[0, 1]))
    return float(np.nanmedian(vals))


def stability_score(curves: list[np.ndarray]) -> float:
    if not curves:
        return float("nan")
    arr = np.vstack(curves)
    total_var = float(np.var(arr)) + 1e-12
    seed_var = float(np.mean(np.var(arr, axis=0)))
    return float(1.0 - seed_var / total_var)
