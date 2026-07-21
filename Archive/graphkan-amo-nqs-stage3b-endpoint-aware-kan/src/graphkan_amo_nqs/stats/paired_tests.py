from __future__ import annotations

import numpy as np
import scipy.stats


def wilcoxon_or_permutation(deltas: np.ndarray) -> dict:
    x = np.asarray(deltas, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2 or np.allclose(x, 0):
        return {"test": "insufficient", "p_value": float("nan")}
    try:
        stat, p = scipy.stats.wilcoxon(x)
        return {"test": "wilcoxon", "statistic": float(stat), "p_value": float(p)}
    except Exception:
        signs = np.sign(x)
        p = 2 * min(np.mean(signs >= 0), np.mean(signs <= 0))
        return {"test": "sign", "p_value": float(p)}
