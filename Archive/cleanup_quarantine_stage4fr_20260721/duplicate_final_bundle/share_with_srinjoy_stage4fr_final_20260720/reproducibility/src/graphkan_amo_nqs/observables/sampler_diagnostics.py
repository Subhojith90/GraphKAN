from __future__ import annotations

import numpy as np


def batch_standard_error(values: np.ndarray, n_batches: int = 8) -> float:
    x = np.asarray(values, dtype=float)
    if len(x) < 2:
        return float("nan")
    n_batches = max(1, min(n_batches, len(x)))
    batches = np.array_split(x, n_batches)
    means = np.array([np.mean(b) for b in batches])
    return float(np.std(means, ddof=1) / np.sqrt(len(means))) if len(means) > 1 else 0.0
