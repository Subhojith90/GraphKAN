from __future__ import annotations

import numpy as np


def bootstrap_ci(values, n_resamples: int = 2000, ci: float = 0.95, seed: int = 0, statistic=np.median) -> tuple[float, float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(int(n_resamples)):
        sample = rng.choice(x, size=len(x), replace=True)
        stats.append(float(statistic(sample)))
    alpha = (1.0 - ci) / 2.0
    return float(statistic(x)), float(np.quantile(stats, alpha)), float(np.quantile(stats, 1 - alpha))
