from __future__ import annotations

import numpy as np
import scipy.stats


def characteristic_radius(r: np.ndarray, values: np.ndarray) -> float:
    idx = int(np.argmax(np.abs(values)))
    return float(np.asarray(r)[idx])


def scale_correlation(r_stars: list[float], scales: list[float]) -> float:
    if len(r_stars) < 2 or len(scales) < 2:
        return float("nan")
    return float(scipy.stats.spearmanr(r_stars, scales, nan_policy="omit").statistic)
