from __future__ import annotations

import numpy as np
import scipy.stats

from graphkan_amo_nqs.geometry.distances import pairwise_distances


def correlation_matching_score(positions: np.ndarray, connected: np.ndarray, r_grid: np.ndarray, f_values: np.ndarray) -> float:
    d = pairwise_distances(positions)
    iu = np.triu_indices_from(d, k=1)
    r_pairs = d[iu]
    c_pairs = np.asarray(connected)[iu]
    # interpolate f to pair distances and compare monotonic pattern with connected correlations
    f_interp = np.interp(r_pairs, r_grid, f_values)
    if len(r_pairs) < 3:
        return float("nan")
    return float(scipy.stats.spearmanr(f_interp, c_pairs, nan_policy="omit").statistic)
