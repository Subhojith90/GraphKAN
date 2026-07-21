from __future__ import annotations

import pandas as pd

from .bootstrap import bootstrap_ci


def uncertainty_summary(df: pd.DataFrame, group_cols: list[str], metric: str) -> pd.DataFrame:
    rows = []
    sub = df[df["metric"] == metric]
    for keys, g in sub.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        med, lo, hi = bootstrap_ci(g["value"].to_numpy(), n_resamples=500, seed=0)
        row = dict(zip(group_cols, keys)); row.update({"metric": metric, "median": med, "ci_low": lo, "ci_high": hi, "n": len(g)})
        rows.append(row)
    return pd.DataFrame(rows)
