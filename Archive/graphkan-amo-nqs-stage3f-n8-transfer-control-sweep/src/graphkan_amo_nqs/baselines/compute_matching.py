from __future__ import annotations

import pandas as pd


def compute_budget_table(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame(columns=["profile", "model", "median_runtime_sec"])
    runtime = metrics[metrics["metric"] == "runtime_sec"].copy()
    if runtime.empty:
        return pd.DataFrame(columns=["profile", "model", "median_runtime_sec"])
    return runtime.groupby(["profile", "model"], as_index=False).agg(median_runtime_sec=("value", "median"), n_runs=("value", "size"))
