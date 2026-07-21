from __future__ import annotations

import pandas as pd


def parameter_count_table(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["model", "parameter_count", "profile", "notes"])
    return df.groupby(["profile", "model"], as_index=False).agg(parameter_count=("parameter_count", "median"))
