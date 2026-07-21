from __future__ import annotations

from pathlib import Path

import pandas as pd

from graphkan_amo_nqs.manifests import atomic_write_dataframe


def append_csv(path: str | Path, rows: list[dict]) -> None:
    p = Path(path)
    if p.exists():
        old = pd.read_csv(p)
        df = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
    else:
        df = pd.DataFrame(rows)
    atomic_write_dataframe(p, df)
