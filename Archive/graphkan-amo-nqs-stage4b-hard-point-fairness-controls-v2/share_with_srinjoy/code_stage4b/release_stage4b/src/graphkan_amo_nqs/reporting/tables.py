from __future__ import annotations

from pathlib import Path

import pandas as pd


def dataframe_to_latex(df: pd.DataFrame, path: str | Path, caption: str, label: str, max_rows: int = 40) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        df = pd.DataFrame({"status": ["not available"]})
    tex = df.head(max_rows).to_latex(index=False, escape=True, longtable=False, caption=caption, label=label)
    p.write_text(tex, encoding="utf-8")


def safe_claim_table(path: str | Path) -> pd.DataFrame:
    rows = [
        {"claim_type": "safe", "claim": "The CPU artifact validates dense/sparse Rydberg Hamiltonian construction on small systems.", "evidence": str(path)},
        {"claim_type": "safe", "claim": "Pilot exact-fullsum KAN-Jastrow runs were executed where result CSVs are present.", "evidence": "results/raw/run_metrics.csv"},
        {"claim_type": "unsafe", "claim": "KAN-Jastrow is generally superior to MLP/RBM/RNN/transformers.", "evidence": "forbidden; no final evidence"},
        {"claim_type": "unsafe", "claim": "IBM hardware validates Rydberg NQS performance.", "evidence": "forbidden; platform mismatch"},
    ]
    return pd.DataFrame(rows)
