#!/usr/bin/env python3
"""Run Stage 4B summaries over all configured batch folders and build a bundle index."""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path
import pandas as pd

BATCHES = [
    ("n8_component_replay", "rbf_endpoint_sum_21p"),
    ("n8_controls_core", "rbf_endpoint_sum_21p"),
    ("n8_capacity_compute", "rbf_endpoint_sum_b12_21p"),
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="results_stage4b")
    a = p.parse_args()
    root = Path(a.root)
    index = []
    for batch, candidate in BATCHES:
        result_dir = root / batch
        raw = result_dir / "raw" / "stage4b_metrics.csv"
        if not raw.exists():
            index.append({"batch": batch, "status": "MISSING", "results": str(result_dir)})
            continue
        subprocess.run([sys.executable, "scripts/summarize_stage4b.py", "--results", str(result_dir), "--candidate", candidate], check=True)
        metrics = pd.read_csv(raw)
        index.append({"batch": batch, "status": "SUMMARISED", "results": str(result_dir), "completed_rows": int((metrics.status == 'COMPLETED').sum())})
    bundle = root / "evidence_bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(index).to_csv(bundle / "batch_index.csv", index=False)
    print(pd.DataFrame(index).to_string(index=False))
    print("Wrote:", bundle)

if __name__ == "__main__":
    main()
