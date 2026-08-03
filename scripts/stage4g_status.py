#!/usr/bin/env python3
"""Report exact Stage 4G completion counts without touching campaign state."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def key(row) -> tuple[str, float, float, str, int]:
    boundary = str(getattr(row, "stage4g_boundary", getattr(row, "boundary", "")))
    return boundary, float(row.delta), float(row.c6), str(getattr(row, "model_label", getattr(row, "model", ""))), int(row.seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("convergence", "calibration", "confirmation"), required=True)
    parser.add_argument("--results", type=Path, default=ROOT / "results_stage4g")
    args = parser.parse_args()
    protocol = args.results / "protocol"
    roster_path = protocol / ("confirmation_roster.csv" if args.phase == "confirmation" else "preselection_roster.csv")
    if not roster_path.is_file():
        raise SystemExit(f"roster not available: {roster_path}")
    roster = pd.read_csv(roster_path)
    roster = roster[roster.phase == args.phase]
    observed = []
    for boundary in ("open", "periodic"):
        path = args.results / args.phase / boundary / "raw" / "stage4g_metrics.csv"
        if path.is_file():
            frame = pd.read_csv(path)
            frame["stage4g_boundary"] = boundary
            observed.append(frame)
    metrics = pd.concat(observed, ignore_index=True) if observed else pd.DataFrame()
    complete_keys = set()
    failed_keys = set()
    if len(metrics):
        complete_keys = {key(row) for row in metrics[metrics.status == "COMPLETED"].itertuples(index=False)}
        failed_keys = {key(row) for row in metrics[metrics.status != "COMPLETED"].itertuples(index=False)}
    rows = []
    for row in roster.itertuples(index=False):
        item_key = key(row)
        rows.append({
            "boundary": row.boundary, "model": row.model,
            "state": "COMPLETE" if item_key in complete_keys else "FAILED_RETRYABLE" if item_key in failed_keys else "PENDING",
        })
    state = pd.DataFrame(rows)
    grouped = state.groupby(["boundary", "state"], as_index=False).size().to_dict(orient="records")
    result = {
        "schema_version": "stage4g-status-v1", "phase": args.phase, "expected": len(roster),
        "complete": int((state.state == "COMPLETE").sum()),
        "failed_retryable": int((state.state == "FAILED_RETRYABLE").sum()),
        "pending": int((state.state == "PENDING").sum()), "by_boundary": grouped,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
