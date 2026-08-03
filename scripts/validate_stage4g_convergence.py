#!/usr/bin/env python3
"""Validate the complete Stage 4G convergence phase before control calibration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
ERRORS = {
    "energy": "exact_model_abs_energy_error_to_ed",
    "density": "exact_model_density_mae_to_ed",
    "correlation": "exact_model_connected_correlation_mae_to_ed",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "stage4g_frozen.yaml")
    parser.add_argument("--results", type=Path, default=ROOT / "results_stage4g")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    metric_frames, sentinel_frames = [], []
    for boundary in ("open", "periodic"):
        raw = args.results / "convergence" / boundary / "raw"
        metric = pd.read_csv(raw / "stage4g_metrics.csv"); metric["stage4g_boundary"] = boundary
        sentinel = pd.read_csv(raw / "stage4g_exact_sentinels.csv"); sentinel["stage4g_boundary"] = boundary
        metric_frames.append(metric); sentinel_frames.append(sentinel)
    metrics = pd.concat(metric_frames, ignore_index=True)
    sentinels = pd.concat(sentinel_frames, ignore_index=True)
    errors = []
    if len(metrics) != 120 or metrics.run_id.nunique() != 120:
        errors.append(f"expected 120 unique metrics rows, found {len(metrics)}/{metrics.run_id.nunique()}")
    if not (metrics.status == "COMPLETED").all():
        errors.append("a convergence run is not complete")
    if not (metrics.scientific_validity_exact == "VALID").all():
        errors.append("a convergence run fails exact health")
    if len(sentinels) != 360:
        errors.append(f"expected 360 exact sentinel rows, found {len(sentinels)}")
    threshold = float(cfg["acceptance"]["convergence"]["max_relative_change_1400_to_2100"])
    floors = cfg["acceptance"]["convergence"]["absolute_floors"]
    rows = []
    keys = ["stage4g_boundary", "delta", "c6", "model_label", "seed"]
    for identity, group in sentinels.groupby(keys):
        by_step = group.set_index("step")
        if not {700, 1400, 2100}.issubset(by_step.index):
            errors.append(f"missing sentinel step for {identity}")
            continue
        for outcome, column in ERRORS.items():
            earlier, final = float(by_step.loc[1400, column]), float(by_step.loc[2100, column])
            relative = abs(earlier - final) / max(abs(final), float(floors[outcome]))
            rows.append({**dict(zip(keys, identity)), "outcome": outcome, "relative_change": relative,
                         "pass": relative <= threshold})
    gate = pd.DataFrame(rows)
    if len(gate) != 360 or not bool(gate["pass"].all()):
        errors.append(f"{int((~gate['pass']).sum()) if len(gate) else 360} convergence comparisons exceed the frozen threshold")
    output = args.results / "convergence" / "convergence_gate.csv"
    gate.to_csv(output, index=False)
    result = {
        "schema_version": "stage4g-convergence-gate-v1", "validation": "PASS" if not errors else "FAIL",
        "blocking_errors": errors, "runs": len(metrics), "sentinels": len(sentinels),
        "comparisons": len(gate), "passes": int(gate["pass"].sum()) if len(gate) else 0,
    }
    (args.results / "convergence" / "convergence_gate.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
