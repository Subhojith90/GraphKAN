#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_stage4d/fullbudget_smoke")
    args = ap.parse_args()
    root = Path(args.results)
    p = root / "raw" / "stage4d_metrics.csv"
    if not p.exists():
        raise SystemExit(f"Missing smoke metrics: {p}")
    df = pd.read_csv(p)
    expected = {"rbf_radial_21p", "rbf_endpoint_sum_21p", "rbf_endpoint_product_21p", "rbf_endpoint_hybrid_22p", "rbf_endpoint_full_23p", "mlp_endpoint_128p"}
    present = set(df.model_label.dropna().astype(str))
    missing = sorted(expected - present)
    failed = df[df.status.astype(str).str.upper() != "COMPLETED"]
    key = {"rbf_endpoint_sum_21p", "rbf_endpoint_hybrid_22p", "mlp_endpoint_128p"}
    key_rows = df[df.model_label.isin(key)]
    key_valid = len(key_rows) == 3 and (key_rows.scientific_validity_exact == "VALID").all() and (key_rows.scientific_validity_sample == "VALID").all()
    payload = {"schema":"stage4d-smoke-gate-v1", "rows":int(len(df)), "completed_rows":int((df.status == "COMPLETED").sum()), "failed_rows":int(len(failed)), "missing_models":missing, "key_models_dual_valid":bool(key_valid), "status":"PASS" if not missing and len(df) == 6 and len(failed) == 0 and key_valid else "FAIL"}
    (root / "stage4d_smoke_gate.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "PASS":
        raise SystemExit("Stage 4D full-budget smoke gate failed.")

if __name__ == "__main__":
    main()
