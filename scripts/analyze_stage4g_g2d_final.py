#!/usr/bin/env python3
"""Evaluate the N=12 Stage 4G campaign after accepted G2d convergence closure."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ERRORS = {
    "energy": "exact_model_abs_energy_error_to_ed",
    "density": "exact_model_density_mae_to_ed",
    "correlation": "exact_model_connected_correlation_mae_to_ed",
}


def read_confirmation(root: Path, filename: str) -> pd.DataFrame:
    frames = []
    for boundary in ("open", "periodic"):
        frame = pd.read_csv(root / "confirmation" / boundary / "raw" / filename)
        frame["stage4g_boundary"] = boundary; frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def analyse(root: Path, output: Path, accepted_g2d: Path, periodic_alias: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    blocking: list[str] = []
    g2d = json.loads((accepted_g2d / "convergence_g2d/convergence_g2d_gate.json").read_text())
    if g2d.get("validation") != "PASS" or g2d.get("composite_group_passes") != 72:
        blocking.append("the supplied G2d convergence closure is not an accepted 72/72 decision")
    metrics = read_confirmation(root, "stage4g_metrics.csv")
    if len(metrics) != 180 or metrics.run_id.nunique() != 180:
        blocking.append(f"confirmation contains {len(metrics)} rows/{metrics.run_id.nunique()} unique, expected 180")
    exact_health = int((metrics.scientific_validity_exact == "VALID").sum()) if len(metrics) else 0
    if exact_health != 180: blocking.append(f"only {exact_health}/180 confirmation runs pass exact health")
    if set(metrics.n_sites.astype(int)) != {12}: blocking.append("confirmation contains a system size other than N=12")

    keys = ["stage4g_boundary", "delta", "c6", "seed"]
    radial = metrics[metrics.model_label == "rbf_radial_b32"].set_index(keys)
    paired_rows = []
    for model in ("endpoint_full_b32", "endpoint_hybrid_b32"):
        endpoint = metrics[metrics.model_label == model].set_index(keys)
        for key in radial.index.intersection(endpoint.index):
            row = dict(zip(keys, key)) | {"model": model}
            for outcome, column in ERRORS.items(): row[f"repair_{outcome}"] = float(radial.loc[key, column] - endpoint.loc[key, column])
            paired_rows.append(row)
    paired = pd.DataFrame(paired_rows); paired.to_csv(output / "n12_paired_endpoint_effects.csv", index=False)
    open_cells = paired[paired.stage4g_boundary == "open"].groupby(["model", "delta", "c6"], as_index=False).agg(
        median_energy_repair=("repair_energy", "median"), median_density_repair=("repair_density", "median"),
        median_correlation_repair=("repair_correlation", "median"))
    open_cells["density_and_correlation_repaired"] = (
        (open_cells.median_density_repair > 0) & (open_cells.median_correlation_repair > 0))
    open_cells.to_csv(output / "n12_open_cell_decisions.csv", index=False)
    counts = {str(model): int(group.density_and_correlation_repaired.sum()) for model, group in open_cells.groupby("model")}
    if counts.get("endpoint_full_b32", 0) < 2:
        blocking.append(f"endpoint-full repairs both observables in only {counts.get('endpoint_full_b32', 0)}/3 N=12 open cells")

    alias = json.loads(periodic_alias.read_text()) if periodic_alias.is_file() else {"validation": "FAIL", "passes": 0}
    if alias.get("validation") != "PASS" or alias.get("passes") != 30 or alias.get("n_sites") != 12:
        blocking.append("the N=12 same-parameter periodic alias proof is not 30/30")

    model_summary = metrics.groupby(["stage4g_boundary", "model_label"], as_index=False).agg(
        runs=("run_id", "count"), median_energy_error=(ERRORS["energy"], "median"),
        median_density_error=(ERRORS["density"], "median"), median_correlation_error=(ERRORS["correlation"], "median"),
        median_wall_seconds=("wall_seconds", "median"), max_parameters=("n_parameters", "max"))
    model_summary.to_csv(output / "n12_model_boundary_summary.csv", index=False)
    result = {
        "schema_version": "stage4g-g2d-final-decision-v1", "validation": "PASS" if not blocking else "FAIL",
        "decision_class": "STAGE4G_COMPLETE" if not blocking else "STAGE4G_FINAL_GATE_FAIL",
        "blocking_errors": blocking, "accepted_g2d_composite_passes": int(g2d.get("composite_group_passes", 0)),
        "confirmation_runs": len(metrics), "confirmation_exact_health_passes": exact_health,
        "endpoint_full_open_cells_passed": counts.get("endpoint_full_b32", 0),
        "endpoint_hybrid_open_cells_passed": counts.get("endpoint_hybrid_b32", 0),
        "endpoint_hybrid_role": "DIAGNOSTIC_NEGATIVE_OR_POSITIVE_RESULT_NONBLOCKING",
        "periodic_alias_passes": int(alias.get("passes", 0)),
        "scientific_decision": (
            "Stage 4G closes the finite-size endpoint-full mechanism confirmation with endpoint-hybrid retained diagnostically"
            if not blocking else "Stage 4G does not close the amended final confirmation gate"),
    }
    (output / "stage4g_final_decision.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--results", type=Path, default=Path("results_stage4g"))
    parser.add_argument("--output", type=Path, default=Path("results_stage4g/analysis")); parser.add_argument("--accepted-g2d", type=Path, required=True)
    parser.add_argument("--periodic-alias", type=Path, required=True); args = parser.parse_args()
    result = analyse(args.results, args.output, args.accepted_g2d, args.periodic_alias); print(json.dumps(result, indent=2, sort_keys=True))
    if result["validation"] != "PASS": raise SystemExit(1)


if __name__ == "__main__": main()
