#!/usr/bin/env python3
"""Evaluate the prospectively frozen Stage 4G G2b convergence amendment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
ERRORS = {
    "energy": "exact_model_abs_energy_error_to_ed",
    "density": "exact_model_density_mae_to_ed",
    "correlation": "exact_model_connected_correlation_mae_to_ed",
}


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a mapping")
    return value


def read_phase(root: Path, filename: str) -> pd.DataFrame:
    frames = []
    for boundary in ("open", "periodic"):
        frame = pd.read_csv(root / "convergence_g2b" / boundary / "raw" / filename)
        frame["stage4g_boundary"] = boundary
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def enrich(sentinels: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in ("delta", "c6", "boundary_condition", "n_sites") if column not in sentinels]
    if missing:
        sentinels = sentinels.merge(
            metrics[["run_id", *missing]].drop_duplicates("run_id"),
            on="run_id", how="left", validate="many_to_one",
        )
    if sentinels[["delta", "c6"]].isna().any().any():
        raise ValueError("sentinel metadata could not be reconstructed")
    return sentinels


def validate(amendment: dict[str, Any], root: Path) -> dict[str, Any]:
    phase = root / "convergence_g2b"
    errors: list[str] = []
    metrics = read_phase(root, "stage4g_metrics.csv")
    sentinels = enrich(read_phase(root, "stage4g_exact_sentinels.csv"), metrics)
    if len(metrics) != 120 or metrics.run_id.nunique() != 120:
        errors.append(f"expected 120 unique metrics rows, found {len(metrics)}/{metrics.run_id.nunique()}")
    if not (metrics.status == "COMPLETED").all():
        errors.append("a G2b run is not complete")
    if not (metrics.scientific_validity_exact == "VALID").all():
        errors.append("a G2b run fails exact health")
    expected_steps = set(amendment["execution"]["exact_sentinel_steps"])
    if len(sentinels) != 360 or set(sentinels.step.unique()) != expected_steps:
        errors.append(f"expected 360 sentinel rows at {sorted(expected_steps)}, found {len(sentinels)}")

    acceptance = amendment["acceptance"]
    earlier_step, final_step = map(int, acceptance["compare_steps"])
    thresholds = {key: float(value) for key, value in acceptance["maximum_absolute_change"].items()}
    group_keys = ["stage4g_boundary", "delta", "c6", "model_label"]
    group_rows = []
    for identity, group in sentinels.groupby(group_keys, sort=True):
        for outcome, column in ERRORS.items():
            by_step = group.groupby("step")[column].median()
            if not {earlier_step, final_step}.issubset(by_step.index):
                continue
            earlier, final = float(by_step.loc[earlier_step]), float(by_step.loc[final_step])
            change = abs(final - earlier)
            group_rows.append({
                **dict(zip(group_keys, identity)), "outcome": outcome,
                f"median_error_{earlier_step}": earlier, f"median_error_{final_step}": final,
                "absolute_change": change, "maximum_absolute_change": thresholds[outcome],
                "tolerance_utilisation": change / thresholds[outcome], "pass": change <= thresholds[outcome],
            })
    group_gate = pd.DataFrame(group_rows)
    expected_groups = int(acceptance["required_group_comparisons"])
    if len(group_gate) != expected_groups or not bool(group_gate["pass"].all()):
        failed = int((~group_gate["pass"]).sum()) if len(group_gate) else expected_groups
        errors.append(f"{failed} of {expected_groups} effect-scale group convergence comparisons fail")
    group_gate.to_csv(phase / "convergence_g2b_group_gate.csv", index=False)

    repair_rows = []
    identity = ["stage4g_boundary", "delta", "c6", "seed", "step"]
    radial = sentinels[sentinels.model_label == "rbf_radial_b32"].set_index(identity)
    for model in acceptance["conclusion_stability"]["endpoint_models"]:
        endpoint = sentinels[sentinels.model_label == model].set_index(identity)
        for key in radial.index.intersection(endpoint.index):
            row = dict(zip(identity, key)) | {"model": model}
            for outcome, column in ERRORS.items():
                row[f"repair_{outcome}"] = float(radial.loc[key, column] - endpoint.loc[key, column])
            repair_rows.append(row)
    repairs = pd.DataFrame(repair_rows)
    repairs.to_csv(phase / "convergence_g2b_endpoint_repairs.csv", index=False)
    stability_rows = []
    stability = acceptance["conclusion_stability"]
    open_repairs = repairs[repairs.stage4g_boundary == "open"]
    for model in stability["endpoint_models"]:
        for step in stability["steps"]:
            cell = open_repairs[(open_repairs.model == model) & (open_repairs.step == step)].groupby(
                ["delta", "c6"], as_index=False
            ).agg(median_density_repair=("repair_density", "median"),
                  median_correlation_repair=("repair_correlation", "median"))
            passed = int(((cell.median_density_repair > 0) & (cell.median_correlation_repair > 0)).sum())
            required = int(stability["minimum_open_cells_with_positive_median_repair"])
            stability_rows.append({"model": model, "step": step, "open_cells_passed": passed,
                                   "open_cells_required": required, "pass": passed >= required})
    stability_gate = pd.DataFrame(stability_rows)
    stability_gate.to_csv(phase / "convergence_g2b_conclusion_stability.csv", index=False)
    if len(stability_gate) != 4 or not bool(stability_gate["pass"].all()):
        errors.append("the open-boundary endpoint conclusion is not stable at both late sentinels")

    periodic_rows = []
    periodic = repairs[repairs.stage4g_boundary == "periodic"]
    for row in periodic.itertuples(index=False):
        maximum = max(abs(float(getattr(row, f"repair_{outcome}"))) for outcome in ERRORS)
        periodic_rows.append({
            "delta": row.delta, "c6": row.c6, "seed": row.seed, "step": row.step,
            "model": row.model, "maximum_exact_error_difference": maximum, "pass": maximum == 0.0,
        })
    periodic_gate = pd.DataFrame(periodic_rows)
    periodic_gate.to_csv(phase / "convergence_g2b_periodic_collapse.csv", index=False)
    if len(periodic_gate) != 90 or not bool(periodic_gate["pass"].all()):
        errors.append("periodic endpoint/radial exact-error collapse is not exact in all 90 comparisons")

    summary = {
        "schema_version": "stage4g-g2b-convergence-gate-v1",
        "validation": "PASS" if not errors else "FAIL",
        "blocking_errors": errors,
        "runs": int(len(metrics)),
        "exact_health_passes": int((metrics.scientific_validity_exact == "VALID").sum()),
        "sentinels": int(len(sentinels)),
        "group_comparisons": int(len(group_gate)),
        "group_passes": int(group_gate["pass"].sum()) if len(group_gate) else 0,
        "conclusion_stability_passes": int(stability_gate["pass"].sum()) if len(stability_gate) else 0,
        "periodic_collapse_passes": int(periodic_gate["pass"].sum()) if len(periodic_gate) else 0,
        "maximum_tolerance_utilisation": float(group_gate.tolerance_utilisation.max()) if len(group_gate) else None,
    }
    (phase / "convergence_g2b_gate.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--amendment", type=Path, default=ROOT / "configs" / "stage4g_g2b_amendment.yaml")
    parser.add_argument("--results", type=Path, default=ROOT / "results_stage4g")
    args = parser.parse_args()
    result = validate(load_yaml(args.amendment), args.results)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["validation"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
