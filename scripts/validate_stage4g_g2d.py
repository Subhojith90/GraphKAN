#!/usr/bin/env python3
"""Evaluate the G2d stabilization tail and the preserved composite decision."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ERRORS = {
    "energy": "exact_model_abs_energy_error_to_ed",
    "density": "exact_model_density_mae_to_ed",
    "correlation": "exact_model_connected_correlation_mae_to_ed",
}


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict): raise ValueError(f"{path} is not a mapping")
    return value


def read_phase(root: Path, filename: str) -> pd.DataFrame:
    frames = []
    for boundary in ("open", "periodic"):
        path = root / boundary / "raw" / filename
        if path.is_file():
            frame = pd.read_csv(path); frame["stage4g_boundary"] = boundary; frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def validate(amendment: dict[str, Any], g2c_root: Path, g2d: Path) -> dict[str, Any]:
    errors: list[str] = []
    metrics, sentinels = read_phase(g2d, "stage4g_metrics.csv"), read_phase(g2d, "stage4g_exact_sentinels.csv")
    if len(metrics) != 20 or metrics.run_id.nunique() != 20:
        errors.append(f"expected 20 unique G2d runs, found {len(metrics)}/{metrics.run_id.nunique() if len(metrics) else 0}")
    if len(metrics) and not (metrics.scientific_validity_exact == "VALID").all(): errors.append("a G2d run fails exact health")
    if len(metrics) and ("initial_checkpoint_sha256" not in metrics or metrics.initial_checkpoint_sha256.fillna("").str.len().ne(64).any()):
        errors.append("a G2d run lacks an attested 64-character initial-checkpoint hash")
    if len(sentinels) != 60 or set(sentinels.step.unique()) != {8000, 9000, 10000}:
        errors.append("G2d requires 60 sentinels at total steps 8000/9000/10000")

    limits = {key: float(value) for key, value in amendment["acceptance"]["maximum_absolute_change"].items()}
    keys = ["stage4g_boundary", "delta", "c6", "model_label"]
    rows = []
    for identity, group in sentinels.groupby(keys, sort=True):
        for outcome, column in ERRORS.items():
            by_step = group.groupby("step")[column].median()
            if not {9000, 10000}.issubset(by_step.index): continue
            change = abs(float(by_step.loc[10000]) - float(by_step.loc[9000]))
            rows.append({**dict(zip(keys, identity)), "outcome": outcome,
                         "median_error_9000": float(by_step.loc[9000]), "median_error_10000": float(by_step.loc[10000]),
                         "absolute_change": change, "maximum_absolute_change": limits[outcome],
                         "tolerance_utilisation": change / limits[outcome], "pass": change <= limits[outcome]})
    target = pd.DataFrame(rows); target.to_csv(g2d / "convergence_g2d_target_gate.csv", index=False)
    if len(target) != 12 or not bool(target["pass"].all()):
        errors.append(f"{int((~target['pass']).sum()) if len(target) else 12} of 12 G2d target comparisons fail")

    g2c_phase = g2c_root / "convergence_g2c"
    predecessor = pd.read_csv(g2c_phase / "convergence_g2c_composite_gate.csv")
    target_ids = {(str(t["boundary"]), float(t["delta"]), float(t["c6"]), str(t["model"])) for t in amendment["execution"]["targets"]}
    unaffected = predecessor[~predecessor.apply(
        lambda row: (str(row.stage4g_boundary), float(row.delta), float(row.c6), str(row.model_label)) in target_ids, axis=1
    )].copy()
    composite = pd.concat([unaffected, target], ignore_index=True, sort=False)
    composite.to_csv(g2d / "convergence_g2d_composite_gate.csv", index=False)
    if len(composite) != 72 or not bool(composite["pass"].all()): errors.append("the 72-row G2b/G2c/G2d composite does not pass")

    base_repairs = pd.read_csv(g2c_root / "predecessor_g2b/convergence_g2b_endpoint_repairs.csv")
    base_full = base_repairs[(base_repairs.stage4g_boundary == "open") & (base_repairs.model == "endpoint_full_b32") &
                             (base_repairs.step == 4200) & ~((base_repairs.delta == 2.0) & (base_repairs.c6 == 5.0))]
    final = sentinels[sentinels.step == 10000]; pair_keys = ["stage4g_boundary", "delta", "c6", "seed", "step"]
    radial = final[final.model_label == "rbf_radial_b32"].set_index(pair_keys)
    endpoint = final[final.model_label == "endpoint_full_b32"].set_index(pair_keys)
    replacements = []
    for key in radial.index.intersection(endpoint.index):
        row = dict(zip(pair_keys, key))
        for outcome, column in ERRORS.items(): row[f"repair_{outcome}"] = float(radial.loc[key, column] - endpoint.loc[key, column])
        replacements.append(row)
    repair = pd.concat([base_full, pd.DataFrame(replacements)], ignore_index=True, sort=False)
    cells = repair.groupby(["delta", "c6"], as_index=False).agg(
        median_density_repair=("repair_density", "median"), median_correlation_repair=("repair_correlation", "median"))
    cells["pass"] = (cells.median_density_repair > 0) & (cells.median_correlation_repair > 0)
    cells.to_csv(g2d / "convergence_g2d_endpoint_full_stability.csv", index=False)
    endpoint_passes = int(cells["pass"].sum())
    if len(cells) != 3 or endpoint_passes < 2: errors.append(f"endpoint-full repairs both observables in only {endpoint_passes}/3 open cells")

    alias = json.loads((g2c_root / "g2c_protocol/periodic_alias_gate.json").read_text())
    if alias.get("validation") != "PASS" or alias.get("passes") != 30: errors.append("periodic alias proof is not 30/30")
    g2c_decision = json.loads((g2c_phase / "convergence_g2c_gate.json").read_text())
    result = {
        "schema_version": "stage4g-g2d-stabilization-gate-v1", "validation": "PASS" if not errors else "FAIL",
        "decision_class": "SCIENTIFIC_GATE_PASS" if not errors else "SCIENTIFIC_GATE_FAIL", "blocking_errors": errors,
        "g2c_preserved_validation": g2c_decision["validation"], "target_runs": len(metrics),
        "target_exact_health_passes": int((metrics.scientific_validity_exact == "VALID").sum()) if len(metrics) else 0,
        "checkpoint_attestations": int(metrics.initial_checkpoint_sha256.fillna("").str.len().eq(64).sum()) if "initial_checkpoint_sha256" in metrics else 0,
        "target_group_passes": int(target["pass"].sum()) if len(target) else 0,
        "composite_group_passes": int(composite["pass"].sum()) if len(composite) else 0,
        "endpoint_full_open_cells_passed": endpoint_passes, "periodic_alias_passes": int(alias.get("passes", 0)),
        "endpoint_hybrid_status": "RETAINED_G2B_NEGATIVE_DIAGNOSTIC",
        "maximum_target_tolerance_utilisation": float(target.tolerance_utilisation.max()) if len(target) else None,
    }
    (g2d / "convergence_g2d_gate.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--amendment", type=Path, default=Path("configs/stage4g_g2d_stabilization.yaml"))
    parser.add_argument("--g2c-root", type=Path, required=True); parser.add_argument("--g2d", type=Path, required=True); args = parser.parse_args()
    result = validate(load_yaml(args.amendment), args.g2c_root, args.g2d); print(json.dumps(result, indent=2, sort_keys=True))
    if result["validation"] != "PASS": raise SystemExit(1)


if __name__ == "__main__": main()
