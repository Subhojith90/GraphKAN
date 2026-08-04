#!/usr/bin/env python3
"""Evaluate G2c without mutating or relabelling the failed G2b evidence."""
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
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a mapping")
    return value


def read_target(root: Path, filename: str) -> pd.DataFrame:
    frames = []
    for boundary in ("open", "periodic"):
        path = root / boundary / "raw" / filename
        if path.is_file():
            frame = pd.read_csv(path); frame["stage4g_boundary"] = boundary; frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def validate(amendment: dict[str, Any], g2b: Path, g2c: Path, periodic_alias: Path) -> dict[str, Any]:
    errors: list[str] = []
    metrics = read_target(g2c, "stage4g_metrics.csv")
    sentinels = read_target(g2c, "stage4g_exact_sentinels.csv")
    if len(metrics) != 25 or metrics.run_id.nunique() != 25:
        errors.append(f"expected 25 unique G2c runs, found {len(metrics)}/{metrics.run_id.nunique() if len(metrics) else 0}")
    if len(metrics) and not (metrics.scientific_validity_exact == "VALID").all():
        errors.append("a G2c target run fails exact health")
    if len(sentinels) != 75 or set(sentinels.step.unique()) != {5600, 6300, 7000}:
        errors.append("G2c does not contain 75 exact sentinel rows at 5600/6300/7000")

    acceptance = amendment["acceptance"]
    limits = {key: float(value) for key, value in acceptance["maximum_absolute_change"].items()}
    rows = []
    keys = ["stage4g_boundary", "delta", "c6", "model_label"]
    for identity, group in sentinels.groupby(keys, sort=True):
        for outcome, column in ERRORS.items():
            by_step = group.groupby("step")[column].median()
            if not {6300, 7000}.issubset(by_step.index):
                continue
            change = abs(float(by_step.loc[7000]) - float(by_step.loc[6300]))
            rows.append({
                **dict(zip(keys, identity)), "outcome": outcome,
                "median_error_6300": float(by_step.loc[6300]), "median_error_7000": float(by_step.loc[7000]),
                "absolute_change": change, "maximum_absolute_change": limits[outcome],
                "tolerance_utilisation": change / limits[outcome], "pass": change <= limits[outcome],
            })
    target_gate = pd.DataFrame(rows)
    if len(target_gate) != 15 or not bool(target_gate["pass"].all()):
        errors.append(f"{int((~target_gate['pass']).sum()) if len(target_gate) else 15} of 15 G2c target comparisons fail")
    target_gate.to_csv(g2c / "convergence_g2c_target_gate.csv", index=False)

    g2b_gate = pd.read_csv(g2b / "convergence_g2b_group_gate.csv")
    target_ids = {
        (str(t["boundary"]), float(t["delta"]), float(t["c6"]), str(t["model"]))
        for t in amendment["execution"]["targets"]
    }
    unaffected = g2b_gate[
        ~g2b_gate.apply(lambda row: (str(row.stage4g_boundary), float(row.delta), float(row.c6), str(row.model_label)) in target_ids, axis=1)
    ].copy()
    composite = pd.concat([unaffected, target_gate], ignore_index=True, sort=False)
    if len(composite) != 72 or not bool(composite["pass"].all()):
        errors.append("the 72-row G2b/G2c composite convergence decision does not pass")
    composite.to_csv(g2c / "convergence_g2c_composite_gate.csv", index=False)

    base_repairs = pd.read_csv(g2b / "convergence_g2b_endpoint_repairs.csv")
    base_full = base_repairs[
        (base_repairs.stage4g_boundary == "open") & (base_repairs.model == "endpoint_full_b32") &
        (base_repairs.step == 4200) & ~((base_repairs.delta == 2.0) & (base_repairs.c6 == 5.0))
    ]
    final = sentinels[sentinels.step == 7000]
    pair_keys = ["stage4g_boundary", "delta", "c6", "seed", "step"]
    radial = final[final.model_label == "rbf_radial_b32"].set_index(pair_keys)
    endpoint = final[final.model_label == "endpoint_full_b32"].set_index(pair_keys)
    replacement = []
    for key in radial.index.intersection(endpoint.index):
        row = dict(zip(pair_keys, key))
        for outcome, column in ERRORS.items():
            row[f"repair_{outcome}"] = float(radial.loc[key, column] - endpoint.loc[key, column])
        replacement.append(row)
    repair = pd.concat([base_full, pd.DataFrame(replacement)], ignore_index=True, sort=False)
    cells = repair.groupby(["delta", "c6"], as_index=False).agg(
        median_density_repair=("repair_density", "median"),
        median_correlation_repair=("repair_correlation", "median"),
    )
    cells["pass"] = (cells.median_density_repair > 0) & (cells.median_correlation_repair > 0)
    cells.to_csv(g2c / "convergence_g2c_endpoint_full_stability.csv", index=False)
    endpoint_full_passes = int(cells["pass"].sum())
    if len(cells) != 3 or endpoint_full_passes < 2:
        errors.append(f"endpoint-full repairs density and correlation in only {endpoint_full_passes}/3 open cells")

    alias = json.loads(periodic_alias.read_text()) if periodic_alias.is_file() else {"validation": "FAIL", "passes": 0}
    if alias.get("validation") != "PASS" or alias.get("passes") != 30:
        errors.append("the exact same-parameter periodic alias gate does not pass 30/30")
    result = {
        "schema_version": "stage4g-g2c-recovery-gate-v1",
        "validation": "PASS" if not errors else "FAIL", "blocking_errors": errors,
        "decision_class": "SCIENTIFIC_GATE_PASS" if not errors else "SCIENTIFIC_GATE_FAIL",
        "g2b_preserved_validation": json.loads((g2b / "convergence_g2b_gate.json").read_text())["validation"],
        "target_runs": len(metrics), "target_exact_health_passes": int((metrics.scientific_validity_exact == "VALID").sum()) if len(metrics) else 0,
        "target_group_passes": int(target_gate["pass"].sum()) if len(target_gate) else 0,
        "composite_group_passes": int(composite["pass"].sum()) if len(composite) else 0,
        "endpoint_full_open_cells_passed": endpoint_full_passes,
        "endpoint_hybrid_status": "RETAINED_G2B_NEGATIVE_DIAGNOSTIC",
        "periodic_alias_passes": int(alias.get("passes", 0)),
        "maximum_target_tolerance_utilisation": float(target_gate.tolerance_utilisation.max()) if len(target_gate) else None,
    }
    (g2c / "convergence_g2c_gate.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--amendment", type=Path, default=Path("configs/stage4g_g2c_recovery.yaml"))
    parser.add_argument("--g2b", type=Path, required=True)
    parser.add_argument("--g2c", type=Path, required=True)
    parser.add_argument("--periodic-alias", type=Path, required=True)
    args = parser.parse_args()
    result = validate(load_yaml(args.amendment), args.g2b, args.g2c, args.periodic_alias)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["validation"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
