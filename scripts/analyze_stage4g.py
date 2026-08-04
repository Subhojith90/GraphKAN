#!/usr/bin/env python3
"""Evaluate the prospectively frozen Stage 4G acceptance gates."""
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


def read_phase(root: Path, phase: str, filename: str) -> pd.DataFrame:
    frames = []
    for boundary in ("open", "periodic"):
        path = root / phase / boundary / "raw" / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        frame["stage4g_boundary"] = boundary
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def enrich_sentinel_metadata(sentinels: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in ("delta", "c6", "boundary_condition", "n_sites") if column not in sentinels]
    if missing:
        lookup = metrics[["run_id", *missing]].drop_duplicates("run_id")
        sentinels = sentinels.merge(lookup, on="run_id", how="left", validate="many_to_one")
    if sentinels[["delta", "c6"]].isna().any().any():
        raise ValueError("sentinel metadata could not be reconstructed")
    return sentinels


def checkpoint_path(root: Path, phase: str, boundary: str, run_id: str) -> Path:
    safe = run_id.replace(":", "__").replace("/", "_")
    path = root / phase / boundary / "checkpoints" / f"{safe}.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def arrays_identical(left: Path, right: Path) -> bool:
    with np.load(left, allow_pickle=False) as a, np.load(right, allow_pickle=False) as b:
        if sorted(a.files) != sorted(b.files):
            return False
        return all(np.array_equal(np.asarray(a[name]), np.asarray(b[name])) for name in sorted(a.files))


def analyse(cfg: dict[str, Any], root: Path, output: Path, convergence_phase: str = "convergence") -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    blocking: list[str] = []

    convergence_metrics = read_phase(root, convergence_phase, "stage4g_metrics.csv")
    convergence = enrich_sentinel_metadata(
        read_phase(root, convergence_phase, "stage4g_exact_sentinels.csv"), convergence_metrics
    )
    if convergence_phase == "convergence_g2b":
        gate_path = root / convergence_phase / "convergence_g2b_gate.json"
        gate_summary = json.loads(gate_path.read_text()) if gate_path.is_file() else {"validation": "FAIL"}
        expected_convergence = 360
        convergence_gate = pd.read_csv(root / convergence_phase / "convergence_g2b_group_gate.csv")
        if gate_summary.get("validation") != "PASS":
            blocking.append("the amended G2b convergence gate does not pass")
        convergence_gate.to_csv(output / "convergence_gate.csv", index=False)
    else:
        expected_convergence = 120 * len(cfg["training"]["exact_sentinel_steps"])
    if len(convergence) != expected_convergence:
        blocking.append(f"convergence sentinel rows {len(convergence)} != {expected_convergence}")
    if convergence_phase != "convergence_g2b":
        convergence_gate_rows = []
        threshold = float(cfg["acceptance"]["convergence"]["max_relative_change_1400_to_2100"])
        floors = cfg["acceptance"]["convergence"]["absolute_floors"]
        group_columns = ["stage4g_boundary", "delta", "c6", "model_label", "seed"]
        for keys, group in convergence.groupby(group_columns):
            by_step = group.set_index("step")
            if not {1400, 2100}.issubset(by_step.index):
                continue
            for outcome, column in ERRORS.items():
                earlier, final = float(by_step.loc[1400, column]), float(by_step.loc[2100, column])
                utilisation = abs(earlier - final) / max(abs(final), float(floors[outcome]))
                convergence_gate_rows.append({
                    **dict(zip(group_columns, keys)), "outcome": outcome, "value_1400": earlier,
                    "value_2100": final, "relative_change": utilisation, "pass": utilisation <= threshold,
                })
        convergence_gate = pd.DataFrame(convergence_gate_rows)
        if len(convergence_gate) != 360 or not bool(convergence_gate["pass"].all()):
            blocking.append("not all 360 convergence/outcome comparisons pass")
        convergence_gate.to_csv(output / "convergence_gate.csv", index=False)

    metrics = read_phase(root, "confirmation", "stage4g_metrics.csv")
    sites = read_phase(root, "confirmation", "stage4g_site_observables.csv")
    correlations = read_phase(root, "confirmation", "stage4g_corr_observables.csv")
    if len(metrics) != 180 or metrics.run_id.nunique() != 180:
        blocking.append(f"confirmation contains {len(metrics)} rows/{metrics.run_id.nunique()} unique, expected 180")
    failed_health = metrics[metrics.scientific_validity_exact != "VALID"]
    if len(failed_health):
        blocking.append(f"{len(failed_health)} confirmation runs fail the exact-health gate")

    paired_rows = []
    key = ["stage4g_boundary", "delta", "c6", "seed"]
    radial = metrics[metrics.model_label == "rbf_radial_b32"].set_index(key)
    for model in ("endpoint_full_b32", "endpoint_hybrid_b32"):
        endpoint = metrics[metrics.model_label == model].set_index(key)
        for index in radial.index.intersection(endpoint.index):
            row = dict(zip(key, index))
            row["model"] = model
            for outcome, column in ERRORS.items():
                row[f"repair_{outcome}"] = float(radial.loc[index, column] - endpoint.loc[index, column])
            paired_rows.append(row)
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(output / "finite_size_paired_effects.csv", index=False)
    open_cell = paired[paired.stage4g_boundary == "open"].groupby(["model", "delta", "c6"], as_index=False).agg(
        median_energy_repair=("repair_energy", "median"),
        median_density_repair=("repair_density", "median"),
        median_correlation_repair=("repair_correlation", "median"),
    )
    open_cell["density_and_correlation_repaired"] = (
        (open_cell.median_density_repair > 0) & (open_cell.median_correlation_repair > 0)
    )
    open_cell.to_csv(output / "open_cell_decisions.csv", index=False)
    minimum = int(cfg["acceptance"]["finite_size"]["minimum_open_cells_preserving_density_and_correlation_repair"])
    for model, group in open_cell.groupby("model"):
        passed = int(group.density_and_correlation_repaired.sum())
        if passed < minimum:
            blocking.append(f"{model} repairs density and correlation in only {passed}/3 open cells")

    periodic = metrics[metrics.stage4g_boundary == "periodic"]
    periodic_rows = []
    for (delta, c6, seed), group in periodic.groupby(["delta", "c6", "seed"]):
        radial_row = group[group.model_label == "rbf_radial_b32"].iloc[0]
        for model in ("endpoint_full_b32", "endpoint_hybrid_b32"):
            endpoint_row = group[group.model_label == model].iloc[0]
            checkpoint_equal = arrays_identical(
                checkpoint_path(root, "confirmation", "periodic", radial_row.run_id),
                checkpoint_path(root, "confirmation", "periodic", endpoint_row.run_id),
            )
            scalar_diff = max(abs(float(radial_row[column]) - float(endpoint_row[column])) for column in ERRORS.values())
            radial_density = sites[sites.run_id == radial_row.run_id].sort_values("site").exact_model_density.to_numpy()
            endpoint_density = sites[sites.run_id == endpoint_row.run_id].sort_values("site").exact_model_density.to_numpy()
            radial_corr = correlations[correlations.run_id == radial_row.run_id].sort_values(["i", "j"]).exact_model_connected.to_numpy()
            endpoint_corr = correlations[correlations.run_id == endpoint_row.run_id].sort_values(["i", "j"]).exact_model_connected.to_numpy()
            vector_equal = np.array_equal(radial_density, endpoint_density) and np.array_equal(radial_corr, endpoint_corr)
            periodic_rows.append({
                "delta": delta, "c6": c6, "seed": seed, "model": model,
                "checkpoint_arrays_identical": checkpoint_equal, "full_observable_arrays_identical": vector_equal,
                "maximum_scalar_error_difference": scalar_diff,
                "pass": checkpoint_equal and vector_equal and scalar_diff == 0.0,
            })
    periodic_gate = pd.DataFrame(periodic_rows)
    periodic_gate.to_csv(output / "periodic_collapse_gate.csv", index=False)
    if len(periodic_gate) != 30 or not bool(periodic_gate["pass"].all()):
        blocking.append("periodic radial/endpoint collapse is not exact in all 30 comparisons")

    summary = {
        "schema_version": "stage4g-final-decision-v1",
        "validation": "PASS" if not blocking else "FAIL",
        "blocking_errors": blocking,
        "confirmation_runs": int(len(metrics)),
        "exact_health_passes": int((metrics.scientific_validity_exact == "VALID").sum()),
        "convergence_comparisons": int(len(convergence_gate)),
        "convergence_passes": int(convergence_gate["pass"].sum()) if len(convergence_gate) else 0,
        "periodic_collapse_comparisons": int(len(periodic_gate)),
        "periodic_collapse_passes": int(periodic_gate["pass"].sum()) if len(periodic_gate) else 0,
        "open_cell_decisions": open_cell.to_dict(orient="records"),
        "scientific_decision": (
            "Stage 4G supports a finite-size-stable boundary-conditioned mechanism map"
            if not blocking else "Stage 4G does not yet close the preregistered confirmation gate"
        ),
    }
    (output / "stage4g_final_decision.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "stage4g_frozen.yaml")
    parser.add_argument("--results", type=Path, default=ROOT / "results_stage4g")
    parser.add_argument("--output", type=Path, default=ROOT / "results_stage4g" / "analysis")
    parser.add_argument("--convergence-phase", choices=("convergence", "convergence_g2b"), default="convergence")
    args = parser.parse_args()
    result = analyse(load_yaml(args.config), args.results, args.output, args.convergence_phase)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["validation"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
