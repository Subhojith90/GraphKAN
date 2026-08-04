#!/usr/bin/env python3
"""Freeze, materialise and close the prospective Stage 4G protocol."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "stage4g_frozen.yaml"
OUTCOMES = (
    "exact_model_abs_energy_error_to_ed",
    "exact_model_density_mae_to_ed",
    "exact_model_connected_correlation_mae_to_ed",
)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def read_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("Stage 4G configuration must be a YAML mapping")
    return value


def cell_key(cell: dict[str, Any]) -> tuple[float, float]:
    return float(cell["delta"]), float(cell["c6"])


def validate_config(cfg: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    protocol = cfg.get("protocol", {})
    physics = cfg.get("physics", {})
    confirmation = physics.get("confirmation_cells", [])
    calibration = physics.get("calibration_cells", [])
    if protocol.get("schema_version") != "stage4g-protocol-v1":
        errors.append("schema_version must be stage4g-protocol-v1")
    if len(confirmation) != 3 or len({cell_key(c) for c in confirmation}) != 3:
        errors.append("exactly three unique confirmation cells are required")
    if len(calibration) != 2 or len({cell_key(c) for c in calibration}) != 2:
        errors.append("exactly two unique calibration cells are required")
    overlap = {cell_key(c) for c in confirmation} & {cell_key(c) for c in calibration}
    if overlap:
        errors.append(f"calibration and confirmation cells overlap: {sorted(overlap)}")
    if physics.get("boundaries") != ["open", "periodic"]:
        errors.append("boundaries must be frozen as [open, periodic]")
    if physics.get("seeds") != [0, 1, 2, 3, 4]:
        errors.append("confirmation seeds must be [0,1,2,3,4]")
    if int(physics.get("confirmation_n_sites", 0)) != 12:
        errors.append("confirmation_n_sites must be 12")
    primary = cfg.get("models", {}).get("primary", {})
    if set(primary) != {"rbf_radial_b32", "endpoint_full_b32", "endpoint_hybrid_b32"}:
        errors.append("the three primary linear models are not frozen correctly")
    candidates = cfg.get("models", {}).get("control_candidates", {})
    families = {spec.get("family") for spec in candidates.values()}
    if families != {"mlp", "rbm"}:
        errors.append("both MLP and RBM calibration families are required")
    steps = cfg.get("training", {}).get("exact_sentinel_steps", [])
    if steps != sorted(set(steps)) or steps[-1:] != [cfg.get("training", {}).get("convergence_budget")]:
        errors.append("exact sentinel steps must be unique, ordered, and end at convergence_budget")
    return {
        "schema_version": "stage4g-protocol-validation-v1",
        "validation": "PASS" if not errors else "FAIL",
        "blocking_errors": errors,
        "confirmation_cells": len(confirmation),
        "calibration_cells": len(calibration),
        "protocol_hash": sha256_bytes(canonical_bytes(cfg)),
    }


def model_spec(spec: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in spec.items() if key != "family"} | {"save_checkpoint": True}


def base_runner_config(
    cfg: dict[str, Any], *, phase: str, boundary: str, cells: list[dict[str, Any]],
    n_sites: int, seeds: list[int], models: dict[str, dict[str, Any]], n_iter: int,
) -> dict[str, Any]:
    training = cfg["training"]
    result = {
        "profile": f"stage4g_{phase}_{boundary}",
        "stage": f"Stage4G-{phase.upper()}",
        "schema_version": "stage4g-run-v1",
        "output_root": str(Path(cfg["protocol"]["output_root"]) / phase / boundary),
        "netket": {"inverted_ordering": False},
        "parameters": {
            "omega": [float(cfg["physics"]["omega"])],
            "cells": [{"delta": float(c["delta"]), "c6": float(c["c6"])} for c in cells],
        },
        "training": {
            "seeds": list(seeds),
            "n_samples": int(training["n_samples"]),
            "n_chains": int(training["n_chains"]),
            "n_discard_per_chain": int(training["n_discard_per_chain"]),
            "n_iter": int(n_iter),
            "chunk_size": int(training["chunk_size"]),
            "learning_rate": float(training["learning_rate"]),
            "diag_shift": float(training["diag_shift"]),
            "grad_clip_norm": float(training["grad_clip_norm"]),
            "n_basis": 12,
            "hidden": 1,
        },
        "sample_health_gate": {
            "min_energy_error": 1.0e-10,
            "min_unique_state_fraction": 0.005,
            **cfg["acceptance"]["health"],
        },
        "exact_health_gate": dict(cfg["acceptance"]["health"]),
        "model_specs": {label: model_spec(spec) for label, spec in models.items()},
        "geometry": {
            "family": "periodic_chain" if boundary == "periodic" else "chain",
            "n_sites": int(n_sites),
            "spacing": 1.0,
        },
        "stage4g_protocol_hash": sha256_bytes(canonical_bytes(cfg)),
    }
    if phase.startswith("convergence"):
        result["training"]["exact_sentinel_steps"] = list(training["exact_sentinel_steps"])
    return result


def roster_rows(
    phase: str, cells: list[dict[str, Any]], boundaries: list[str], models: list[str],
    seeds: list[int], n_sites: int,
) -> list[dict[str, Any]]:
    rows = []
    for boundary in boundaries:
        for cell in cells:
            for model in models:
                for seed in seeds:
                    run_id = (
                        f"stage4g__{phase}__N{n_sites}__B{boundary}"
                        f"__D{float(cell['delta']):.2f}".replace(".", "p")
                        + f"__C{float(cell['c6']):g}__M{model}__S{seed}"
                    )
                    rows.append({
                        "phase": phase, "run_id": run_id, "n_sites": n_sites,
                        "boundary": boundary, "delta": float(cell["delta"]), "c6": float(cell["c6"]),
                        "cell_role": cell["role"], "model": model, "seed": seed,
                    })
    return rows


def write_yaml(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def freeze(cfg: dict[str, Any], output: Path) -> dict[str, Any]:
    validation = validate_config(cfg)
    if validation["validation"] != "PASS":
        raise SystemExit(json.dumps(validation, indent=2))
    output.mkdir(parents=True, exist_ok=True)
    physics, models = cfg["physics"], cfg["models"]
    convergence_models = dict(models["primary"])
    convergence_models["mlp_b12_h7"] = models["control_candidates"]["mlp_b12_h7"]
    for boundary in physics["boundaries"]:
        write_yaml(output / "runner_configs" / f"convergence_{boundary}.yaml", base_runner_config(
            cfg, phase="convergence", boundary=boundary, cells=physics["confirmation_cells"], n_sites=10,
            seeds=physics["seeds"], models=convergence_models, n_iter=cfg["training"]["convergence_budget"],
        ))
        write_yaml(output / "runner_configs" / f"calibration_{boundary}.yaml", base_runner_config(
            cfg, phase="calibration", boundary=boundary, cells=physics["calibration_cells"],
            n_sites=physics["calibration_n_sites"], seeds=physics["calibration_seeds"],
            models=models["control_candidates"], n_iter=cfg["training"]["stage4f_budget"],
        ))
    rows = roster_rows("convergence", physics["confirmation_cells"], physics["boundaries"],
                       list(convergence_models), physics["seeds"], 10)
    rows += roster_rows("calibration", physics["calibration_cells"], physics["boundaries"],
                        list(models["control_candidates"]), physics["calibration_seeds"], physics["calibration_n_sites"])
    with (output / "preselection_roster.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    source_paths = [
        ROOT / "configs/stage4g_frozen.yaml",
        ROOT / "configs/stage4g_smoke_open.yaml",
        ROOT / "configs/stage4g_smoke_periodic.yaml",
        ROOT / "scripts/stage4g_protocol.py",
        ROOT / "scripts/stage4g_gauge.py",
        ROOT / "scripts/run_stage4g.py",
        ROOT / "scripts/validate_stage4g_smoke.py",
        ROOT / "scripts/stage4g_status.py",
        ROOT / "scripts/stage4g_merge_shards.py",
        ROOT / "scripts/validate_stage4g_convergence.py",
        ROOT / "scripts/analyze_stage4g.py",
        ROOT / "src/graphkan_amo_nqs/models/stage4g_flax_nqs.py",
        ROOT / "tests/test_stage4g_protocol.py",
        ROOT / ".github/workflows/stage4g-confirmation.yml",
    ]
    manifest = {
        **validation,
        "schema_version": "stage4g-protocol-freeze-v1",
        "frozen_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "preselection_runs": len(rows),
        "expected_convergence_runs": sum(row["phase"] == "convergence" for row in rows),
        "expected_calibration_runs": sum(row["phase"] == "calibration" for row in rows),
        "expected_confirmation_runs_after_selection": 180,
        "source_hashes": {str(path.relative_to(ROOT)): sha256_file(path) for path in source_paths},
    }
    write_json(output / "protocol_manifest.json", manifest)
    return manifest


def verify_freeze(cfg: dict[str, Any], output: Path) -> dict[str, Any]:
    manifest_path = output / "protocol_manifest.json"
    errors: list[str] = []
    if not manifest_path.is_file():
        return {"schema_version": "stage4g-protocol-freeze-verification-v1", "validation": "FAIL",
                "blocking_errors": [f"missing {manifest_path}"]}
    manifest = json.loads(manifest_path.read_text())
    expected_protocol_hash = sha256_bytes(canonical_bytes(cfg))
    if manifest.get("protocol_hash") != expected_protocol_hash:
        errors.append("protocol hash differs from the frozen manifest")
    for relative, expected in manifest.get("source_hashes", {}).items():
        path = ROOT / relative
        actual = sha256_file(path) if path.is_file() else None
        if actual != expected:
            errors.append(f"source hash mismatch: {relative}")
    roster = output / "preselection_roster.csv"
    if not roster.is_file() or sum(1 for _ in roster.open()) - 1 != int(manifest.get("preselection_runs", -1)):
        errors.append("preselection roster count differs from the frozen manifest")
    return {
        "schema_version": "stage4g-protocol-freeze-verification-v1",
        "validation": "PASS" if not errors else "FAIL", "blocking_errors": errors,
        "protocol_hash": expected_protocol_hash, "source_hashes_checked": len(manifest.get("source_hashes", {})),
    }


def select_controls(cfg: dict[str, Any], output: Path, metrics_paths: list[Path]) -> dict[str, Any]:
    frames = [pd.read_csv(path) for path in metrics_paths]
    data = pd.concat(frames, ignore_index=True)
    candidates = cfg["models"]["control_candidates"]
    expected = roster_rows(
        "calibration", cfg["physics"]["calibration_cells"], cfg["physics"]["boundaries"],
        list(candidates), cfg["physics"]["calibration_seeds"], cfg["physics"]["calibration_n_sites"],
    )
    keys = {("periodic" if "periodic" in str(row["boundary_condition"]) else "open",
             float(row["delta"]), float(row["c6"]), str(row["model_label"]), int(row["seed"])) for _, row in data.iterrows()}
    missing = [row["run_id"] for row in expected if (row["boundary"], row["delta"], row["c6"], row["model"], row["seed"]) not in keys]
    if missing:
        raise SystemExit(f"calibration is incomplete; {len(missing)} rows are missing")
    data["control_family"] = data["model_label"].map({label: spec["family"] for label, spec in candidates.items()})
    if data["control_family"].isna().any():
        raise SystemExit("calibration metrics contain an unregistered control label")
    rank_columns = []
    for outcome in OUTCOMES:
        column = f"rank_{outcome}"
        data[column] = data.groupby(["control_family", "boundary_condition", "delta", "c6"])[outcome].rank(method="average", pct=True)
        rank_columns.append(column)
    data["joint_fractional_rank"] = data[rank_columns].sum(axis=1)
    summary = data.groupby("model_label", as_index=False).agg(
        joint_fractional_rank=("joint_fractional_rank", "mean"),
        n_parameters=("n_parameters", "max"), runs=("run_id", "count"),
    )
    selections: dict[str, str] = {}
    for family in ("mlp", "rbm"):
        labels = [label for label, spec in candidates.items() if spec["family"] == family]
        group = summary[summary.model_label.isin(labels)].sort_values(
            ["joint_fractional_rank", "n_parameters", "model_label"], kind="mergesort"
        )
        selections[family] = str(group.iloc[0].model_label)
    selected_specs = {label: candidates[label] for label in selections.values()}
    confirmation_models = {**cfg["models"]["primary"], **cfg["models"]["capacity_control"], **selected_specs}
    for boundary in cfg["physics"]["boundaries"]:
        write_yaml(output / "runner_configs" / f"confirmation_{boundary}.yaml", base_runner_config(
            cfg, phase="confirmation", boundary=boundary, cells=cfg["physics"]["confirmation_cells"],
            n_sites=cfg["physics"]["confirmation_n_sites"], seeds=cfg["physics"]["seeds"],
            models=confirmation_models, n_iter=cfg["training"]["convergence_budget"],
        ))
    rows = roster_rows("confirmation", cfg["physics"]["confirmation_cells"], cfg["physics"]["boundaries"],
                       list(confirmation_models), cfg["physics"]["seeds"], cfg["physics"]["confirmation_n_sites"])
    pd.DataFrame(rows).to_csv(output / "confirmation_roster.csv", index=False)
    summary.to_csv(output / "control_calibration_ranking.csv", index=False)
    decision = {
        "schema_version": "stage4g-control-selection-v1", "validation": "PASS",
        "protocol_hash": sha256_bytes(canonical_bytes(cfg)), "selected": selections,
        "selection_rule": cfg["selection"]["rule"], "confirmation_runs": len(rows),
        "metrics_sha256": {str(path): sha256_file(path) for path in metrics_paths},
    }
    write_json(output / "control_selection.json", decision)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate", "freeze", "verify-freeze", "select-controls"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=ROOT / "results_stage4g" / "protocol")
    parser.add_argument("--metrics", nargs="*", type=Path, default=[])
    args = parser.parse_args()
    cfg = read_config(args.config)
    if args.command == "validate":
        result = validate_config(cfg)
    elif args.command == "freeze":
        result = freeze(cfg, args.output)
    elif args.command == "verify-freeze":
        result = verify_freeze(cfg, args.output)
    else:
        if not args.metrics:
            raise SystemExit("select-controls requires --metrics CALIBRATION_METRICS...")
        result = select_controls(cfg, args.output, args.metrics)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result.get("validation") != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
