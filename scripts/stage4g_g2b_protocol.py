#!/usr/bin/env python3
"""Freeze and verify the Stage 4G G2b convergence amendment."""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from stage4g_protocol import (
    ROOT,
    base_runner_config,
    canonical_bytes,
    read_config,
    roster_rows,
    sha256_bytes,
    sha256_file,
    validate_config,
    write_json,
    write_yaml,
)

BASE_CONFIG = ROOT / "configs" / "stage4g_frozen.yaml"
AMENDMENT_CONFIG = ROOT / "configs" / "stage4g_g2b_amendment.yaml"


def read_amendment(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("G2b amendment must be a YAML mapping")
    return value


def validate(base: dict[str, Any], amendment: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    base_validation = validate_config(base)
    if base_validation["validation"] != "PASS":
        errors.extend(base_validation["blocking_errors"])
    meta = amendment.get("amendment", {})
    execution = amendment.get("execution", {})
    acceptance = amendment.get("acceptance", {})
    if meta.get("schema_version") != "stage4g-g2b-amendment-v1":
        errors.append("unexpected amendment schema")
    if meta.get("predecessor_protocol_sha256") != base_validation["protocol_hash"]:
        errors.append("predecessor protocol hash does not identify the frozen Stage 4G protocol")
    if execution.get("exact_sentinel_steps") != [2800, 3500, 4200]:
        errors.append("G2b exact sentinels must be [2800, 3500, 4200]")
    if int(execution.get("n_iter", 0)) != 4200:
        errors.append("G2b must run freshly for 4200 iterations")
    thresholds = acceptance.get("maximum_absolute_change", {})
    if set(thresholds) != {"energy", "density", "correlation"} or any(float(v) <= 0 for v in thresholds.values()):
        errors.append("all three positive effect-scale thresholds are required")
    return {
        "schema_version": "stage4g-g2b-amendment-validation-v1",
        "validation": "PASS" if not errors else "FAIL",
        "blocking_errors": errors,
        "base_protocol_hash": base_validation["protocol_hash"],
        "amendment_hash": sha256_bytes(canonical_bytes(amendment)),
    }


def freeze(base: dict[str, Any], amendment: dict[str, Any], output: Path) -> dict[str, Any]:
    validation = validate(base, amendment)
    if validation["validation"] != "PASS":
        raise SystemExit(json.dumps(validation, indent=2))
    output.mkdir(parents=True, exist_ok=True)
    physics, models = base["physics"], base["models"]
    convergence_models = dict(models["primary"])
    convergence_models["mlp_b12_h7"] = models["control_candidates"]["mlp_b12_h7"]
    for boundary in physics["boundaries"]:
        runner = base_runner_config(
            base, phase="convergence_g2b", boundary=boundary,
            cells=physics["confirmation_cells"], n_sites=10, seeds=physics["seeds"],
            models=convergence_models, n_iter=int(amendment["execution"]["n_iter"]),
        )
        runner["training"]["exact_sentinel_steps"] = list(amendment["execution"]["exact_sentinel_steps"])
        runner["stage4g_amendment_hash"] = validation["amendment_hash"]
        write_yaml(output / "runner_configs" / f"convergence_g2b_{boundary}.yaml", runner)
        write_yaml(output / "runner_configs" / f"calibration_{boundary}.yaml", base_runner_config(
            base, phase="calibration", boundary=boundary, cells=physics["calibration_cells"],
            n_sites=physics["calibration_n_sites"], seeds=physics["calibration_seeds"],
            models=models["control_candidates"], n_iter=base["training"]["stage4f_budget"],
        ))
    rows = roster_rows(
        "convergence_g2b", physics["confirmation_cells"], physics["boundaries"],
        list(convergence_models), physics["seeds"], 10,
    )
    rows += roster_rows(
        "calibration", physics["calibration_cells"], physics["boundaries"],
        list(models["control_candidates"]), physics["calibration_seeds"], physics["calibration_n_sites"],
    )
    with (output / "preselection_roster.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    sources = [
        ROOT / "configs/stage4g_frozen.yaml",
        ROOT / "configs/stage4g_g2b_amendment.yaml",
        ROOT / "docs/STAGE4G_G2B_AMENDMENT.md",
        ROOT / "scripts/stage4g_g2b_protocol.py",
        ROOT / "scripts/validate_stage4g_g2b.py",
        ROOT / "scripts/stage4g_merge_shards.py",
        ROOT / "scripts/run_stage4g.py",
        ROOT / "scripts/analyze_stage4g.py",
        ROOT / "scripts/stage4g_protocol.py",
        ROOT / "src/graphkan_amo_nqs/models/stage4g_flax_nqs.py",
        ROOT / "tests/test_stage4g_protocol.py",
        ROOT / ".github/workflows/stage4g-confirmation.yml",
    ]
    manifest = {
        **validation,
        "schema_version": "stage4g-g2b-protocol-freeze-v1",
        "frozen_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "expected_g2b_runs": 120,
        "expected_calibration_runs": 72,
        "expected_confirmation_runs_after_selection": 180,
        "source_hashes": {str(path.relative_to(ROOT)): sha256_file(path) for path in sources},
    }
    write_json(output / "protocol_manifest.json", manifest)
    return manifest


def verify(base: dict[str, Any], amendment: dict[str, Any], output: Path) -> dict[str, Any]:
    errors: list[str] = []
    manifest_path = output / "protocol_manifest.json"
    if not manifest_path.is_file():
        errors.append(f"missing {manifest_path}")
        manifest: dict[str, Any] = {}
    else:
        manifest = json.loads(manifest_path.read_text())
    expected = validate(base, amendment)
    for key in ("base_protocol_hash", "amendment_hash"):
        if manifest.get(key) != expected.get(key):
            errors.append(f"{key} differs from the frozen manifest")
    for relative, digest in manifest.get("source_hashes", {}).items():
        path = ROOT / relative
        if not path.is_file() or sha256_file(path) != digest:
            errors.append(f"source hash mismatch: {relative}")
    roster = output / "preselection_roster.csv"
    if not roster.is_file() or sum(1 for _ in roster.open()) - 1 != 192:
        errors.append("preselection roster is missing or incomplete")
    return {
        "schema_version": "stage4g-g2b-freeze-verification-v1",
        "validation": "PASS" if not errors else "FAIL",
        "blocking_errors": errors,
        "source_hashes_checked": len(manifest.get("source_hashes", {})),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate", "freeze", "verify-freeze"))
    parser.add_argument("--base-config", type=Path, default=BASE_CONFIG)
    parser.add_argument("--amendment", type=Path, default=AMENDMENT_CONFIG)
    parser.add_argument("--output", type=Path, default=ROOT / "results_stage4g" / "g2b_protocol")
    args = parser.parse_args()
    base, amendment = read_config(args.base_config), read_amendment(args.amendment)
    result = validate(base, amendment) if args.command == "validate" else (
        freeze(base, amendment, args.output) if args.command == "freeze" else verify(base, amendment, args.output)
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["validation"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
