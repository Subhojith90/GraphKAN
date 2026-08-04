#!/usr/bin/env python3
"""Freeze and verify the Stage 4G G2d checkpoint-stabilization protocol."""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from stage4g_protocol import (
    ROOT, base_runner_config, canonical_bytes, read_config, sha256_bytes,
    sha256_file, validate_config, write_json, write_yaml,
)

BASE_CONFIG = ROOT / "configs/stage4g_frozen.yaml"
AMENDMENT_CONFIG = ROOT / "configs/stage4g_g2d_stabilization.yaml"


def read_amendment(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("G2d amendment must be a mapping")
    return value


def validate(base: dict[str, Any], amendment: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    base_result = validate_config(base)
    if base_result["validation"] != "PASS":
        errors.extend(base_result["blocking_errors"])
    meta, execution, acceptance = amendment.get("amendment", {}), amendment.get("execution", {}), amendment.get("acceptance", {})
    if meta.get("schema_version") != "stage4g-g2d-stabilization-v1":
        errors.append("unexpected G2d schema")
    expected_execution = (7000, 3000, 0.000025, [8000, 9000, 10000])
    observed_execution = (
        int(execution.get("step_offset", 0)), int(execution.get("additional_iterations", 0)),
        float(execution.get("learning_rate", 0)), execution.get("exact_sentinel_steps"),
    )
    if observed_execution != expected_execution:
        errors.append("G2d stabilization schedule differs from the frozen 7000+3000 low-rate tail")
    targets = execution.get("targets", [])
    identities = {(t.get("boundary"), float(t.get("delta", 0)), float(t.get("c6", 0)), t.get("model")) for t in targets}
    if len(targets) != 4 or len(identities) != 4:
        errors.append("G2d requires four unique matched target groups")
    limits = {"energy": 0.016505923696, "density": 0.002038144719, "correlation": 0.000371585150}
    if {k: float(v) for k, v in acceptance.get("maximum_absolute_change", {}).items()} != limits:
        errors.append("G2d limits must remain identical to G2b")
    return {
        "schema_version": "stage4g-g2d-stabilization-validation-v1",
        "validation": "PASS" if not errors else "FAIL", "blocking_errors": errors,
        "base_protocol_hash": base_result["protocol_hash"],
        "amendment_hash": sha256_bytes(canonical_bytes(amendment)),
    }


def freeze(base: dict[str, Any], amendment: dict[str, Any], output: Path) -> dict[str, Any]:
    result = validate(base, amendment)
    if result["validation"] != "PASS":
        raise SystemExit(json.dumps(result, indent=2))
    output.mkdir(parents=True, exist_ok=True)
    physics, models = base["physics"], base["models"]
    selected = dict(models["primary"]); selected["mlp_b12_h7"] = models["control_candidates"]["mlp_b12_h7"]
    execution = amendment["execution"]
    for boundary in physics["boundaries"]:
        runner = base_runner_config(
            base, phase="convergence_g2d", boundary=boundary, cells=physics["confirmation_cells"],
            n_sites=10, seeds=physics["seeds"], models=selected,
            n_iter=int(execution["additional_iterations"]),
        )
        runner["training"]["step_offset"] = int(execution["step_offset"])
        runner["training"]["learning_rate"] = float(execution["learning_rate"])
        runner["training"]["exact_sentinel_steps"] = list(execution["exact_sentinel_steps"])
        runner["stage4g_g2d_amendment_hash"] = result["amendment_hash"]
        write_yaml(output / "runner_configs" / f"convergence_g2d_{boundary}.yaml", runner)
    rows = []
    for target in execution["targets"]:
        for seed in physics["seeds"]:
            rows.append({
                "phase": "convergence_g2d", "run_id": f"g2d__{target['tag']}__S{seed}", "n_sites": 10,
                "boundary": target["boundary"], "delta": float(target["delta"]), "c6": float(target["c6"]),
                "cell_role": "g2d_target", "model": target["model"], "seed": seed,
            })
    with (output / "preselection_roster.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    sources = [
        ROOT / "configs/stage4g_frozen.yaml", ROOT / "configs/stage4g_g2d_stabilization.yaml",
        ROOT / "docs/STAGE4G_G2D_STABILIZATION.md", ROOT / "scripts/stage4g_g2d_protocol.py",
        ROOT / "scripts/validate_stage4g_g2d.py", ROOT / "scripts/run_stage4g.py",
        ROOT / "scripts/stage4g_merge_shards.py", ROOT / ".github/workflows/stage4g-g2d-stabilization.yml",
    ]
    manifest = {
        **result, "schema_version": "stage4g-g2d-stabilization-freeze-v1",
        "frozen_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "expected_target_runs": len(rows),
        "source_hashes": {str(path.relative_to(ROOT)): sha256_file(path) for path in sources},
    }
    write_json(output / "protocol_manifest.json", manifest)
    return manifest


def verify(base: dict[str, Any], amendment: dict[str, Any], output: Path) -> dict[str, Any]:
    errors: list[str] = []
    manifest_path = output / "protocol_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    expected = validate(base, amendment)
    for key in ("base_protocol_hash", "amendment_hash"):
        if manifest.get(key) != expected.get(key): errors.append(f"{key} differs from the G2d manifest")
    for relative, digest in manifest.get("source_hashes", {}).items():
        path = ROOT / relative
        if not path.is_file() or sha256_file(path) != digest: errors.append(f"source hash mismatch: {relative}")
    roster = output / "preselection_roster.csv"
    if not roster.is_file() or sum(1 for _ in roster.open()) - 1 != 20: errors.append("G2d roster is incomplete")
    return {"schema_version": "stage4g-g2d-stabilization-freeze-verification-v1",
            "validation": "PASS" if not errors else "FAIL", "blocking_errors": errors,
            "source_hashes_checked": len(manifest.get("source_hashes", {}))}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=("validate", "freeze", "verify-freeze"))
    parser.add_argument("--base-config", type=Path, default=BASE_CONFIG); parser.add_argument("--amendment", type=Path, default=AMENDMENT_CONFIG)
    parser.add_argument("--output", type=Path, default=ROOT / "results_stage4g/g2d_protocol"); args = parser.parse_args()
    base, amendment = read_config(args.base_config), read_amendment(args.amendment)
    result = validate(base, amendment) if args.command == "validate" else (freeze(base, amendment, args.output) if args.command == "freeze" else verify(base, amendment, args.output))
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["validation"] != "PASS": raise SystemExit(1)


if __name__ == "__main__": main()
