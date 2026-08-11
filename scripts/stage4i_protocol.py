#!/usr/bin/env python3
"""Freeze and verify the prospective Stage 4I protocol and source ledger."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/stage4i_frozen.yaml"
ARMS = (
    "stage4i_radial",
    "stage4i_endpoint_full_raw",
    "stage4i_endpoint_full_orthogonal",
)


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("Stage 4I config must be a mapping")
    return value


def validate_config(cfg: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    protocol = cfg.get("protocol", {})
    gauge = cfg.get("gauge", {})
    pilot = cfg.get("counterfactual", {})
    training = pilot.get("training", {})
    acceptance = pilot.get("acceptance", {})
    execution = cfg.get("execution", {})
    if protocol.get("schema_version") != "stage4i-frozen-v1":
        errors.append("unexpected Stage 4I schema")
    if protocol.get("predecessor_decision") != "NEGATIVE_CLOSURE_PASS":
        errors.append("Stage 4H negative closure was not preserved")
    if gauge.get("sizes") != [10, 12] or gauge.get("seeds") != [0, 1, 2, 3, 4]:
        errors.append("G1 must use N=10/12 and five matched seeds")
    cell = pilot.get("cell", {})
    if (
        pilot.get("n_sites"),
        pilot.get("boundary"),
        float(cell.get("delta", 0)),
        float(cell.get("c6", 0)),
    ) != (12, "open", 2.0, 5.0):
        errors.append("G2 must remain the hard open N12 (2,5) cell")
    if tuple(pilot.get("arms", {})) != ARMS:
        errors.append("G2 arm roster or order changed")
    if (
        training.get("additional_iterations"),
        training.get("exact_sentinel_steps"),
        training.get("n_samples"),
        float(training.get("learning_rate", 0)),
    ) != (6000, [3000, 6000], 4096, 2.5e-5):
        errors.append("G2 continuation schedule changed")
    if (
        acceptance.get("energy_safe_seeds_required"),
        acceptance.get("correlation_positive_seeds_required"),
        float(acceptance.get("maximum_density_degradation", 0)),
        acceptance.get("negative_closure_correlation_worse_seeds"),
    ) != (5, 4, 1e-4, 4):
        errors.append("G2 decision thresholds changed")
    if (
        execution.get("expected_runs") != 15
        or execution.get("expected_shards") != 3
        or not execution.get("local_only")
    ):
        errors.append("Stage 4I must remain a 15-run, three-shard local campaign")
    return {
        "schema_version": "stage4i-protocol-validation-v1",
        "validation": "PASS" if not errors else "FAIL",
        "blocking_errors": errors,
        "protocol_hash": hashlib.sha256(canonical_bytes(cfg)).hexdigest(),
    }


def roster(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    cell = cfg["counterfactual"]["cell"]
    return [
        {
            "run_id": f"stage4i__{cell['tag']}__{arm}__S{seed}",
            "shard": arm,
            "n_sites": 12,
            "boundary": "open",
            "delta": float(cell["delta"]),
            "c6": float(cell["c6"]),
            "cell_tag": cell["tag"],
            "arm": arm,
            "seed": int(seed),
        }
        for arm in ARMS
        for seed in cfg["counterfactual"]["seeds"]
    ]


def source_paths() -> list[Path]:
    relative = (
        "configs/stage4i_frozen.yaml",
        "scripts/stage4i_protocol.py",
        "scripts/analyse_stage4i_gauge.py",
        "scripts/run_stage4i_counterfactual.py",
        "scripts/run_stage4i_local.py",
        "scripts/analyse_stage4i.py",
        "scripts/run_stage4i_test_profiles.py",
        "scripts/verify_stage4i_release.py",
        "scripts/package_stage4i.py",
        "src/graphkan_amo_nqs/training/stage4i_orthogonal.py",
        "src/graphkan_amo_nqs/models/stage4g_flax_nqs.py",
        "tests/test_stage4i_protocol.py",
        "tests/test_stage4i_orthogonal.py",
        "tests/test_stage4i_gate.py",
        "tests/test_stage4i_profiles.py",
        "docs/STAGE4I_CLOSURE.md",
        "requirements/stage4i-base-py312.txt",
        "requirements/stage4i-full-nqs-py312.txt",
    )
    return [ROOT / item for item in relative]


def freeze(cfg: dict[str, Any], output: Path) -> dict[str, Any]:
    result = validate_config(cfg)
    if result["validation"] != "PASS":
        raise ValueError(json.dumps(result, indent=2))
    missing = [str(p.relative_to(ROOT)) for p in source_paths() if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Stage 4I source ledger incomplete: {missing}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "frozen_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    rows = roster(cfg)
    with (output / "frozen_roster.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        **result,
        "schema_version": "stage4i-protocol-freeze-v1",
        "frozen_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "expected_runs": len(rows),
        "source_hashes": {
            str(p.relative_to(ROOT)): sha256_file(p) for p in source_paths()
        },
    }
    (output / "protocol_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def verify_freeze(cfg: dict[str, Any], output: Path) -> dict[str, Any]:
    errors = list(validate_config(cfg)["blocking_errors"])
    manifest_path = output / "protocol_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    if (
        manifest.get("protocol_hash")
        != hashlib.sha256(canonical_bytes(cfg)).hexdigest()
    ):
        errors.append("protocol hash mismatch")
    for relative, expected in manifest.get("source_hashes", {}).items():
        path = ROOT / relative
        if not path.is_file() or sha256_file(path) != expected:
            errors.append(f"source hash mismatch: {relative}")
    if not (output / "frozen_roster.csv").is_file():
        errors.append("frozen roster missing")
    return {
        "schema_version": "stage4i-freeze-verification-v1",
        "validation": "PASS" if not errors else "FAIL",
        "blocking_errors": errors,
        "source_hashes_checked": len(manifest.get("source_hashes", {})),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate", "freeze", "verify-freeze"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results_stage4i/protocol"
    )
    args = parser.parse_args()
    cfg = read_config(args.config)
    result = (
        validate_config(cfg)
        if args.command == "validate"
        else freeze(cfg, args.output)
        if args.command == "freeze"
        else verify_freeze(cfg, args.output)
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["validation"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
