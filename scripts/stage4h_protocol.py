#!/usr/bin/env python3
"""Validate, freeze, and verify the prospective Stage 4H protocol."""

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
DEFAULT_CONFIG = ROOT / "configs/stage4h_warmstart.yaml"
ARMS = (
    "radial_continuation",
    "endpoint_full_warmstart",
    "endpoint_hybrid_warmstart",
    "nongeometric_warmstart",
)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("Stage 4H configuration must be a mapping")
    return value


def validate_config(cfg: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    protocol, physics = cfg.get("protocol", {}), cfg.get("physics", {})
    training, acceptance, execution = (
        cfg.get("training", {}),
        cfg.get("acceptance", {}),
        cfg.get("execution", {}),
    )
    if protocol.get("schema_version") != "stage4h-warmstart-v1":
        errors.append("unexpected protocol schema")
    if (physics.get("n_sites"), physics.get("boundary"), physics.get("seeds")) != (
        12,
        "open",
        [0, 1, 2, 3, 4],
    ):
        errors.append("physics must be N=12 open with seeds 0..4")
    cells = physics.get("cells", [])
    identities = [(float(c.get("delta", 0)), float(c.get("c6", 0))) for c in cells]
    if (
        identities != [(1.25, 3.0), (1.5, 4.0), (2.0, 5.0)]
        or len({c.get("tag") for c in cells}) != 3
    ):
        errors.append("the three preregistered Stage 4H cells must remain frozen")
    if tuple(cfg.get("arms", {})) != ARMS:
        errors.append("the four matched continuation arms or their order changed")
    if (
        int(training.get("additional_iterations", 0)),
        training.get("exact_sentinel_steps"),
        int(training.get("immutable_checkpoint_interval", 0)),
        int(training.get("n_samples", 0)),
        float(training.get("learning_rate", 0)),
    ) != (6000, [3000, 6000], 250, 4096, 2.5e-5):
        errors.append("training must retain the frozen 6000-step low-rate schedule")
    if 6000 % int(training.get("chunk_size", 1)) or 250 % int(
        training.get("chunk_size", 1)
    ):
        errors.append("chunk size must divide checkpoint and terminal steps")
    expected_gates = (1e-12, 1e-4, 14, 4, 2, 30)
    observed_gates = (
        float(acceptance.get("transplant_max_log_amplitude_difference", 0)),
        float(acceptance.get("energy_safety_margin", 0)),
        int(acceptance.get("energy_safety_minimum_pairs", 0)),
        int(acceptance.get("paired_seed_sign_minimum", 0)),
        int(acceptance.get("scientific_rescue_minimum_cells", 0)),
        int(acceptance.get("periodic_alias_required_passes", 0)),
    )
    if observed_gates != expected_gates:
        errors.append("acceptance gates differ from the supervisory work order")
    if (
        int(execution.get("expected_runs", 0)) != 60
        or int(execution.get("expected_shards", 0)) != 12
    ):
        errors.append("execution must contain 60 runs in 12 cell-by-arm shards")
    return {
        "schema_version": "stage4h-protocol-validation-v1",
        "validation": "PASS" if not errors else "FAIL",
        "blocking_errors": errors,
        "protocol_hash": sha256_bytes(canonical_bytes(cfg)),
    }


def roster(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for cell in cfg["physics"]["cells"]:
        for arm in ARMS:
            shard = f"{cell['tag']}__{arm}"
            for seed in cfg["physics"]["seeds"]:
                rows.append(
                    {
                        "run_id": f"stage4h__{cell['tag']}__{arm}__S{seed}",
                        "shard": shard,
                        "n_sites": 12,
                        "boundary": "open",
                        "delta": float(cell["delta"]),
                        "c6": float(cell["c6"]),
                        "cell_tag": cell["tag"],
                        "cell_role": cell["role"],
                        "arm": arm,
                        "seed": int(seed),
                    }
                )
    return rows


def source_paths() -> list[Path]:
    return [
        ROOT / relative
        for relative in (
            "configs/stage4h_warmstart.yaml",
            "src/graphkan_amo_nqs/training/stage4h_transplant.py",
            "scripts/stage4h_protocol.py",
            "scripts/stage4h_transplant_preflight.py",
            "scripts/run_stage4h_local.py",
            "scripts/run_stage4h_warmstart.py",
            "scripts/merge_stage4h.py",
            "scripts/analyse_stage4h.py",
            "scripts/package_stage4h.py",
            "tests/test_stage4h_transplant.py",
            "tests/test_stage4h_protocol.py",
            "tests/test_stage4h_gate.py",
            "docs/STAGE4H_WARMSTART.md",
        )
    ]


def freeze(cfg: dict[str, Any], output: Path) -> dict[str, Any]:
    result = validate_config(cfg)
    if result["validation"] != "PASS":
        raise SystemExit(json.dumps(result, indent=2))
    output.mkdir(parents=True, exist_ok=True)
    (output / "frozen_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    rows = roster(cfg)
    with (output / "frozen_roster.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    missing_sources = [
        str(p.relative_to(ROOT)) for p in source_paths() if not p.is_file()
    ]
    if missing_sources:
        raise FileNotFoundError(f"Stage 4H source ledger incomplete: {missing_sources}")
    manifest = {
        **result,
        "schema_version": "stage4h-protocol-freeze-v1",
        "frozen_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "expected_runs": len(rows),
        "expected_shards": len({row["shard"] for row in rows}),
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
    manifest_path, roster_path, config_path = (
        output / "protocol_manifest.json",
        output / "frozen_roster.csv",
        output / "frozen_config.yaml",
    )
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    if not manifest:
        errors.append("protocol manifest missing")
    if not roster_path.is_file() or sum(1 for _ in roster_path.open()) - 1 != 60:
        errors.append("frozen roster is incomplete")
    if not config_path.is_file() or read_config(config_path) != cfg:
        errors.append("frozen config differs from requested config")
    if manifest.get("protocol_hash") != sha256_bytes(canonical_bytes(cfg)):
        errors.append("protocol hash mismatch")
    for relative, digest in manifest.get("source_hashes", {}).items():
        path = ROOT / relative
        if not path.is_file() or sha256_file(path) != digest:
            errors.append(f"source hash mismatch: {relative}")
    return {
        "schema_version": "stage4h-protocol-freeze-verification-v1",
        "validation": "PASS" if not errors else "FAIL",
        "blocking_errors": errors,
        "source_hashes_checked": len(manifest.get("source_hashes", {})),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate", "freeze", "verify-freeze"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results_stage4h/protocol"
    )
    args = parser.parse_args()
    cfg = read_config(args.config)
    result = (
        validate_config(cfg)
        if args.command == "validate"
        else (
            freeze(cfg, args.output)
            if args.command == "freeze"
            else verify_freeze(cfg, args.output)
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["validation"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
