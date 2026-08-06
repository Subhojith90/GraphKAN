#!/usr/bin/env python3
"""Freeze the final Stage 4G resume on top of the accepted G2d decision."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from stage4g_protocol import ROOT, freeze as freeze_base, read_config, sha256_file, verify_freeze, write_json

BASE = ROOT / "configs/stage4g_frozen.yaml"
RESUME = ROOT / "configs/stage4g_final_resume.yaml"


def read_resume(path: Path) -> dict:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict): raise ValueError("final resume must be a mapping")
    return value


def validate(resume: dict) -> dict:
    errors = []
    meta, execution, acceptance = resume.get("resume", {}), resume.get("execution", {}), resume.get("acceptance", {})
    if meta.get("schema_version") != "stage4g-final-resume-v1": errors.append("unexpected final-resume schema")
    if execution.get("calibration_runs") != 72 or execution.get("confirmation_runs") != 180 or execution.get("confirmation_n_sites") != 12:
        errors.append("final resume counts must be 72 calibration and 180 N=12 confirmation runs")
    if execution.get("rerun_convergence") is not False or execution.get("reuse_completed_convergence") is not True:
        errors.append("the final resume must reuse, not rerun, accepted G2d convergence")
    if acceptance.get("endpoint_full_minimum_open_cells_repairing_density_and_correlation") != 2:
        errors.append("endpoint-full must retain the two-of-three open-cell decision")
    if acceptance.get("endpoint_hybrid_role") != "diagnostic_nonblocking": errors.append("endpoint-hybrid must remain diagnostic")
    return {"schema_version": "stage4g-final-resume-validation-v1", "validation": "PASS" if not errors else "FAIL", "blocking_errors": errors}


def freeze(resume: dict, output: Path) -> dict:
    result = validate(resume)
    if result["validation"] != "PASS": raise SystemExit(json.dumps(result, indent=2))
    freeze_base(read_config(BASE), output)
    sources = [
        ROOT / "configs/stage4g_final_resume.yaml", ROOT / "scripts/stage4g_final_protocol.py",
        ROOT / "scripts/analyze_stage4g_g2d_final.py", ROOT / "scripts/validate_stage4g_periodic_alias.py",
        ROOT / "scripts/stage4g_merge_shards.py", ROOT / ".github/workflows/stage4g-final-confirmation.yml",
    ]
    manifest = {**result, "schema_version": "stage4g-final-resume-freeze-v1",
                "frozen_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                "source_hashes": {str(path.relative_to(ROOT)): sha256_file(path) for path in sources}}
    write_json(output / "final_resume_manifest.json", manifest); return manifest


def verify(resume: dict, output: Path) -> dict:
    errors = list(validate(resume)["blocking_errors"])
    base = verify_freeze(read_config(BASE), output)
    if base["validation"] != "PASS": errors.extend(base["blocking_errors"])
    path = output / "final_resume_manifest.json"
    manifest = json.loads(path.read_text()) if path.is_file() else {}
    for relative, digest in manifest.get("source_hashes", {}).items():
        source = ROOT / relative
        if not source.is_file() or sha256_file(source) != digest: errors.append(f"source hash mismatch: {relative}")
    return {"schema_version": "stage4g-final-resume-freeze-verification-v1",
            "validation": "PASS" if not errors else "FAIL", "blocking_errors": errors,
            "source_hashes_checked": len(manifest.get("source_hashes", {}))}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=("validate", "freeze", "verify-freeze"))
    parser.add_argument("--resume", type=Path, default=RESUME); parser.add_argument("--output", type=Path, default=ROOT / "results_stage4g/protocol")
    args = parser.parse_args(); resume = read_resume(args.resume)
    result = validate(resume) if args.command == "validate" else (freeze(resume, args.output) if args.command == "freeze" else verify(resume, args.output))
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["validation"] != "PASS": raise SystemExit(1)


if __name__ == "__main__": main()
