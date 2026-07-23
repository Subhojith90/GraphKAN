#!/usr/bin/env python3
"""Raw-byte verifier for a Stage 4F-R.1 repository or extracted release."""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(root: Path) -> dict[str, Any]:
    errors: list[str] = []
    campaign = root / "results_stage4f_r" / "frozen"
    runs_root = campaign / "runs"
    run_dirs = sorted(path for path in runs_root.iterdir() if path.is_dir()) if runs_root.is_dir() else []
    payload_count = 0
    payload_matches = 0
    complete_count = 0

    if len(run_dirs) != 960:
        errors.append(f"canonical run directory count is {len(run_dirs)}, expected 960")
    for run in run_dirs:
        manifest_path = run / "artifact_manifest.json"
        complete_path = run / "COMPLETE.json"
        if not manifest_path.is_file() or not complete_path.is_file():
            errors.append(f"missing manifest or COMPLETE record: {run.name}")
            continue
        try:
            manifest = json.loads(manifest_path.read_text())
            complete = json.loads(complete_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"unreadable manifest for {run.name}: {exc}")
            continue
        files = manifest.get("files", {})
        if len(files) != 12:
            errors.append(f"{run.name}: artifact manifest has {len(files)} payloads, expected 12")
        for relative, expected in files.items():
            payload = run / relative
            payload_count += 1
            if not payload.is_file():
                errors.append(f"{run.name}: missing {relative}")
            elif sha256(payload) != expected:
                errors.append(f"{run.name}: raw-byte hash mismatch for {relative}")
            else:
                payload_matches += 1
        if complete.get("run_id") != run.name:
            errors.append(f"{run.name}: COMPLETE run identity mismatch")
        if complete.get("artifact_manifest_sha256") != sha256(manifest_path):
            errors.append(f"{run.name}: COMPLETE artifact-manifest link mismatch")
        complete_count += 1

    registry_path = campaign / "registry.sqlite"
    registry_rows = 0
    registry_hash_matches = 0
    if not registry_path.is_file():
        errors.append("missing campaign registry.sqlite")
    else:
        connection = sqlite3.connect(f"file:{registry_path}?mode=ro", uri=True)
        rows = connection.execute("SELECT run_id,status,complete_hash FROM runs ORDER BY run_id").fetchall()
        connection.close()
        registry_rows = len(rows)
        for run_id, status, expected in rows:
            complete_path = runs_root / run_id / "COMPLETE.json"
            if status != "COMPLETE":
                errors.append(f"registry row is not COMPLETE: {run_id} ({status})")
            elif complete_path.is_file() and sha256(complete_path) == expected:
                registry_hash_matches += 1
            else:
                errors.append(f"registry COMPLETE hash mismatch: {run_id}")
        if registry_rows != 960:
            errors.append(f"registry row count is {registry_rows}, expected 960")

    ledger_path = root / "release_integrity" / "frozen_source_ledger.json"
    source_matches = 0
    source_count = 0
    if not ledger_path.is_file():
        errors.append("missing frozen source ledger")
    else:
        ledger = json.loads(ledger_path.read_text())
        campaign_manifest = json.loads((campaign / "campaign_manifest.json").read_text())
        authoritative = campaign_manifest["protocol_identity"]["source_ledger"]
        supplied = ledger.get("source_ledger", {})
        if supplied != authoritative:
            errors.append("release campaign ledger is not exactly equal to campaign_manifest protocol_identity.source_ledger")
        source_count = len(supplied)
        for relative_text, expected_hash in supplied.items():
            relative = Path(relative_text)
            candidates = [
                root / "frozen_execution_source" / relative,
                root / "release_integrity" / "frozen_source_overrides" / relative,
                root / relative,
            ]
            source = next((candidate for candidate in candidates if candidate.is_file()), None)
            if source is None:
                errors.append(f"missing frozen source file: {relative}")
            elif sha256(source) != expected_hash:
                errors.append(f"frozen source hash mismatch: {relative}")
            else:
                source_matches += 1
        if source_count != 105:
            errors.append(f"frozen source ledger count is {source_count}, expected 105")

    tool_ledger_path = root / "release_integrity/release_tool_ledger.json"
    release_tool_matches = 0
    release_tool_count = 0
    if not tool_ledger_path.is_file():
        errors.append("missing separate release-tool ledger")
    else:
        tool_ledger = json.loads(tool_ledger_path.read_text()).get("source_ledger", {})
        release_tool_count = len(tool_ledger)
        for relative, expected in tool_ledger.items():
            path = root / relative
            if not path.is_file():
                errors.append(f"missing release tool: {relative}")
            elif sha256(path) != expected:
                errors.append(f"release-tool hash mismatch: {relative}")
            else:
                release_tool_matches += 1

    attestation_path = root / "release_integrity/generated/repository_attestation.json"
    attestation_status = "NOT_PRESENT_IN_DEVELOPMENT_TREE"
    if attestation_path.is_file():
        attestation = json.loads(attestation_path.read_text())
        listing = root / "release_integrity/generated/git_ls_tree.txt"
        valid = (
            len(attestation.get("source_commit", "")) == 40
            and len(attestation.get("source_tree", "")) == 40
            and attestation.get("tracked_worktree_clean") is True
            and listing.is_file()
            and sha256(listing) == attestation.get("git_ls_tree_sha256")
        )
        attestation_status = "PASS" if valid else "FAIL"
        if not valid:
            errors.append("repository commit/source-tree attestation is invalid")

    return {
        "schema_version": "stage4fr1-raw-integrity-v1",
        "validation": "PASS" if not errors else "FAIL",
        "canonical_runs": len(run_dirs),
        "payload_hashes": {"matched": payload_matches, "checked": payload_count, "expected": 11520},
        "complete_records": complete_count,
        "registry_rows": registry_rows,
        "registry_complete_hashes": registry_hash_matches,
        "frozen_source_hashes": {"matched": source_matches, "expected": 105},
        "release_tool_hashes": {"matched": release_tool_matches, "expected": release_tool_count},
        "repository_attestation": attestation_status,
        "blocking_errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--output")
    args = parser.parse_args()
    result = verify(Path(args.root).resolve())
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered)
    print(rendered, end="")
    if result["validation"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
