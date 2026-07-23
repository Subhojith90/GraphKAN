#!/usr/bin/env python3
"""Prove that the release campaign ledger exactly equals its frozen authority."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    campaign_manifest_path = root / "results_stage4f_r/frozen/campaign_manifest.json"
    release_ledger_path = root / "release_integrity/frozen_source_ledger.json"
    campaign = json.loads(campaign_manifest_path.read_text())["protocol_identity"]["source_ledger"]
    release = json.loads(release_ledger_path.read_text())["source_ledger"]
    errors: list[str] = []
    if release != campaign:
        errors.append("release ledger does not exactly equal the campaign source ledger")
    matched = 0
    for relative, expected in campaign.items():
        override = root / "release_integrity/frozen_source_overrides" / relative
        source = override if override.is_file() else root / relative
        if not source.is_file():
            errors.append(f"missing campaign source: {relative}")
        elif sha256(source) != expected:
            errors.append(f"campaign source hash mismatch: {relative}")
        else:
            matched += 1
    output = {
        "schema_version": "stage4fr1b-campaign-ledger-identity-v1",
        "validation": "PASS" if not errors else "FAIL",
        "campaign_ledger_entries": len(campaign),
        "release_ledger_entries": len(release),
        "exact_ledger_equality": release == campaign,
        "source_hashes_matched": matched,
        "campaign_manifest_sha256": sha256(campaign_manifest_path),
        "blocking_errors": errors,
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

