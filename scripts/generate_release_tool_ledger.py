#!/usr/bin/env python3
"""Generate the distinct post-campaign release/replay-tool ledger."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


FILES = [
    ".github/workflows/stage4fr1b-independent-replay.yml",
    "configs/stage4fr1b_second_host_replay.yaml",
    "release_integrity/stage4fr1b_replay_tolerances.json",
    "requirements/stage4fr1b-linux-py312.in",
    "requirements/stage4fr1b-linux-py312.lock.txt",
    "scripts/attest_stage4fr1b_environment.py",
    "scripts/build_stage4fr1_release.py",
    "scripts/calibrate_stage4fr1b_tolerances.py",
    "scripts/compare_stage4fr1_replay.py",
    "scripts/compare_stage4fr1b_replay.py",
    "scripts/finalize_stage4fr_publication.py",
    "scripts/package_stage4fr1b_replay.py",
    "scripts/validate_stage4fr1b_replay.py",
    "scripts/build_stage4fr1b_composite_release.py",
    "scripts/generate_stage4fr1b_sbom.py",
    "scripts/stage4f1b_campaign.py",
    "scripts/verify_campaign_ledger_identity.py",
    "scripts/verify_stage4fr1_release.py",
    "verify_repository.sh",
]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    ledger = {relative: hashlib.sha256((root / relative).read_bytes()).hexdigest() for relative in FILES}
    output = {
        "schema_version": "stage4fr1b-release-tool-ledger-v1",
        "authority": "post-campaign release and replay tooling; not the frozen campaign execution ledger",
        "file_count": len(ledger),
        "source_ledger": ledger,
    }
    destination = root / "release_integrity/release_tool_ledger.json"
    destination.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(destination), "files": len(ledger)}, indent=2))


if __name__ == "__main__":
    main()
