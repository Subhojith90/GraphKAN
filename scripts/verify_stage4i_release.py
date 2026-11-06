#!/usr/bin/env python3
"""Verify Stage 4I source closure from a clean Git extraction, read-only."""

from __future__ import annotations

import argparse
import json
import subprocess
import tarfile
import tempfile
from pathlib import Path

from stage4i_protocol import ROOT, sha256_file


def verify(protocol: Path, output: Path) -> dict:
    manifest = json.loads((protocol / "protocol_manifest.json").read_text())
    errors = []
    with tempfile.TemporaryDirectory(prefix="stage4i-clean-") as temporary:
        temp = Path(temporary)
        archive = temp / "repository.tar"
        with archive.open("wb") as handle:
            subprocess.run(
                ["git", "archive", "--format=tar", "HEAD"],
                cwd=ROOT,
                stdout=handle,
                check=True,
            )
        extraction = temp / "repo"
        extraction.mkdir()
        with tarfile.open(archive) as tar:
            tar.extractall(extraction, filter="data")
        for relative, expected in manifest.get("source_hashes", {}).items():
            path = extraction / relative
            if not path.is_file():
                errors.append(f"missing from Git extraction: {relative}")
            elif sha256_file(path) != expected:
                errors.append(f"hash mismatch in Git extraction: {relative}")
    result = {
        "schema_version": "stage4i-clean-extraction-v1",
        "validation": "PASS" if not errors else "FAIL",
        "blocking_errors": errors,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_hashes_checked": len(manifest.get("source_hashes", {})),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--protocol", type=Path, default=Path("results_stage4i/local_campaign/protocol")
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("results_stage4i/test_profiles/clean_extraction_gate.json"),
    )
    a = p.parse_args()
    result = verify(a.protocol, a.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["validation"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
