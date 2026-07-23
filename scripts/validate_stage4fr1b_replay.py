#!/usr/bin/env python3
"""Read-only clean-extraction regression for the accepted Stage 4F-R.1b replay."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


EXPECTED_ARCHIVE_SHA256 = "cf14747198ae8302d6bad298428faf3bb156d610841b9126590813be214ee8e6"
EXPECTED_COMMIT = "8992748cef1cd526891c3a15b084112567de58c9"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}\n"
            f"{result.stdout}{result.stderr}"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archive",
        type=Path,
        default="releases/GraphKAN_Stage4FR1b_Independent_Linux_Replay.zip",
    )
    parser.add_argument("--strict", action="store_true", help="Required compatibility flag.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.strict:
        raise SystemExit("--strict is required for the archival regression")

    root = Path(__file__).resolve().parents[1]
    archive = args.archive if args.archive.is_absolute() else root / args.archive
    errors: list[str] = []
    archive_hash = sha256(archive)
    if archive_hash != EXPECTED_ARCHIVE_SHA256:
        errors.append(f"replay archive SHA-256 mismatch: {archive_hash}")

    with tempfile.TemporaryDirectory(prefix="stage4fr1b-replay-regression-") as temporary:
        work = Path(temporary)
        extracted = work / "archive"
        campaign = work / "results_stage4fr1b" / "independent_replay"
        extracted.mkdir()
        with zipfile.ZipFile(archive) as handle:
            damaged = handle.testzip()
            if damaged:
                errors.append(f"ZIP CRC failure: {damaged}")
            handle.extractall(extracted)
        source_campaign = extracted / "independent_replay"
        campaign.parent.mkdir(parents=True)
        shutil.move(str(source_campaign), campaign)

        validator = run(
            [
                sys.executable,
                str(root / "scripts/stage4f1c_campaign.py"),
                "validate",
                "--campaign",
                "results_stage4fr1b/independent_replay",
                "--strict-integrity",
                "--reconcile-registry",
                "--require-terminal",
                "--require-complete",
                "--no-write",
            ],
            cwd=work,
        )
        validation = json.loads(validator.stdout)
        if validation.get("validation") != "PASS":
            errors.append("strict campaign validation failed")

        regenerated = work / "replay_comparison.regenerated.json"
        run(
            [
                sys.executable,
                str(root / "scripts/compare_stage4fr1b_replay.py"),
                "--replay",
                str(campaign),
                "--reference",
                str(root / "results_stage4f_r/frozen"),
                "--config",
                str(root / "configs/stage4fr1b_second_host_replay.yaml"),
                "--tolerances",
                str(root / "release_integrity/stage4fr1b_replay_tolerances.json"),
                "--expected-commit",
                EXPECTED_COMMIT,
                "--output",
                str(regenerated),
            ],
            cwd=root,
        )
        archived_comparison = campaign / "replay_comparison.json"
        byte_identical = regenerated.read_bytes() == archived_comparison.read_bytes()
        if not byte_identical:
            errors.append("regenerated comparison is not byte-identical")
        comparison = json.loads(regenerated.read_text())
        if comparison.get("validation") != "PASS":
            errors.append("replay comparison failed")

    result = {
        "archive_sha256": archive_hash,
        "blocking_errors": errors,
        "comparison_record_byte_identical": byte_identical,
        "periodic_collapse_checks_passed": sum(
            bool(item.get("pass")) for item in comparison["periodic_collapse_checks"]
        ),
        "replay_rows_passed": sum(bool(item.get("pass")) for item in comparison["comparisons"]),
        "schema_version": "stage4fr1b-replay-regression-v1",
        "strict_campaign_validation": validation["validation"],
        "validation": "PASS" if not errors else "FAIL",
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        destination = args.output if args.output.is_absolute() else root / args.output
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered)
    print(rendered, end="")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
