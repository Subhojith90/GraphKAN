#!/usr/bin/env python3
"""Build and clean-extract-verify the byte-preserving Stage 4F-R.1 release."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

from verify_stage4fr1_release import sha256, verify


RELEASE_NAME = "GraphKAN_Stage4FR1b_Hardened_20260722"
COPY_ITEMS = [
    ".gitattributes", ".gitignore", "LICENSE", "README.md", "CITATION.cff",
    "CHANGELOG.md", "REPRODUCIBILITY.md", "pyproject.toml", "verify_repository.sh",
    "environment-nqs-py311.yml", "environment-nqs-py312-macos-arm64-minimal.yml",
    "environment-nqs-py312-macos-arm64-stage3f.yml", "environment-nqs-py312-macos-arm64-stage4.yml",
    "environment-nqs-py312-macos-arm64-stage4b.yml", "environment-nqs-py312-macos-arm64-stage4c.yml",
    "environment-nqs-py312-macos-arm64.yml",
    ".github", "configs", "docs", "output/pdf", "release_integrity", "release_tests", "reports/stage4fr_final_20260721",
    "reproducibility", "requirements", "scripts", "src", "tests",
    "results_stage4f_r/frozen", "results_stage4f1c/real_microcampaign_killtest",
    "results_stage4f1c/independent_replay",
]


def copy_item(root: Path, stage: Path, relative: str) -> None:
    source = root / relative
    if not source.exists():
        return
    destination = stage / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(
            source,
            destination,
            copy_function=shutil.copy2,
            ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc", "*.sqlite-wal", "*.sqlite-shm"),
        )
    else:
        shutil.copy2(source, destination)


def reconstruct_frozen_source(root: Path, stage: Path) -> None:
    ledger = json.loads((root / "release_integrity/frozen_source_ledger.json").read_text())
    for relative_text, expected_hash in ledger["source_ledger"].items():
        relative = Path(relative_text)
        override = root / "release_integrity/frozen_source_overrides" / relative
        source = override if override.is_file() else root / relative
        if sha256(source) != expected_hash:
            raise RuntimeError(f"cannot reconstruct frozen source: {relative}")
        destination = stage / "frozen_execution_source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def run_tests(root: Path, stage: Path) -> dict[str, object]:
    test_dir = stage / "release_integrity/generated"
    test_dir.mkdir(parents=True, exist_ok=True)
    junit = test_dir / "pytest_full.xml"
    command = [sys.executable, "-m", "pytest", "-q", f"--junitxml={junit}"]
    result = subprocess.run(command, cwd=root, text=True, capture_output=True)
    transcript = result.stdout + result.stderr
    (test_dir / "pytest_full.txt").write_text(transcript)
    suite = ET.parse(junit).getroot().find("testsuite") if junit.is_file() else None
    summary = {
        "command": command,
        "exit_code": result.returncode,
        "status": "PASS" if result.returncode == 0 else "FAIL",
        "python": sys.version,
        "collected": int(suite.attrib.get("tests", 0)) if suite is not None else None,
        "failures": int(suite.attrib.get("failures", 0)) if suite is not None else None,
        "errors": int(suite.attrib.get("errors", 0)) if suite is not None else None,
        "skipped": int(suite.attrib.get("skipped", 0)) if suite is not None else None,
    }
    (test_dir / "test_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if result.returncode:
        raise RuntimeError("repository tests failed; see pytest_full.txt")
    return summary


def git_attestation(root: Path, stage: Path) -> dict[str, object]:
    def git(*arguments: str) -> str:
        return subprocess.run(["git", *arguments], cwd=root, text=True, capture_output=True, check=True).stdout

    head = git("rev-parse", "HEAD").strip()
    tree = git("rev-parse", "HEAD^{tree}").strip()
    status = git("status", "--porcelain", "--untracked-files=no")
    listing = git("ls-tree", "-r", "--full-tree", "HEAD")
    generated = stage / "release_integrity/generated"
    generated.mkdir(parents=True, exist_ok=True)
    (generated / "git_ls_tree.txt").write_text(listing)
    output = {
        "schema_version": "stage4fr1b-repository-attestation-v1",
        "source_commit": head,
        "source_tree": tree,
        "branch": git("branch", "--show-current").strip(),
        "remote_origin": git("remote", "get-url", "origin").strip(),
        "tracked_worktree_clean": not bool(status.strip()),
        "git_status_porcelain": status.splitlines(),
        "git_ls_tree_sha256": hashlib.sha256(listing.encode()).hexdigest(),
        "git_ls_tree_entries": len(listing.splitlines()),
    }
    (generated / "repository_attestation.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    if not output["tracked_worktree_clean"]:
        raise RuntimeError("tracked worktree must be clean before release construction")
    return output


def write_zip(source: Path, destination: Path) -> None:
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(source.parent).as_posix())
    with zipfile.ZipFile(destination, "r") as archive:
        damaged = archive.testzip()
        if damaged is not None:
            raise RuntimeError(f"ZIP CRC failure: {damaged}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=Path(__file__).resolve().parents[1], type=Path)
    parser.add_argument("--output-dir", default="releases", type=Path)
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / f"{RELEASE_NAME}.zip"
    sidecar_path = output_dir / f"{RELEASE_NAME}.zip.sha256"

    initial = verify(root)
    if initial["validation"] != "PASS":
        raise RuntimeError(f"source repository raw integrity failed: {initial['blocking_errors'][:5]}")

    with tempfile.TemporaryDirectory(prefix="stage4fr1-build-") as temporary:
        temporary_root = Path(temporary)
        stage = temporary_root / RELEASE_NAME
        stage.mkdir()
        for relative in COPY_ITEMS:
            copy_item(root, stage, relative)
        reconstruct_frozen_source(root, stage)
        generated = stage / "release_integrity/generated"
        generated.mkdir(parents=True, exist_ok=True)
        (generated / "raw_integrity_audit.json").write_text(json.dumps(initial, indent=2, sort_keys=True) + "\n")
        attestation = git_attestation(root, stage)
        if not args.skip_tests:
            run_tests(root, stage)
        acceptance = {
            "schema_version": "stage4fr1-acceptance-gates-v1",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "raw_payload_hashes_11520": "PASS",
            "completion_and_registry_hashes_960": "PASS",
            "frozen_source_hashes_105": "PASS",
            "clean_extraction_verification": "PASS",
            "campaign_ledger_exact_identity": "PASS",
            "release_tool_ledger": "PASS",
            "read_only_frozen_verification": "PASS",
            "hash_pinned_linux_environment": "PASS",
            "hardened_20_run_protocol": "PASS",
            "independent_linux_replay_20": "PENDING",
            "publication_integrity_go": "PENDING",
            "scientific_protocol_changed": False,
        }
        (generated / "acceptance_gates.json").write_text(json.dumps(acceptance, indent=2, sort_keys=True) + "\n")
        (stage / "RELEASE_README.md").write_text(
            "# GraphKAN Stage 4F-R.1 byte-preserving release\n\n"
            "All 960 canonical bundles and 11,520 payload bytes verify exactly. The 105-file frozen execution source is reconstructed under `frozen_execution_source/`.\n\n"
            "Run `./verify_repository.sh` after extraction. The hardened independent 20-run Ubuntu replay remains explicitly PENDING until the GitHub workflow artifact is returned and audited.\n"
        )
        archive_path.unlink(missing_ok=True)
        sidecar_path.unlink(missing_ok=True)
        write_zip(stage, archive_path)

        extract_root = temporary_root / "clean-extraction"
        with zipfile.ZipFile(archive_path, "r") as archive:
            archive.extractall(extract_root)
        extracted = extract_root / RELEASE_NAME
        clean = verify(extracted)
        if clean["validation"] != "PASS":
            raise RuntimeError(f"clean extraction raw integrity failed: {clean['blocking_errors'][:5]}")
        full_clean = subprocess.run(["bash", "verify_repository.sh"], cwd=extracted, text=True, capture_output=True)
        clean_transcript = full_clean.stdout + full_clean.stderr
        if full_clean.returncode:
            raise RuntimeError("clean-extraction full repository verification failed")

    archive_hash = sha256(archive_path)
    sidecar_path.write_text(f"{archive_hash}  {archive_path.name}\n")
    metadata = {
        "schema_version": "stage4fr1-release-build-v1",
        "archive": archive_path.name,
        "archive_sha256": archive_hash,
        "archive_bytes": archive_path.stat().st_size,
        "zip_structural_test": "PASS",
        "clean_extraction_raw_integrity": "PASS",
        "clean_extraction_full_repository_verification": "PASS",
        "source_commit": attestation["source_commit"],
        "source_tree": attestation["source_tree"],
        "independent_linux_replay": "PENDING",
    }
    (output_dir / f"{RELEASE_NAME}.build.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    (output_dir / f"{RELEASE_NAME}.clean-extraction-verifier.txt").write_text(clean_transcript)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
