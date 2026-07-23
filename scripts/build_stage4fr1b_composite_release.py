#!/usr/bin/env python3
"""Build the final composite release from one clean, immutable Git commit."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path


RELEASE_NAME = "GraphKAN_Stage4FR1b_Final_Composite_20260723"
TAG = "stage4fr1b-final-20260723"
CAMPAIGN_ARCHIVE = "GraphKAN_Stage4FR1b_Hardened_20260722.zip"
REPLAY_ARCHIVE = "GraphKAN_Stage4FR1b_Independent_Linux_Replay.zip"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=root, text=True, capture_output=True, check=True
    ).stdout.strip()


def zip_test(path: Path) -> None:
    with zipfile.ZipFile(path) as handle:
        damaged = handle.testzip()
        if damaged:
            raise RuntimeError(f"ZIP CRC failure in {path.name}: {damaged}")


def add_file(archive: zipfile.ZipFile, source: Path, arcname: str) -> None:
    info = zipfile.ZipInfo.from_file(source, arcname=arcname)
    info.compress_type = zipfile.ZIP_DEFLATED
    with source.open("rb") as handle:
        archive.writestr(info, handle.read(), compresslevel=6)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-untagged", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    commit = git(root, "rev-parse", "HEAD")
    tree = git(root, "rev-parse", "HEAD^{tree}")
    status = git(root, "status", "--porcelain", "--untracked-files=no")
    if status:
        raise RuntimeError("tracked worktree must be clean")
    tags = git(root, "tag", "--points-at", "HEAD").splitlines()
    if not args.allow_untagged and TAG not in tags:
        raise RuntimeError(f"HEAD must carry immutable release tag {TAG}")

    campaign = root / "releases" / CAMPAIGN_ARCHIVE
    replay = root / "releases" / REPLAY_ARCHIVE
    for archive in (campaign, replay):
        zip_test(archive)

    source_zip = output / f"{RELEASE_NAME}_Source.zip"
    source_tar = output / f"{RELEASE_NAME}_Source.tar.gz"
    subprocess.run(
        ["git", "archive", "--format=zip", f"--output={source_zip}", "HEAD"],
        cwd=root,
        check=True,
    )
    subprocess.run(
        ["git", "archive", "--format=tar.gz", f"--output={source_tar}", "HEAD"],
        cwd=root,
        check=True,
    )
    zip_test(source_zip)
    with zipfile.ZipFile(source_zip) as handle:
        mode = handle.getinfo("verify_repository.sh").external_attr >> 16
        if not mode & stat.S_IXUSR:
            raise RuntimeError("source ZIP does not preserve verify_repository.sh executable mode")
        forbidden = [
            name
            for name in handle.namelist()
            if "__pycache__/" in name
            or ".pytest_cache/" in name
            or name.endswith((".pyc", ".pyo"))
        ]
        if forbidden:
            raise RuntimeError(f"source ZIP contains cache files: {forbidden[:5]}")
    with tarfile.open(source_tar, "r:gz") as handle:
        verifier = handle.getmember("verify_repository.sh")
        if not verifier.mode & stat.S_IXUSR:
            raise RuntimeError("source TAR does not preserve verify_repository.sh executable mode")
        forbidden = [
            member.name
            for member in handle.getmembers()
            if "__pycache__/" in member.name
            or ".pytest_cache/" in member.name
            or member.name.endswith((".pyc", ".pyo"))
        ]
        if forbidden:
            raise RuntimeError(f"source TAR contains cache files: {forbidden[:5]}")

    source_zip_sidecar = output / f"{source_zip.name}.sha256"
    source_tar_sidecar = output / f"{source_tar.name}.sha256"
    source_zip_sidecar.write_text(f"{sha256(source_zip)}  {source_zip.name}\n")
    source_tar_sidecar.write_text(f"{sha256(source_tar)}  {source_tar.name}\n")

    with tempfile.TemporaryDirectory(prefix="stage4fr1b-source-verify-") as verify_temporary:
        extracted_source = Path(verify_temporary) / "source"
        extracted_source.mkdir()
        with zipfile.ZipFile(source_zip) as handle:
            handle.extractall(extracted_source)
        repository_verification = subprocess.run(
            ["bash", "verify_repository.sh"],
            cwd=extracted_source,
            text=True,
            capture_output=True,
        )
        repository_transcript = repository_verification.stdout + repository_verification.stderr
        if repository_verification.returncode:
            raise RuntimeError("clean source extraction failed full repository verification")
        replay_verification = subprocess.run(
            [
                sys.executable,
                "scripts/validate_stage4fr1b_replay.py",
                "--strict",
            ],
            cwd=extracted_source,
            text=True,
            capture_output=True,
        )
        replay_transcript = replay_verification.stdout + replay_verification.stderr
        if replay_verification.returncode:
            raise RuntimeError("clean source extraction failed replay regression")

    assets = {
        campaign.name: sha256(campaign),
        replay.name: sha256(replay),
        source_zip.name: sha256(source_zip),
        source_tar.name: sha256(source_tar),
        "GraphKAN_Stage4FR1b.sbom.spdx.json": sha256(
            root / "release_integrity/GraphKAN_Stage4FR1b.sbom.spdx.json"
        ),
    }
    provenance = {
        "assets": assets,
        "commit": commit,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_tree": tree,
        "immutable_tag": TAG,
        "replay_dispatch_commit": "8992748cef1cd526891c3a15b084112567de58c9",
        "schema_version": "stage4fr1b-final-composite-provenance-v1",
        "tracked_worktree_clean": True,
    }

    with tempfile.TemporaryDirectory(prefix="stage4fr1b-composite-") as temporary:
        stage = Path(temporary) / RELEASE_NAME
        stage.mkdir()
        copies = [
            campaign,
            root / "releases" / f"{CAMPAIGN_ARCHIVE}.sha256",
            root / "releases/GraphKAN_Stage4FR1b_Hardened_20260722.build.json",
            replay,
            root / "releases" / f"{REPLAY_ARCHIVE}.sha256",
            root / "releases/GraphKAN_Stage4FR1b_Independent_Linux_Replay.verification.json",
            source_zip,
            source_zip_sidecar,
            source_tar,
            source_tar_sidecar,
            root / "release_manifest.json",
            root / "release_integrity/commit_roles.json",
            root / "release_integrity/GraphKAN_Stage4FR1b.sbom.spdx.json",
            root / "docs/STAGE4FR1B_INDEPENDENT_REPLAY_CLOSURE.md",
            root / "docs/STAGE4FR1B_FINAL_COMPOSITE_RELEASE.md",
        ]
        for source in copies:
            shutil.copy2(source, stage / source.name)
        (stage / "COMPOSITE_PROVENANCE.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n"
        )
        (stage / "CLEAN_EXTRACTION_REPOSITORY_VERIFICATION.txt").write_text(
            repository_transcript
        )
        (stage / "CLEAN_EXTRACTION_REPLAY_REGRESSION.txt").write_text(
            replay_transcript
        )
        checksum_rows = [
            f"{sha256(path)}  {path.name}"
            for path in sorted(stage.iterdir())
            if path.is_file() and path.name != "SHA256SUMS.txt"
        ]
        (stage / "SHA256SUMS.txt").write_text("\n".join(checksum_rows) + "\n")

        composite = output / f"{RELEASE_NAME}.zip"
        composite.unlink(missing_ok=True)
        with zipfile.ZipFile(composite, "w") as handle:
            for source in sorted(stage.iterdir()):
                add_file(handle, source, f"{RELEASE_NAME}/{source.name}")
        zip_test(composite)
        sidecar = output / f"{composite.name}.sha256"
        sidecar.write_text(f"{sha256(composite)}  {composite.name}\n")

    build = {
        **provenance,
        "composite_archive": composite.name,
        "composite_archive_bytes": composite.stat().st_size,
        "composite_archive_sha256": sha256(composite),
        "source_tar_executable_mode": "PRESERVED",
        "clean_extraction_replay_regression": "PASS",
        "clean_extraction_repository_verification": "PASS",
        "source_zip_cache_hygiene": "PASS",
        "source_zip_executable_mode": "PRESERVED",
        "zip_structural_test": "PASS",
    }
    build_path = output / f"{RELEASE_NAME}.build.json"
    build_path.write_text(json.dumps(build, indent=2, sort_keys=True) + "\n")
    print(json.dumps(build, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
