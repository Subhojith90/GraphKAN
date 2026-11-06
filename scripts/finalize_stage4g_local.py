#!/usr/bin/env python3
"""Assemble and validate the completed Stage 4G Actions artifacts locally."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from analyze_stage4g_g2d_final import analyse
from stage4g_final_protocol import read_resume, verify
from stage4g_merge_shards import merge


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_extract(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive) as handle:
        bad = handle.testzip()
        if bad is not None:
            raise ValueError(f"CRC failure in {archive.name}: {bad}")
        root = destination.resolve()
        for member in handle.infolist():
            target = (destination / member.filename).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"unsafe member in {archive.name}: {member.filename}")
        handle.extractall(destination)


def unique(artifacts: Path, prefix: str) -> Path:
    matches = sorted(artifacts.glob(f"{prefix}*.zip"))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {prefix}*.zip in {artifacts}, found {[p.name for p in matches]}")
    return matches[0]


def discover(artifacts: Path) -> tuple[Path, Path, list[Path]]:
    preflight = unique(artifacts, "stage4g-final-preflight")
    selection = unique(artifacts, "stage4g-final-control-selection")
    confirmations = sorted(artifacts.glob("stage4g-confirmation-*-final*.zip"))
    names = {path.name.split(".zip", 1)[0] for path in confirmations}
    if len(confirmations) != 12 or len(names) != 12:
        raise ValueError(f"expected 12 unique final confirmation ZIPs, found {len(confirmations)}/{len(names)}")
    return preflight, selection, confirmations


def build(artifacts: Path, output: Path, keep_work: Path | None = None) -> dict:
    preflight, selection, confirmations = discover(artifacts)
    temporary = Path(tempfile.mkdtemp(prefix="stage4g-final-local-"))
    try:
        results = temporary / "results_stage4g"
        downloaded = temporary / "downloaded"
        results.mkdir(); downloaded.mkdir()
        safe_extract(preflight, results)
        safe_extract(selection, results)
        for archive in confirmations:
            safe_extract(archive, downloaded / archive.stem)

        resume = read_resume(ROOT / "configs/stage4g_final_resume.yaml")
        frozen = verify(resume, results / "protocol")
        if frozen["validation"] != "PASS":
            raise ValueError(f"final protocol verification failed: {frozen['blocking_errors']}")
        merged = merge(downloaded, results, "confirmation")
        if merged["validation"] != "PASS":
            raise ValueError(f"confirmation merge failed: {merged['blocking_errors']}")
        decision = analyse(
            results, results / "analysis", results / "accepted_g2d",
            results / "environment/n12_periodic_alias_gate.json",
        )

        inputs = [preflight, selection, *confirmations]
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True
        ).stdout.strip()
        manifest = {
            "schema_version": "stage4g-local-finalization-v1",
            "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "host": platform.platform(), "git_commit": commit,
            "input_artifacts": {path.name: sha256(path) for path in inputs},
            "input_artifact_count": len(inputs), "confirmation_artifacts": len(confirmations),
            "protocol_verification": frozen, "merge_summary": merged, "final_decision": decision,
        }
        (results / "environment/local_finalization_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )

        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError(output)
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as handle:
            for path in sorted(results.rglob("*")):
                if path.is_file():
                    handle.write(path, Path("results_stage4g") / path.relative_to(results))
        with zipfile.ZipFile(output) as handle:
            bad = handle.testzip()
            if bad is not None:
                raise ValueError(f"output CRC failure: {bad}")
        digest = sha256(output)
        sidecar = output.with_suffix(output.suffix + ".sha256")
        sidecar.write_text(f"{digest}  {output.name}\n")
        manifest["output_archive"] = str(output)
        manifest["output_sha256"] = digest
        manifest["output_size_bytes"] = output.stat().st_size
        if keep_work is not None:
            if keep_work.exists():
                raise FileExistsError(keep_work)
            shutil.copytree(results, keep_work)
        return manifest
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, required=True, help="Directory containing the 14 downloaded Actions ZIPs.")
    parser.add_argument("--output", type=Path, required=True, help="New complete-evidence ZIP to create.")
    parser.add_argument("--keep-work", type=Path, help="Optional directory in which to retain the assembled evidence tree.")
    args = parser.parse_args()
    result = build(args.artifacts.resolve(), args.output.resolve(), args.keep_work.resolve() if args.keep_work else None)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["final_decision"]["validation"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
