#!/usr/bin/env python3
"""Create a byte-verifiable Stage 4I G0-G2 evidence package."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

from stage4i_protocol import ROOT, source_paths


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def package(campaign: Path, output: Path) -> dict:
    required = [
        campaign / "protocol",
        campaign / "gauge",
        campaign / "merged",
        Path("results_stage4i/test_profiles"),
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Stage 4I evidence incomplete: {missing}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="stage4i-package-") as t:
        root = Path(t) / "GraphKAN_Stage4I_Evidence"
        root.mkdir()
        for item in required:
            shutil.copytree(
                item,
                root / item.name
                if item != Path("results_stage4i/test_profiles")
                else "test_profiles",
            )
        for source in source_paths():
            target = root / "source" / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        files = sorted(p for p in root.rglob("*") if p.is_file())
        manifest = {str(p.relative_to(root)): sha(p) for p in files}
        (root / "SHA256SUMS.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        with zipfile.ZipFile(
            output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as z:
            for p in sorted(root.rglob("*")):
                if p.is_file():
                    z.write(p, p.relative_to(root.parent))
    digest = sha(output)
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{digest}  {output.name}\n"
    )
    return {
        "schema_version": "stage4i-package-v1",
        "validation": "PASS",
        "archive": str(output),
        "sha256": digest,
        "bytes": output.stat().st_size,
        "payloads": len(manifest) + 1,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--campaign", type=Path, default=Path("results_stage4i/local_campaign")
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("releases/GraphKAN_Stage4I_G0_G2_Evidence_Local_20260810.zip"),
    )
    a = p.parse_args()
    result = package(a.campaign, a.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
