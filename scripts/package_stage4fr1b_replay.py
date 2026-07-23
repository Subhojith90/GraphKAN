#!/usr/bin/env python3
"""Create a durable, checksummed archive of independent replay evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", default="results_stage4fr1b/independent_replay", type=Path)
    parser.add_argument("--output-dir", default="results_stage4fr1b/release", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / "GraphKAN_Stage4FR1b_Independent_Linux_Replay.zip"
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
        if args.replay.is_dir():
            for path in sorted(args.replay.rglob("*")):
                if path.is_file():
                    archive.write(path, (Path(args.replay.name) / path.relative_to(args.replay)).as_posix())
    with zipfile.ZipFile(destination) as archive:
        damaged = archive.testzip()
    if damaged:
        raise RuntimeError(f"ZIP CRC failure: {damaged}")
    digest = sha256(destination)
    sidecar = destination.with_suffix(destination.suffix + ".sha256")
    sidecar.write_text(f"{digest}  {destination.name}\n")
    print(json.dumps({"archive": str(destination), "sha256": digest, "zip_test": "PASS"}, indent=2))


if __name__ == "__main__":
    main()
