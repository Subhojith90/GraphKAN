#!/usr/bin/env python3
"""Create and verify the byte-preserving local Stage 4H evidence archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from graphkan_amo_nqs.training.stage4h_transplant import sha256_file


PREFIX = "GraphKAN_Stage4H_Complete_Evidence"


def package(campaign: Path, output: Path) -> dict:
    gate_path = campaign / "merged/analysis/stage4h_final_gate.json"
    if not gate_path.is_file():
        raise FileNotFoundError(
            "Stage 4H final gate is missing; finalize the campaign first"
        )
    gate = json.loads(gate_path.read_text())
    files = sorted(path for path in campaign.rglob("*") if path.is_file())
    if not files:
        raise ValueError("Stage 4H campaign contains no evidence files")
    payload_hashes = {
        str(path.relative_to(campaign)): sha256_file(path) for path in files
    }
    manifest = {
        "schema_version": "stage4h-local-evidence-package-v1",
        "decision_class": gate.get("decision_class", "UNKNOWN"),
        "scientific_gate_validation": gate.get("validation", "UNKNOWN"),
        "payload_count": len(files),
        "payload_hashes": payload_hashes,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path in files:
            relative = str(path.relative_to(campaign))
            info = zipfile.ZipInfo(
                f"{PREFIX}/{relative}", date_time=(2026, 8, 7, 0, 0, 0)
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(
                info,
                path.read_bytes(),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )
        info = zipfile.ZipInfo(
            f"{PREFIX}/PACKAGE_MANIFEST.json", date_time=(2026, 8, 7, 0, 0, 0)
        )
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o100644 << 16
        archive.writestr(
            info, json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n"
        )
    temporary.replace(output)
    with zipfile.ZipFile(output) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise RuntimeError(f"ZIP CRC failure: {bad_member}")
        stored = json.loads(archive.read(f"{PREFIX}/PACKAGE_MANIFEST.json"))
        for relative, expected in stored["payload_hashes"].items():
            observed = hashlib.sha256(archive.read(f"{PREFIX}/{relative}")).hexdigest()
            if observed != expected:
                raise RuntimeError(f"archived payload hash mismatch: {relative}")
    digest = sha256_file(output)
    sidecar = output.with_suffix(output.suffix + ".sha256")
    sidecar.write_text(f"{digest}  {output.name}\n")
    return {
        "schema_version": "stage4h-local-package-result-v1",
        "validation": "PASS",
        "archive": str(output.resolve()),
        "archive_sha256": digest,
        "sidecar": str(sidecar.resolve()),
        "payloads_verified": len(files),
        "decision_class": gate.get("decision_class", "UNKNOWN"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--campaign", type=Path, default=Path("results_stage4h/local_campaign_20260807")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("releases/GraphKAN_Stage4H_Complete_Evidence_Local_20260807.zip"),
    )
    args = parser.parse_args()
    result = package(args.campaign, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
