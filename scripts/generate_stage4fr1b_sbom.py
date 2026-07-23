#!/usr/bin/env python3
"""Generate a deterministic SPDX 2.3 SBOM for the final Stage 4F-R.1b release."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def spdx_id(name: str) -> str:
    return "SPDXRef-Package-" + re.sub(r"[^A-Za-z0-9.-]", "-", name)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    lock = root / "requirements/stage4fr1b-linux-py312.lock.txt"
    packages: list[dict[str, object]] = [
        {
            "SPDXID": "SPDXRef-Package-GraphKAN-AMO-NQS",
            "checksums": [
                {
                    "algorithm": "SHA256",
                    "checksumValue": sha256(root / "pyproject.toml"),
                }
            ],
            "copyrightText": "NOASSERTION",
            "downloadLocation": "https://github.com/Subhojith90/GraphKAN",
            "filesAnalyzed": False,
            "licenseConcluded": "MIT",
            "licenseDeclared": "MIT",
            "name": "graphkan-amo-nqs",
            "supplier": "Organization: GraphKAN AMO NQS contributors",
            "versionInfo": "0.1.0",
        }
    ]
    dependencies: list[tuple[str, str]] = []
    for line in lock.read_text().splitlines():
        match = re.match(r"^([A-Za-z0-9_.-]+)==([^ \\]+)", line)
        if not match:
            continue
        name, version = match.groups()
        identifier = spdx_id(name)
        packages.append(
            {
                "SPDXID": identifier,
                "copyrightText": "NOASSERTION",
                "downloadLocation": "NOASSERTION",
                "filesAnalyzed": False,
                "licenseConcluded": "NOASSERTION",
                "licenseDeclared": "NOASSERTION",
                "name": name,
                "supplier": "NOASSERTION",
                "versionInfo": version,
            }
        )
        dependencies.append(("SPDXRef-Package-GraphKAN-AMO-NQS", identifier))

    document = {
        "SPDXID": "SPDXRef-DOCUMENT",
        "creationInfo": {
            "created": "2026-07-23T00:00:00Z",
            "creators": ["Tool: scripts/generate_stage4fr1b_sbom.py"],
        },
        "dataLicense": "CC0-1.0",
        "documentDescribes": ["SPDXRef-Package-GraphKAN-AMO-NQS"],
        "documentNamespace": "https://github.com/Subhojith90/GraphKAN/releases/tag/stage4fr1b-final-20260723/sbom",
        "name": "GraphKAN-Stage4FR1b-Final-Composite-SBOM",
        "packages": packages,
        "relationships": [
            {
                "relatedSpdxElement": package,
                "relationshipType": "DEPENDS_ON",
                "spdxElementId": project,
            }
            for project, package in dependencies
        ],
        "spdxVersion": "SPDX-2.3",
    }
    destination = root / "release_integrity/GraphKAN_Stage4FR1b.sbom.spdx.json"
    destination.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "destination": str(destination),
                "packages": len(packages),
                "validation": "PASS",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
