from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from stage4f_tx import Stage4FRegistry, validate_run_bundle, load_campaign_config, write_json


def main():
    parser = argparse.ArgumentParser(description="Summarize Stage 4F.1 transactional campaign")
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--config", default="configs/stage4f_campaign.yaml")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    root = Path(args.campaign)
    cfg = load_campaign_config(Path(args.config))
    reg = Stage4FRegistry(root / "registry.sqlite")
    summary = reg.summary()

    validation = []
    for d in sorted((root / "runs").glob("*")) if (root / "runs").exists() else []:
        ok, errors = validate_run_bundle(d, cfg["completion_requirements"])
        validation.append({"run_bundle": str(d), "valid": ok, "errors": errors})

    out = {
        "campaign": str(root),
        "registry_summary": summary,
        "validated_bundles": validation,
        "invalid_bundle_count": sum(1 for x in validation if not x["valid"]),
    }

    write_json(root / "stage4f1_summary.json", out)
    print(json.dumps(out, indent=2, sort_keys=True))

    if args.strict and out["invalid_bundle_count"] != 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
