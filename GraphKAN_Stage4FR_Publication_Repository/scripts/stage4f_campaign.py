from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from stage4f_tx import (
    Stage4FRegistry,
    campaign_paths,
    init_campaign,
    load_campaign_config,
    run_fault_injection,
    validate_run_bundle,
    write_json,
)


def cmd_init(args):
    result = init_campaign(Path(args.config))
    print(json.dumps(result, indent=2, sort_keys=True))


def cmd_fault_test(args):
    result = run_fault_injection(Path(args.config), Path(args.plan))
    print(json.dumps(result, indent=2, sort_keys=True))


def cmd_validate(args):
    root = Path(args.campaign)
    registry_path = root / "registry.sqlite"
    if not registry_path.exists():
        raise SystemExit(f"missing registry: {registry_path}")

    reg = Stage4FRegistry(registry_path)
    summary = reg.summary()

    complete_dirs = list((root / "runs").glob("*")) if (root / "runs").exists() else []
    valid = 0
    invalid = 0
    invalid_details = []

    cfg_path = Path(args.config)
    requirements = {}
    if cfg_path.exists():
        cfg = load_campaign_config(cfg_path)
        requirements = cfg.get("completion_requirements", {})

    for d in complete_dirs:
        ok, errors = validate_run_bundle(d, requirements)
        if ok:
            valid += 1
        else:
            invalid += 1
            invalid_details.append({"bundle": str(d), "errors": errors})

    out = {
        "campaign": str(root),
        "registry_summary": summary,
        "valid_complete_bundles": valid,
        "invalid_complete_bundles": invalid,
        "invalid_details": invalid_details,
    }

    write_json(root / "validation_summary.json", out)
    print(json.dumps(out, indent=2, sort_keys=True))

    if invalid != 0:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description="Stage 4F.1 transactional campaign CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init")
    p.add_argument("--config", required=True)
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("fault-test")
    p.add_argument("--config", default="configs/stage4f_campaign.yaml")
    p.add_argument("--plan", required=True)
    p.set_defaults(func=cmd_fault_test)

    p = sub.add_parser("validate")
    p.add_argument("--campaign", required=True)
    p.add_argument("--config", default="configs/stage4f_campaign.yaml")
    p.set_defaults(func=cmd_validate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
