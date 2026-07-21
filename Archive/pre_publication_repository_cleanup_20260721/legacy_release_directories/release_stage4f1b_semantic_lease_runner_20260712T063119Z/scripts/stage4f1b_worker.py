from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from stage4f1b_tx import (
    RunSpec,
    Stage4F1BRegistry,
    atomic_promote_no_overwrite,
    load_config,
    expected_runs,
    sha256_file,
    validate_run_bundle_semantic,
    write_json,
    write_valid_synthetic_bundle,
)


def row_to_runspec(row) -> RunSpec:
    return RunSpec(
        campaign_id=row["campaign_id"],
        protocol_hash=row["protocol_hash"],
        boundary=row["boundary"],
        delta=row["delta"],
        c6=row["c6"],
        model=row["model"],
        seed=int(row["seed"]),
        lane=row["lane"],
        priority=int(row["priority"]),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage4f1b_campaign.yaml")
    parser.add_argument("--lane", choices=["core", "control"], default="core")
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--max-runs", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--real-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    root = Path(cfg["campaign"]["output_root"])
    reg = Stage4F1BRegistry(root / "registry.sqlite")
    lease_seconds = int(cfg.get("scheduler", {}).get("lease_seconds", 300))
    backoff = int(cfg.get("scheduler", {}).get("backoff_seconds", 60))

    completed = 0

    for _ in range(args.max_runs):
        row = reg.claim_next(args.lane, args.worker_id, lease_seconds=lease_seconds)
        if row is None:
            print("NO_PENDING_RUN")
            break

        run = row_to_runspec(row)

        try:
            if args.dry_run:
                tmp = write_valid_synthetic_bundle(root, run)
            elif args.real_run:
                raise RuntimeError("real-run integration placeholder: enable after base-config artifact adapter is wired")
            else:
                raise RuntimeError("choose --dry-run or --real-run")

            final = root / "runs" / run.run_id
            atomic_promote_no_overwrite(tmp, final)

            ok, errors = validate_run_bundle_semantic(final, run.run_id)
            if not ok:
                reg.mark_failed(run.run_id, args.worker_id, "; ".join(errors), retryable=True, backoff_seconds=backoff)
                continue

            reg.mark_complete(run.run_id, args.worker_id, final, sha256_file(final / "COMPLETE.json"))
            completed += 1
            print(f"COMPLETE {run.run_id}")

        except Exception as exc:
            reg.mark_failed(run.run_id, args.worker_id, str(exc), retryable=True, backoff_seconds=backoff)
            print(f"FAILED {run.run_id}: {exc}", file=sys.stderr)

    print(json.dumps({"worker_id": args.worker_id, "completed": completed}, indent=2))


if __name__ == "__main__":
    main()
