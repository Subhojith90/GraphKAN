from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

from stage4f_tx import (
    RunSpec,
    Stage4FRegistry,
    atomic_promote,
    campaign_paths,
    hash_artifacts,
    load_campaign_config,
    sha256_file,
    validate_run_bundle,
    write_json,
    write_mock_run_bundle,
)


def main():
    parser = argparse.ArgumentParser(description="Stage 4F.1 transactional worker")
    parser.add_argument("--config", default="configs/stage4f_campaign.yaml")
    parser.add_argument("--lane", choices=["core", "control"], default="core")
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--max-runs", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_campaign_config(Path(args.config))
    paths = campaign_paths(cfg)
    reg = Stage4FRegistry(paths["registry"])

    completed = 0
    for _ in range(args.max_runs):
        row = reg.claim_next(args.lane, args.worker_id)
        if row is None:
            print("NO_PENDING_RUN")
            break

        run = RunSpec(
            campaign_id=row["campaign_id"],
            boundary=row["boundary"],
            delta=row["delta"],
            c6=row["c6"],
            model=row["model"],
            seed=int(row["seed"]),
            lane=row["lane"],
        )

        try:
            if not args.dry_run:
                raise RuntimeError(
                    "Stage 4F.1 worker physics execution is intentionally disabled. "
                    "Use --dry-run for scheduler validation. Enable VMC only after Stage 4F.1 fault tests pass."
                )

            tmp = write_mock_run_bundle(paths["runs"], run, cfg["completion_requirements"], fault=None)
            final = paths["runs"] / run.run_id
            atomic_promote(tmp, final)

            ok, errors = validate_run_bundle(final, cfg["completion_requirements"])
            if not ok:
                reg.mark_failed(run.run_id, args.worker_id, "; ".join(errors), retryable=True)
                continue

            reg.mark_complete(
                run.run_id,
                args.worker_id,
                final,
                sha256_file(final / "COMPLETE.json"),
                sha256_file(final / "resolved_config.yaml"),
            )
            completed += 1
            print(f"COMPLETE {run.run_id}")

        except Exception as exc:
            reg.mark_failed(run.run_id, args.worker_id, str(exc), retryable=True)
            print(f"FAILED {run.run_id}: {exc}", file=sys.stderr)

    print(json.dumps({"worker_id": args.worker_id, "completed": completed}, indent=2))


if __name__ == "__main__":
    main()
