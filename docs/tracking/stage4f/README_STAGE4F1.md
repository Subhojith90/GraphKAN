# Stage 4F.1 Transactional Scheduler and Provenance Repair

This patch addresses the Stage 4F supervisory audit.

This is not a completed Stage 4F physics result. This is an engineering and provenance repair stage required before resuming the frozen Stage 4F open/periodic campaign.

## Purpose

Stage 4F.1 replaces unsafe shared-CSV completion logic with a transactional run registry and per-run artifact bundles.

A run is complete only when all required artifacts exist, hashes validate, the exit status is clean, the resolved config is archived, and COMPLETE.json is written last.

## Current Claim Boundary

Allowed:

- Stage 4F.1 implements transactional scheduler/provenance repair.
- Fault-injection tests reject incomplete runs.
- The registry prevents duplicate worker claims.
- Completion is semantic, not based on metrics-row presence alone.

Not allowed:

- completed Stage 4F physics result;
- generic KAN superiority;
- endpoint superiority;
- endpoint_full runtime blame;
- dropping endpoint_full after partial evidence;
- Stage 4G;
- manuscript readiness;
- public artifact readiness.

## Commands

Initialize the campaign registry:

PYTHONPATH=scripts python scripts/stage4f_campaign.py init --config configs/stage4f_campaign.yaml

Run fault-injection tests:

PYTHONPATH=scripts python scripts/stage4f_campaign.py fault-test --config configs/stage4f_campaign.yaml --plan configs/stage4f_fault_injection.yaml

Run release tests:

PYTHONPATH=scripts python -m pytest -q tests/test_stage4f1_transactional_scheduler.py -ra

Run a dry-run worker:

PYTHONPATH=scripts python scripts/stage4f_worker.py --config configs/stage4f_campaign.yaml --lane core --worker-id local-core-0 --max-runs 2 --dry-run

Summarize:

PYTHONPATH=scripts python scripts/summarize_stage4f.py --campaign results_stage4f1/transactional_campaign --strict

Package:

PYTHONPATH=scripts python scripts/package_stage4f_release.py --campaign results_stage4f1/transactional_campaign

## Next Step After This Patch

If Stage 4F.1 passes, resume the original frozen Stage 4F open/periodic campaign using the transactional runner.

Stage 4G remains blocked until Stage 4F-R campaign completion and independent replay.
