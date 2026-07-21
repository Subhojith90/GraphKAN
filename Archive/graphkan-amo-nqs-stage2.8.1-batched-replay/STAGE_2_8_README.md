# Stage 2.8 — Batched Execution Gate

Stage 2.8 replaces the all-or-nothing Stage 2.7 launch pattern with independently reviewable batches.

## Why

Stage 2.7 expanded to 240 hard-regime runs before producing an end-of-profile report. Stage 2.8 is designed for CPU execution and checkpoints every completed run directly to CSV.

## Batch order

1. **Batch A** — N=6 smoke: 6 runs, 200 iterations.
2. **Batch B** — N=8 smoke: 6 runs, 200 iterations. Run only after Batch A is inspected.
3. **Batch C** — N=6 diagnostic: 9 runs, 450 iterations. Run only after A/B complete without execution errors.

Do not run the old Stage 2.7 full hard-regime or fairness configs from this package.

## Safe interruption

Each completed run is appended to `raw/stage2_7_metrics.csv`, site observables, correlation observables, and history immediately. A Ctrl+C interruption therefore retains completed work.

## Commands

```bash
PYTHONPATH=src python scripts/run_stage2_7_hard_regime.py --config configs/netket_stage2_8_batch_a_n6_smoke.yaml --overwrite
PYTHONPATH=src python scripts/summarize_stage2_8_batch.py --results results_stage2_8/batch_a_n6_smoke
```

Only after reviewing Batch A, run Batch B; then Batch C.
