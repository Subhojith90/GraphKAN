# Stage 4C — Adjacent Hard-Regime Patch and Fairness-Repair Gate

## Purpose
Stage 4C tests whether the compact RBF endpoint-sum Pareto behavior survives an adjacent open-chain hard-regime patch after Stage 4B.1 artifact and gate repairs.

## Frozen design
- Systems: `N={6,8}`, open chains
- Parameters: `Omega=1`, `Delta={1.5,2.0,2.5}`, `C6={3,4,5}`
- Seeds: five matched seeds per model/cell
- Candidate: 21-parameter RBF endpoint-sum
- Controls: 21p RBF radial, 21p endpoint-product, MLP radial/endpoint at 64p and 128p
- All models use the same baseline VMC settings: 4096 samples, 64 chains, 700 iterations, SR shift 0.10.
- Exact energy, site-density MAE, connected-correlation MAE, runtime and sampled diagnostics are retained.

`smooth_endpoint_sum` is intentionally excluded because Stage 4B established numerical equivalence with endpoint-sum.

## Required gate
The candidate must have at least 90% exact-valid candidate rows and be non-dominated in at least 70% of exact-valid patch cells under exact energy, density, connected correlation, and the documented runtime condition.

## Run
```bash
bash scripts/run_stage4c_clean_replay.sh
bash scripts/run_stage4c_batch.sh configs/netket_stage4c_hard_patch.yaml --overwrite
PYTHONPATH=src python scripts/summarize_stage4c.py --results results_stage4c/hard_patch --candidate rbf_endpoint_sum_21p
```

Recommended CPU-friendly execution uses two non-overlapping batches. Use `--overwrite` only for the first batch; the second batch appends native rows. `--resume` safely skips run IDs already present after an interruption.
```bash
bash scripts/run_stage4c_batch.sh configs/netket_stage4c_hard_patch.yaml --only-n-sites 6 --overwrite
bash scripts/run_stage4c_batch.sh configs/netket_stage4c_hard_patch.yaml --only-n-sites 8 --resume
```
Do not start two overlapping invocations against the same output directory.
