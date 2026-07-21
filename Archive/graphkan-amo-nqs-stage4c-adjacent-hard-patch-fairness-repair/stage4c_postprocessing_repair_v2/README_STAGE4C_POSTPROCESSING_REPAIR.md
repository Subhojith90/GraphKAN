# Stage 4C Post-processing Repair v2

This overlay patches aggregation only. It must be applied **after** the frozen 630-row Stage 4C VMC run has finished. It never edits `raw/stage4c_metrics.csv` and it creates new outputs under `results_stage4c/hard_patch/aggregated_repaired/`.

## Required input
- `results_stage4c/hard_patch/raw/stage4c_metrics.csv`
- all 18 patch cells complete at 35 rows each

## Corrected behaviour
- pairs trajectories only by `(n_sites, delta, c6, seed)`
- checks patch completion before aggregation
- candidate exact-valid rate below 0.90 forces `FAIL`
- calculates non-domination separately in each patch cell
- reports exact and sampled validity separately
- uses control/candidate runtime ratio with the stated 1.25 bound

## Commands
```bash
unzip -q stage4c-postprocessing-repair-v2.zip -d .
cp -R stage4c_postprocessing_repair_v2/scripts/* scripts/
cp -R stage4c_postprocessing_repair_v2/tests/* tests/
PYTHONPATH=src python -m pytest -q tests/test_stage4c_repaired_pairing.py
PYTHONPATH=src python scripts/summarize_stage4c_repaired.py --results results_stage4c/hard_patch --candidate rbf_endpoint_sum_21p
cat results_stage4c/hard_patch/aggregated_repaired/stage4c_decision_gate_repaired.csv
```

Do not use `summarize_stage4c.py` for the release decision. Retain it untouched for provenance; use the repaired script's outputs.
