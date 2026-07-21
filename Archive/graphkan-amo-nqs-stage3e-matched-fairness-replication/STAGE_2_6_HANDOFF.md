# Stage 2.6 Handoff — Convention-Locked Observable Gate

## Why this stage exists
Stage 2.5 passed an internal phase-aware validation gate but a supervisory audit identified three blockers to scale-up: unspecified NetKet Spin ordering, an execution lock that disagreed with the run metadata, and missing exact density/correlation validation.

## Implemented fixes
- Uses `Spin(s=1/2, N=..., inverted_ordering=False)` explicitly.
- Uses physical Rydberg occupation `n=(1-sigma)/2` consistently in models, parity phase, sampler diagnostics, density and correlation calculations.
- Adds `scripts/check_netket_conventions.py` to verify finite-basis numbering, binary occupation mapping, and dense-vs-NetKet operator equality.
- Captures `pip freeze` into `results_stage2_6/environment/pip_freeze.txt` during an executed run.
- Provides recorded macOS arm64 package pins matching the Stage 2.5 execution metadata.
- Writes `density_mae_to_ed` and `connected_correlation_mae_to_ed` for every completed Stage 2.6 row.
- Uses one frozen training budget across model families. Parameter counts remain reported and are not claimed to be equal.

## Commands
```bash
PYTHONPATH=src python -m pytest -q -ra
PYTHONPATH=src python scripts/preflight_nqs.py
PYTHONPATH=src python scripts/check_netket_conventions.py --config configs/netket_stage2_6_conventions.yaml
PYTHONPATH=src python scripts/run_netket_vmc_stage2_6.py --config configs/netket_stage2_6_observable_grid.yaml --overwrite
PYTHONPATH=src python scripts/summarize_stage2_6.py
```

## Stop rules
Do not run Graph-KAN or scale past N=8. Do not aggregate invalid rows. If less than 80% of rows are valid, or if KAN-Jastrow is dominated by both MLP-pair and smooth Jastrow on energy, correlation and runtime, classify the outcome as diagnose/pivot rather than method promotion.
