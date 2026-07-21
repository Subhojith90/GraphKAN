# Stage 4D — Mechanism, Artifact, and N=10 Checkpoint Gate

## Why this is lean
Stage 4C used 630 trajectories. Stage 4D is deliberately staged, resumable, and budget-guarded:

- `mechanism_screen`: 84 trajectories (N={6,8}, Δ={1.5,2.0}, C6=4, 3 seeds, 7 models). It is a mechanism screen with 2048 samples / 350 iterations, not a broad scale campaign.
- `low_detuning_autopsy`: 42 trajectories (N=8, Δ=1.5, C6={4,5}, 3 seeds, 7 models), launched only after the core screen is healthy.
- `n10_pilot`: 48 trajectories (N=10 × 4 cells × 2 seeds × 6 models). It is a finite-size checkpoint pilot, not a final 5-seed campaign.

This reduces expected work by roughly an order of magnitude versus repeating a 630-row Stage 4C-style grid. Actual wall-clock time remains hardware-dependent; no script can honestly guarantee a fixed runtime.

## Mandatory order
1. Run Stage 4C.1 artifact patch and clean extraction test.
2. Run the Stage 4D mechanism screen.
3. Inspect mechanism outputs; do not launch N=10 if mechanism/profile health is broken.
4. Run N=10 pilot only after the screen is healthy.

## Commands
```bash
# Stage 4C.1 artifact repair
PYTHONPATH=src python scripts/package_stage4c1_release.py --results-root results_stage4c --output release_stage4c1
(cd release_stage4c1 && python -m pytest -q release_tests && shasum -a 256 -c MANIFEST.sha256)

# Validate assets and count planned work
PYTHONPATH=src python -m pytest -q tests/test_stage4d_assets.py
PYTHONPATH=src python scripts/stage4d_preflight_budget.py --config configs/netket_stage4d_mechanism_screen.yaml --max-runs 100

# Mechanism screen (resumable)
bash scripts/run_stage4d_batch.sh configs/netket_stage4d_mechanism_screen.yaml --resume
PYTHONPATH=src python scripts/summarize_stage4d.py --results results_stage4d/mechanism_screen

# Low-detuning autopsy only after reviewing the core screen
PYTHONPATH=src python scripts/stage4d_preflight_budget.py --config configs/netket_stage4d_low_detuning_autopsy.yaml --max-runs 60
bash scripts/run_stage4d_batch.sh configs/netket_stage4d_low_detuning_autopsy.yaml --resume
PYTHONPATH=src python scripts/summarize_stage4d.py --results results_stage4d/low_detuning_autopsy

# N=10 pilot only after reviewing the mechanism screen and autopsy
PYTHONPATH=src python scripts/stage4d_preflight_budget.py --config configs/netket_stage4d_n10_pilot.yaml --max-runs 60
bash scripts/run_stage4d_batch.sh configs/netket_stage4d_n10_pilot.yaml --resume
PYTHONPATH=src python scripts/summarize_stage4d.py --results results_stage4d/n10_pilot
```

## Claims allowed
- Stage 4D is an internal mechanism and finite-size checkpoint.
- Endpoint sum/product/hybrid trade-offs can be compared on the registered screen.

## Claims forbidden
- Generic KAN superiority, Graph-KAN value, universal finite-size scaling, publication readiness.
