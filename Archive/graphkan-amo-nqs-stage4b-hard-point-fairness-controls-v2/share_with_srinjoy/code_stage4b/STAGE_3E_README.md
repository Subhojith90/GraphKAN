# Stage 3E — Matched-Fairness and Five-Seed Replication Gate

## Purpose
Stage 3E is the final N=6 hard-regime fairness gate before any N=8 work. It studies the known open Rydberg-chain instance `N=6, Omega=1, Delta=2, C6=4` with all models explicitly matched at **19 trainable parameters**.

## Models
- sine radial pair model;
- sine endpoint-conditioned pair model;
- RBF radial pair model;
- RBF endpoint-conditioned pair model;
- radial MLP-pair control;
- endpoint-aware MLP-pair control.

The duplicated RBF radial label from Stage 3D has been removed. `basis_family` is recorded in all raw outputs. The MLP controls are exactly 19 parameters: `mlp_pair(n_basis=10, hidden=1)` and `mlp_endpoint_pair(n_basis=7, hidden=1)`.

## Run order
Run the two-seed checkpoint first:

```bash
PYTHONPATH=src python -m pytest -q -ra
PYTHONPATH=src python scripts/run_stage3e.py --config configs/netket_stage3e_fairness_checkpoint.yaml --overwrite
PYTHONPATH=src python scripts/summarize_stage3e.py --results results_stage3e/fairness_checkpoint_n6
```

Only after review run the five-seed replication:

```bash
PYTHONPATH=src python scripts/run_stage3e.py --config configs/netket_stage3e_fairness_replication.yaml --overwrite
PYTHONPATH=src python scripts/summarize_stage3e.py --results results_stage3e/fairness_replication_n6
```

## Claim discipline
This stage can test a basis-family × endpoint-feature interaction against matched MLP controls at one hard point. It cannot establish broad generalisation, Graph-KAN value, geometry transfer, or publication readiness on its own.
