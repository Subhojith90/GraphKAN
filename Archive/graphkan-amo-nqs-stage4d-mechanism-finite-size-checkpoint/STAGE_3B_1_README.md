# Stage 3B.1 — Basis-Taxonomy-Locked Replay

Purpose: repair the Stage 3B basis-family ambiguity before any Stage 3C work.

## Scope

Exactly the six stored Stage 3B gate rows are replayed: N=6, omega=1, delta=2, C6=4; 2 seeds; 700 iterations; 4096 samples. The topology, seeds, hyperparameters and health thresholds are unchanged.

## Taxonomy rule

- `kan_jastrow`, `kan_boundary_jastrow`, and `kan_endpoint_jastrow` use `basis_family: sine`.
- `smooth_jastrow` uses `basis_family: rbf`.
- The resolved family is written to metrics, site-observable and correlation-observable CSV files.

## Required commands

```bash
python -m pip install -e '.[nqs,dev]'
PYTHONPATH=src python -m pytest -q -ra
PYTHONPATH=src python scripts/run_stage3b_endpoint_kan.py --config configs/netket_stage3b_endpoint_kan_gate.yaml --overwrite
PYTHONPATH=src python scripts/summarize_stage3b.py --results results_stage3b
```

Do not run Stage 3C unless this replay preserves the endpoint result and records `basis_family=sine` for both KAN variants.
