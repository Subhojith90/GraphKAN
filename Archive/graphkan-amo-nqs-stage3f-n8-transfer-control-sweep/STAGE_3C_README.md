# Stage 3C — Endpoint-Feature Ablation and Replication Gate

## Status
**Prepared but blocked.** Do not execute until Stage 3B.1 confirms that `kan_endpoint_jastrow` uses `basis_family=sine` and reproduces the N=6 endpoint improvement.

## Scope
This package turns the Stage 3B diagnostic into an auditable causal test. It does not authorise N>=10, Graph-KAN, geometry transfer, or manuscript claims.

### Gate 1: N=6 feature ablation
At N=6, Delta=2, C6=4, compares radial-only, pair-centre, and each symmetric endpoint feature separately: sum, difference, product, and the full three-feature endpoint variant. Every KAN uses a sine radial basis and 19 trainable parameters.

### Gate 2: N=8 Batch A
At N=8, Delta=2, C6=4, two seeds: radial sine KAN, endpoint sine KAN, RBF Jastrow, MLP-pair, and endpoint-aware MLP. All models have 21 trainable parameters.

### Gate 3: N=8 five-seed replication
Locked until Gate 2 passes.

## Publication-grade artefacts
Each run writes exact all-state energy, density, connected-correlation and site-resolved observables; the resolved model kind and basis family are recorded in every raw row.

## Commands
```bash
PYTHONPATH=src python -m pytest -q -ra
# only after Stage 3B.1 pass
PYTHONPATH=src python scripts/run_stage3c.py --config configs/netket_stage3c_endpoint_ablation.yaml --overwrite
PYTHONPATH=src python scripts/summarize_stage3c.py --results results_stage3c/endpoint_ablation_n6
# only after ablation decision
PYTHONPATH=src python scripts/run_stage3c.py --config configs/netket_stage3c_n8_batch_a.yaml --overwrite
PYTHONPATH=src python scripts/summarize_stage3c.py --results results_stage3c/n8_batch_a
```
