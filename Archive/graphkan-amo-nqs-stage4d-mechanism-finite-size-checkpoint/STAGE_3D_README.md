# Stage 3D — Basis-Family × Endpoint-Feature Factorial Gate

## Status
Implementation prepared. **Blocked until Stage 3B.1 taxonomy-locked replay is archived and reviewed.**

## Scientific question
The original Stage 3B endpoint result used an ambiguous basis dispatch. Stage 3C showed that explicit sine endpoint features did not reproduce the density repair. Stage 3D isolates whether the positive signal is caused by:

1. sine versus RBF radial basis family;
2. endpoint features;
3. their interaction; or
4. a control-model/optimisation effect.

## Design
At fixed `N=6, omega=1, delta=2, C6=4`, the checkpoint factorial compares:

- sine radial KAN;
- sine endpoint KAN;
- RBF radial KAN proxy;
- RBF endpoint KAN proxy;
- matched RBF-Jastrow control;
- MLP pair control;
- endpoint-aware MLP control.

`basis_family` is explicit in every config and persisted in raw metrics, site observables, and correlation observables. KAN/RBF pair variants use equal nominal parameter budgets; MLP capacities are recorded rather than falsely described as parameter matched.

## Publication-grade material produced
- explicit taxonomy table and unit-tested basis dispatch;
- exact all-state energy, density, and connected-correlation observables;
- paired factorial contrasts for endpoint and basis effects;
- seed-level raw rows and immediate checkpoints;
- site-resolved density error tables;
- validity/failure accounting, not valid-only ranking;
- preflight, config hash, run manifest, and minimal freeze output.

This is publication-quality *evidence collection*, not publication readiness by itself. No superiority or transfer claim is permitted until the five-seed replication and N=8 checkpoint pass.

## Execution sequence
1. Run the two-seed checkpoint only:

```bash
PYTHONPATH=src python scripts/run_stage3d.py \
  --config configs/netket_stage3d_factorial_checkpoint.yaml --overwrite
PYTHONPATH=src python scripts/summarize_stage3d.py \
  --results results_stage3d/factorial_checkpoint_n6
```

2. Review all raw/aggregated outputs. Unlock five-seed replication only if the taxonomy is explicit and the checkpoint contains no execution or convention failure.

3. Run the five-seed replication, then the N=8 checkpoint, separately. Do not run them in one shell chain.
