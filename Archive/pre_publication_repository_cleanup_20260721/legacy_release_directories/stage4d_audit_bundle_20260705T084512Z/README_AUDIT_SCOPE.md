# Stage 4D Audit Bundle

Created: 20260705T084512Z
Project root: graphkan-amo-nqs-stage4d-v2-fullbudget-smoke-mechanism-gate

## Scope

This bundle contains:
- Stage 4C.1 evidence release, when present locally.
- Stage 4D source, frozen configurations, tests, logs, raw CSV outputs,
  checkpoints, summaries, coefficient exports, and SHA manifests.
- Quarantined failed/under-budget runs retained as failure provenance.

## Precision and engineering status

- Production VMC runs used JAX x64.
- Stage 4D model-build regression test passed for seven declared N=10 model kinds.
- RBM dispatch was repaired: RBM now reports `basis_family=none` and completes.
- RBM remains density-flagged at the N=10 smoke cell; it is retained as a
  negative-control baseline, not a headline competitor.

## N=10 five-seed checkpoint

Cell:
- N=10, Omega=1.0, Delta=1.5, C6=4.0
- 5 seeds, 4096 samples, 64 chains, 700 iterations
- 25/25 exact-valid and 25/25 sample-valid trajectories

Median exact errors:
- MLP endpoint, 130 actual parameters:
  energy 0.037591; density 0.019713; correlation 0.004835
- MLP radial, 130 actual parameters:
  energy 0.040487; density 0.019784; correlation 0.006548
- Endpoint sum RBF, 23 actual parameters:
  energy 0.284550; density 0.055611; correlation 0.030255
- Endpoint product RBF, 23 actual parameters:
  energy 0.366066; density 0.047442; correlation 0.032038
- Compact radial RBF, 23 actual parameters:
  energy 0.469766; density 0.028853; correlation 0.032122

## Conservative conclusion

The compact endpoint-feature models remain useful as local N=6/N=8
boundary-aware failure-repair diagnostics. At the replicated N=10 checkpoint,
the MLP-128 controls outperform the compact endpoint models on energy,
density, and connected-correlation error. This bundle does not claim generic
KAN, GraphKAN, or endpoint-feature superiority.
