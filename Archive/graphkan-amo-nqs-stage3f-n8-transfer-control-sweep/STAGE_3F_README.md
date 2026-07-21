# Stage 3F — N=8 Hard-Point Transfer and Control-Sweep Gate

## Status
**Blocked execution package.** Run the clean replay and N=6 endpoint-component ablation before N=8 Batch A. This is a transfer checkpoint, not N>=10 scale-up approval.

## Scientific question
At the frozen hard regime \(N=8,\Omega=1,\Delta=2,C_6=4\), does the Stage 3E RBF endpoint-conditioned pair effect remain valid without an unacceptable connected-correlation penalty?

## Required sequence
1. Create a clean environment from `environment-nqs-py312-macos-arm64-stage3f.yml`.
2. Run `scripts/run_stage3f_clean_replay.sh` and archive `artifacts/clean_replay/stage3f_clean_replay.log`.
3. Run `configs/netket_stage3f_component_ablation_n6.yaml`; it exports the raw sum/difference/product/full endpoint results demanded by the audit.
4. Run `configs/netket_stage3f_n8_batch_a.yaml` (two seeds).
5. Run `configs/netket_stage3f_mlp_control_sweep_n8.yaml` only as a validation-only control sweep; do not select a winner from test-style metrics.
6. Continue to five seeds only if the joint energy-density-correlation gate passes.

## Primary N=8 Batch A models
- RBF radial pair, 19 parameters;
- RBF endpoint-conditioned pair, 19 parameters;
- sine radial pair, 19 parameters;
- sine endpoint-conditioned pair, 19 parameters;
- RBF smooth/Jastrow, 19 parameters;
- capacity-swept MLP pair and endpoint-aware MLP controls (reported separately).

## Joint pass condition
RBF endpoint must be valid in both seeds, improve median exact energy and density over RBF radial, and must not worsen exact connected-correlation MAE by more than `0.01` absolute or `25%` relative. Results are diagnostic only.
