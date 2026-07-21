# Stage 2.5 — Phase-Aware Known-Good VMC Gate

## Why this stage exists
Stage 2.4 verified that the NetKet and dense Hamiltonians agree at machine precision, but all custom ansätze used a real log-amplitude. For the implemented positive transverse-field convention, the ground-state wavefunction carries an occupation-parity sign. A strictly real log-amplitude produces a non-negative wavefunction and cannot represent that state.

Stage 2.5 adds the fixed exact gauge phase `i*pi*sum_i n_i` for positive omega, while retaining trainable real amplitudes. It also replaces the one-scalar product state with site-wise fields and adds a phase-aware RBM as a known-good capacity control.

## Gate
Only `N=6, omega=1, delta=1, C6=1` is run. Do not run larger sweeps.

Run:
```bash
python -m pip install -e '.[nqs,dev]'
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python scripts/preflight_nqs.py
PYTHONPATH=src python scripts/run_netket_vmc_stabilized.py --config configs/netket_stage2_5_phase_aware_gate.yaml --overwrite
PYTHONPATH=src python scripts/summarize_stage2_5.py
```

## Green signal
The immediate code-level gate is passed only if:
1. NetKet-vs-dense operator check remains at <= 1e-10;
2. RBM has at least two VALID seeds;
3. at least one custom model has at least two VALID seeds;
4. no architecture ranking is reported from invalid trajectories.

Stage 2.5 is a variational-state correctness test, not a paper benchmark.
