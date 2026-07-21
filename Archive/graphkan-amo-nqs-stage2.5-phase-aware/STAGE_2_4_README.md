# Stage 2.4 — ED-Calibrated VMC Stabilisation

This package preserves Stage 2.3 as a completed execution audit. Stage 2.4 is not a performance benchmark.
It runs only N=6 and N=8 chains where exact diagonalisation is cheap, verifies the NetKet operator against the
exact dense operator for every Hamiltonian instance, and labels each run as `VALID` or `FLAGGED`.

A run is flagged when it has non-finite estimation, near-zero estimator uncertainty, near-zero energy variance,
low sample diversity, unavailable sample diagnostics, or a large absolute error from the ED ground state.

## Run

```bash
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python scripts/preflight_nqs.py
PYTHONPATH=src python scripts/run_netket_vmc_stabilized.py --config configs/netket_stage2_4_ed_calibrated.yaml --overwrite
PYTHONPATH=src python scripts/summarize_stage2_4.py
```

## Green-light rule

Do not proceed to a larger benchmark unless every model has at least two `VALID` seeds for every N=6/N=8 parameter
point, the operator checks pass at <=1e-10, and valid-only median ED errors are reported separately from flagged rows.
