# Stage 2.3 — Flax float64 hotfix

## Defect
On NetKet 3.22.3/JAX 0.7.1, the MLP-pair ansatz parameters were initialised in `float32`, while the stochastic-reconfiguration/VMC update path carried `float64` values. JAX `while_loop` requires invariant carry dtypes, so the MLP-pair smoke run failed after the product and Jastrow models completed.

## Fix
All custom Flax NQS parameters and Dense layers are now explicitly `float64`; pair-feature arrays are emitted as `float64`. This creates one consistent dtype policy for the custom ansatz family under NetKet VMC.

## Re-run
```bash
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python scripts/run_netket_vmc.py --config configs/netket_smoke.yaml --overwrite
```

Use `--overwrite` because the earlier run has retained partial result rows by design.
