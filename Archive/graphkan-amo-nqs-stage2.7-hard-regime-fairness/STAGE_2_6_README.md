# Stage 2.6 — Convention-Locked Observable Gate

This stage implements the supervisor-required fixes before any performance scale-up.

## Must-pass sequence

```bash
python -m pip install -e '.[nqs,dev]'
PYTHONPATH=src python -m pytest -q -ra
PYTHONPATH=src python scripts/preflight_nqs.py
PYTHONPATH=src python scripts/check_netket_conventions.py --config configs/netket_stage2_6_conventions.yaml
PYTHONPATH=src python scripts/run_netket_vmc_stage2_6.py --config configs/netket_stage2_6_observable_grid.yaml --overwrite
PYTHONPATH=src python scripts/summarize_stage2_6.py
```

## Convention policy

`Spin(..., inverted_ordering=False)` is explicit. Under this current NetKet convention, `+1=up` and `-1=down`; physical Rydberg occupation is therefore `n=(1-sigma)/2`. The fixed parity phase uses that same physical occupation. The convention checker records the policy and verifies the dense-operator match for each prescribed instance.

## Scope

The grid is `N={6,8}`, `Delta={-1,0,1,2}`, `C6={0.25,1,4}`, five seeds, and five models. Every valid run records ED energy error, density MAE, and connected-correlation MAE. Invalid rows are retained and excluded from comparisons.

No Graph-KAN, geometry-transfer, Bose-Hubbard, or manuscript claim is authorised by this stage.
