# Stage 3B — Endpoint-Aware KAN/Jastrow Gate

## Question
Stage 3A showed that a pair-centre correction slightly reduced an edge-localised density error but did not pass the observable gate. Stage 3B tests whether the missing information is **endpoint environment**, not merely pair centre.

For site `i`, define nearest-boundary distance `d_i = min(i, N-1-i)`. For a pair `(i,j)`, the endpoint features are

`d_i+d_j`, `|d_i-d_j|`, and `d_i d_j`.

They are invariant under chain reflection and distinguish edge-edge, edge-bulk, and bulk-bulk pairs even at identical separation.

## Scope
Six runs only: `N=6`, `Omega=1`, `Delta=2`, `C6=4`, 2 seeds, 700 iterations, 4096 samples.

Models are parameter-matched at 19 trainable parameters:
- radial sine KAN/Jastrow with 13 radial basis coefficients;
- endpoint-aware sine KAN/Jastrow with 10 radial + 3 endpoint coefficients;
- RBF Jastrow with 13 radial basis coefficients.

## Pass condition
Endpoint-aware KAN must reduce median exact density MAE relative to the 19-parameter radial KAN **and** must not worsen median exact energy error by more than 20%. The test is diagnostic only; two seeds are not evidence of superiority.

## Run
```bash
PYTHONPATH=src python -m pytest -q -ra
PYTHONPATH=src python scripts/run_stage3b_endpoint_kan.py --config configs/netket_stage3b_endpoint_kan_gate.yaml --overwrite
PYTHONPATH=src python scripts/summarize_stage3b.py --results results_stage3b
```

Do not run larger sizes or Graph-KAN from this package.
