# Stage 3A: Boundary-Aware KAN Density Gate

## Why this stage exists
Stage 2.8.1 showed a repeatable hard-regime result at `N=6, Delta=2, C6=4`: radial KAN-Jastrow had substantially lower energy error than matched RBF and MLP controls but systematically failed density MAE. Exact all-state and sampled metrics were close, excluding estimator noise as the primary cause.

The existing radial KAN already includes one learnable field per site. Therefore Stage 3A **does not** claim to add site fields. It adds a strictly controlled, reflection-symmetric pair-centre correction to the radial KAN pair edge:

`edge(i,j) = f_radial(r_ij) + g_boundary(|2 c_ij - 1|)`.

This permits pairs at equal separation but different open-chain boundary locations to have different weights. It is **not Graph-KAN**.

## Gate
Six runs only: N=6, Delta=2, C6=4; radial KAN, boundary-aware KAN, and RBF; two fixed seeds each; 700 iterations; exact all-state observables.

## Run
```bash
python -m pip install -e '.[nqs,dev]'
PYTHONPATH=src python -m pytest -q -ra
PYTHONPATH=src python scripts/run_stage3a_boundary_kan.py --config configs/netket_stage3a_boundary_kan_gate.yaml --overwrite
PYTHONPATH=src python scripts/summarize_stage3a.py --results results_stage3a
```

## Pass criterion
The boundary-aware KAN has lower median exact density MAE than radial KAN without worsening median exact energy error by more than 20%. This gate is diagnostic only; it does not authorise Graph-KAN or N>=10 scale-up.
