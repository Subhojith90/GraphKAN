# Stage 2.7: Hard-Regime Diagnostic and Matched-Fairness Gate

Stage 2.7 is the prerequisite for any Stage 3 discussion. It does **not** authorise N>=10, Graph-KAN, geometry transfer, or manuscript claims.

## Goals
1. Reproduce the C6=4, Delta=2 hard regime using a clean locked environment.
2. Export sampled and exact-all-state density/correlation observables.
3. Separate sample-count/SR sensitivity from ansatz-form limitations.
4. Compare sine-KAN/Jastrow, RBF Jastrow, and MLP-pair at closely matched parameter counts and compute.

## Required commands
```bash
python -m pip install -e '.[nqs,dev]'
PYTHONPATH=src python -m pytest -q -ra
PYTHONPATH=src python scripts/preflight_nqs.py
PYTHONPATH=src python scripts/check_netket_conventions.py --config configs/netket_stage2_6_conventions.yaml
PYTHONPATH=src python scripts/run_stage2_7_hard_regime.py --config configs/netket_stage2_7_hard_regime.yaml --overwrite
PYTHONPATH=src python scripts/run_stage2_7_fairness.py --config configs/netket_stage2_7_matched_fairness.yaml --overwrite
PYTHONPATH=src python scripts/summarize_stage2_7.py
```

All outputs are written under `results_stage2_7/`.
