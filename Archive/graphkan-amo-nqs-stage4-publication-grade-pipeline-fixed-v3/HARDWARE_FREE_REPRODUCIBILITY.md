# Hardware-Free Reproducibility

The core artifact is hardware-free. All generated results in this package use CPU-executable deterministic geometry generation, SciPy/NumPy Hamiltonian construction, exact diagonalisation where feasible, exact full-basis variational evaluation, and script-generated figures/tables.

No AMO hardware is required.

IBM Quantum is not used for the main claims. The IBM optional profile is recorded as not run and remains supplementary-only because superconducting gate hardware does not validate Rydberg-array NQS performance.

## Minimal reproduction

```bash
cd graphkan-amo-nqs-real
python -m venv .venv
source .venv/bin/activate
pip install -e .
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python scripts/run_quick_ci.py --config configs/quick_ci.yaml
PYTHONPATH=src python scripts/make_figures.py --config configs/plotting.yaml
PYTHONPATH=src python scripts/make_tables.py
PYTHONPATH=src python scripts/make_analysis_report.py
```

## Full CPU pilot reproduction

```bash
PYTHONPATH=src python scripts/compute_ed_references.py --config configs/ed_reference.yaml
PYTHONPATH=src python scripts/run_exact_fullsum_pilot.py --config configs/exact_fullsum_pilot.yaml
PYTHONPATH=src python scripts/run_rydberg_pilot.py --config configs/rydberg_chain_pilot.yaml
PYTHONPATH=src python scripts/aggregate_results.py --input results --output results/aggregated
PYTHONPATH=src python scripts/make_figures.py --config configs/plotting.yaml
PYTHONPATH=src python scripts/make_tables.py
PYTHONPATH=src python scripts/make_analysis_report.py
PYTHONPATH=src python scripts/create_release_manifest.py
```
