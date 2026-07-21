# GraphKAN AMO NQS — Stage 2 Handoff

## Delivered

- Python 3.11 environment specification and NQS dependency compatibility file.
- Machine-readable runtime preflight (`scripts/preflight_nqs.py`).
- A NetKet `LocalOperator` Rydberg Hamiltonian builder matching the repository binary occupation convention.
- Trainable Flax product, smooth-Jastrow, MLP-pair, and KAN-Jastrow real-amplitude ansätze.
- Strict NetKet VMC runner (`scripts/run_netket_vmc.py`): it never substitutes the CPU pilot backend.
- A smoke config and a registered five-seed benchmark config.
- Stage-2 execution guide and optional-dependency tests.

## Local validation

`PYTHONPATH=src python -m pytest -q` -> 31 passed.

The local environment does not have the full NetKet/Flax stack. This is recorded by the generated preflight JSON; no VMC outputs were fabricated.

## First commands for the execution machine

```bash
conda env create -f environment-nqs-py311.yml
conda activate graphkan-amo-nqs-py311
python -m pip install --upgrade pip
python -m pip install -e '.[nqs,dev]'
PYTHONPATH=src python scripts/preflight_nqs.py
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python scripts/run_netket_vmc.py --config configs/netket_smoke.yaml
```

Only after smoke passes should the registered profile be launched.
