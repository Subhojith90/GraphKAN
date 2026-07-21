# Stage 2 — Trainable NetKet/Flax VMC execution

This stage is the first trainable neural-quantum-state stage. It is not a relabelled exact-fullsum run. The runner uses NetKet's `MCState`, local Metropolis sampling, Optax Adam, and stochastic reconfiguration. It writes `backend=netket_flax` on every completed row and records failures rather than substituting the CPU implementation.

## 1. Create an isolated Python 3.11 environment

```bash
conda env create -f environment-nqs-py311.yml
conda activate graphkan-amo-nqs-py311
python -m pip install --upgrade pip
python -m pip install -e '.[nqs,dev]'
```

For GPU JAX, install the JAX wheel recommended by the official JAX installer for your CUDA version *before* installing NetKet. Do not mix system Python packages with this environment.

## 2. Verify the runtime

```bash
PYTHONPATH=src python scripts/preflight_nqs.py
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python scripts/run_netket_vmc.py --config configs/netket_smoke.yaml --dry-run
PYTHONPATH=src python scripts/run_netket_vmc.py --config configs/netket_smoke.yaml
```

The smoke output is a wiring check only. Do not include it in paper tables or figures.

## 3. Run the registered benchmark

```bash
PYTHONPATH=src python scripts/run_netket_vmc.py --config configs/netket_rydberg_registered.yaml
```

Before extending the grid, archive the exact config used and record the Git commit hash. Do not change a model-specific learning rate, sample count, SR shift, seed set, or iteration count within a registered comparison.

## 4. Mandatory post-run checks

- Verify every completed row has `backend=netket_flax`.
- Preserve `FAILED` rows and the preflight JSON.
- Compare the NetKet operator against ED for `N=6` before interpreting optimisation results.
- Compare energy density, connected correlations, density, structure factor, parameter count, and wall time; energy alone is insufficient.
- Use paired seed-instance statistics. Do not pool heterogeneous Hamiltonian points as independent samples.

## Known scope limits

The Stage 2 models are compact real-amplitude ansätze. They are appropriate for the real stoquastic Rydberg Hamiltonian used here, but they are not a general complex-wavefunction benchmark. Graph-KAN remains intentionally excluded until KAN-Jastrow clears the registered baseline gate.
