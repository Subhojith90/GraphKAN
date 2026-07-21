# Stage 4E.1 — Periodic-Boundary Infrastructure Patch

This patch adds an explicit 1D periodic chain geometry and a shared, geometry-aware distance path.
It avoids inferring periodicity from Cartesian positions. The minimum-image distance matrix is used consistently by dense/sparse Rydberg operators, the NetKet operator, Flax pair bases, geometry identity, and the Stage 4E runner.

## Scientific guardrail

`periodic_chain` sets `endpoint_features_enabled: false`. Endpoint ansatz declarations are reduced to their radial counterparts in the model factory, so no endpoint coefficient remains trainable under periodic boundaries.

## Apply

From the existing project root:

```bash
python /path/to/stage4e1-periodic-boundary-patch/apply_stage4e1_patch.py --project-root .
```

The patcher creates timestamped backups under `.stage4e1_backup/`.

## Validate before VMC

```bash
JAX_ENABLE_X64=1 PYTHONPATH=src python -m pytest -q \
  tests/test_periodic_chain_geometry.py \
  tests/test_periodic_flax_features_optional.py \
  tests/test_periodic_netket_operator_optional.py

PYTHONPATH=src python -m pytest -q -ra
```

## Registered smoke configs

- `configs/netket_stage4e1_n10_open_boundary_smoke.yaml`
- `configs/netket_stage4e1_n10_periodic_boundary_smoke.yaml`

They are one-seed construction/physics diagnostics only. Do not interpret them as the periodic mechanism result. Run only after the validation suite passes.
