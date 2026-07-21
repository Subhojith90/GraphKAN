
# Stage 4E.1-R command transcript

conda activate graphkan-stage4b-py312

JAX_ENABLE_X64=1 PYTHONPATH=src python -m pytest -q tests/test_periodic_chain_geometry.py tests/test_periodic_flax_features_optional.py tests/test_periodic_netket_operator_optional.py -ra
JAX_ENABLE_X64=1 PYTHONPATH=src python -m pytest -q -ra

JAX_ENABLE_X64=1 PYTHONPATH=src python scripts/run_stage4e.py --config configs/netket_stage4e1_n10_open_boundary_smoke.yaml --overwrite
JAX_ENABLE_X64=1 PYTHONPATH=src python scripts/run_stage4e.py --config configs/netket_stage4e1_n10_periodic_boundary_smoke.yaml --overwrite

JAX_ENABLE_X64=1 PYTHONPATH=src python scripts/run_stage4e.py --config configs/netket_stage4e1_n10_open_boundary_paired5.yaml --resume
JAX_ENABLE_X64=1 PYTHONPATH=src python scripts/run_stage4e.py --config configs/netket_stage4e1_n10_periodic_boundary_paired5.yaml --resume

The original terminal session was not captured with tee. Reconstructed VMC summaries are included under logs/ and raw metrics remain the source of truth.
