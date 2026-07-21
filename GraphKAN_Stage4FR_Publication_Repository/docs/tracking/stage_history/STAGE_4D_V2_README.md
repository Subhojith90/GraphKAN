# Stage 4D v2

```bash
PYTHONPATH=src python scripts/package_stage4c1_release.py --stage4c-results-root "../graphkan-amo-nqs-stage4c-adjacent-hard-patch-fairness-repair/results_stage4c" --output release_stage4c1
(cd release_stage4c1 && python -m pytest -q release_tests && shasum -a 256 -c MANIFEST.sha256)
PYTHONPATH=src python -m pytest -q tests/test_stage4d_v2_smoke_assets.py tests/test_stage4c_repaired_pairing.py
bash scripts/run_stage4d_batch.sh configs/netket_stage4d_fullbudget_smoke.yaml --overwrite
PYTHONPATH=src python scripts/stage4d_validate_smoke.py --results results_stage4d/fullbudget_smoke
bash scripts/run_stage4d_batch.sh configs/netket_stage4d_mechanism_mini.yaml --resume
```

Do not use Stage 4D v1 scientifically. N=10 remains blocked.
