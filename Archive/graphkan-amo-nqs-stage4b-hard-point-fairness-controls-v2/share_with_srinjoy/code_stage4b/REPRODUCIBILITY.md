# Reproducibility

1. Create the declared environment:

```bash
conda env create -f environment-nqs-py312-macos-arm64-stage4b.yml
conda activate graphkan-stage4b-py312
```

2. Run the clean replay:

```bash
bash scripts/run_stage4b_clean_replay.sh
```

3. Execute the three explicit Stage 4B batches, preferably one at a time:

```bash
PYTHONPATH=src python scripts/run_stage4b_controls.py --config configs/netket_stage4b_n8_component_replay.yaml --overwrite
PYTHONPATH=src python scripts/summarize_stage4b.py --results results_stage4b/n8_component_replay --candidate rbf_endpoint_sum_21p

PYTHONPATH=src python scripts/run_stage4b_controls.py --config configs/netket_stage4b_n8_controls_core.yaml --overwrite
PYTHONPATH=src python scripts/summarize_stage4b.py --results results_stage4b/n8_controls_core --candidate rbf_endpoint_sum_21p

PYTHONPATH=src python scripts/run_stage4b_controls.py --config configs/netket_stage4b_n8_capacity_compute.yaml --overwrite
PYTHONPATH=src python scripts/summarize_stage4b.py --results results_stage4b/n8_capacity_compute --candidate rbf_endpoint_sum_b12_21p
```

4. Build the batch index and, only after all three batches are complete, make the release archive:

```bash
PYTHONPATH=src python scripts/summarize_stage4b_all.py --root results_stage4b
PYTHONPATH=src python scripts/package_stage4b_release.py --results-root results_stage4b --output release_stage4b
```

The clean replay log covers environment installation, tests, preflight, and convention checks. Each VMC command must be captured in a terminal log and included with the release.
