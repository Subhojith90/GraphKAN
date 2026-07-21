# Stage 4 — Frozen N=8 Five-Seed Replication

This package is self-contained and executes through `scripts/run_stage4.py` and `scripts/summarize_stage4.py`.

## Environment

```bash
conda env create -f environment-nqs-py312-macos-arm64-stage4.yml
conda activate graphkan-stage4-py312
bash scripts/run_stage4_clean_replay.sh
```

## Experiment

```bash
PYTHONPATH=src python scripts/run_stage4.py \
  --config configs/netket_stage4_n8_five_seed_replication.yaml \
  --overwrite
PYTHONPATH=src python scripts/summarize_stage4.py \
  --results results_stage4/n8_five_seed_replication
```

## Frozen models

- `rbf_radial_21p`
- `rbf_endpoint_sum_21p`
- `rbf_endpoint_full_21p`
- `sine_radial_21p`
- `sine_endpoint_full_21p`

The output format intentionally stays compatible with the validated Stage 3F raw schemas (`stage3f_metrics.csv`, histories, site observables and correlation observables). This is a compatibility decision, not a relabelling of the underlying scientific stage.

## v3 correction
The Stage 4 summary reports **two separate N=8 RBF gates**: endpoint-sum versus RBF-radial, and endpoint-full versus RBF-radial. This avoids treating the removed legacy label `rbf_endpoint_21p` as the designated candidate.
