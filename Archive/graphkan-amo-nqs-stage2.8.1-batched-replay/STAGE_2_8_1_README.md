# Stage 2.8.1 — Comparable Mini-Batch Replay

This is a corrective patch to Stage 2.8 Batch A. The original smoke profile reduced the Stage 2.7 optimisation budget from `n_iter: 700` and `n_discard_per_chain: 256` to 200 and 128. It therefore tested execution only, not a scientifically comparable hard-regime run.

This profile retains the small 6-run scope:

- N=6, delta=2, C6=4
- 3 matched models: RBF-Jastrow, MLP-pair, sine-KAN/Jastrow
- 2 seeds

but restores the Stage 2.7 training budget exactly:

- n_iter=700
- n_discard_per_chain=256
- n_samples=4096
- diag_shift=0.10

Run:

```bash
PYTHONPATH=src python scripts/run_stage2_7_hard_regime.py \
  --config configs/netket_stage2_8_1_batch_a_n6_replay700.yaml \
  --overwrite
PYTHONPATH=src python scripts/summarize_stage2_8_batch.py \
  --results results_stage2_8_1/batch_a_n6_replay700
```

Interpretation rule: compare these six runs only against Stage 2.7 configurations having the same model, `n_samples=4096`, `diag_shift=0.10`, `n_iter=700`, and `n_discard_per_chain=256`.
