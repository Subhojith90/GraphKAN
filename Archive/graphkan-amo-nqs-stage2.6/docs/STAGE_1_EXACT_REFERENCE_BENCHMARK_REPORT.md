# Stage 1 — Registered Exact-Reference Benchmark Report

## Scope

This report documents a dependency-light, exact-fullsum screening benchmark added after the
initial CPU pilot. It is deliberately **not** a NetKet/VMC result. Each candidate ansatz is
optimised on the full computational basis for small Rydberg chains and compared to an exact
diagonalisation reference.

## Registered protocol

- Models: product, smooth Jastrow, MLP-pair, KAN-Jastrow.
- Geometries: open chains with \(N=8\) and \(N=10\), unit spacing.
- Hamiltonian grid: \(\Omega=1\), \(\Delta\in\{-1,1\}\), \(C_6\in\{1,4\}\).
- Seeds: 0, 1, 2, 3, 4.
- Training: identical SciPy L-BFGS-B exact full-basis objective; maximum 65 iterations; ridge \(10^{-8}\).
- Total: 160 matched optimisation runs and 1,720 raw metric rows.

The immutable execution declaration is stored in
`results_exact_reference_benchmark_v1/raw/benchmark_declaration.json`.

## Results

Median errors pooled over 40 model-instance-seed combinations per model:

| Model | Median energy-density error | Median connected-correlation MAE |
|---|---:|---:|
| MLP-pair | 0.004805 | 0.004415 |
| smooth Jastrow | 0.005409 | 0.005176 |
| KAN-Jastrow | 0.009782 | 0.010205 |
| product | 0.027176 | 0.009665 |

Paired comparisons use baseline minus KAN-Jastrow, so positive values favour KAN-Jastrow.

| Comparison | Metric | Median paired delta | Holm-adjusted p-value | Interpretation |
|---|---|---:|---:|---|
| MLP-pair − KAN-Jastrow | energy-density error | -0.000247 | 0.0734 | Unfavourable to KAN; not significant after Holm correction. |
| smooth Jastrow − KAN-Jastrow | energy-density error | -0.000240 | 1.0000 | Inconclusive after correction; median favours smooth Jastrow. |
| MLP-pair − KAN-Jastrow | correlation MAE | -0.000216 | 0.0734 | Unfavourable to KAN; not significant after Holm correction. |
| smooth Jastrow − KAN-Jastrow | correlation MAE | -0.000361 | 0.0734 | Unfavourable to KAN; not significant after Holm correction. |

## Decision

**Do not make a KAN-superiority claim.** The registered exact-reference benchmark confirms
that KAN-Jastrow is better than the product-state sanity baseline but is not demonstrated to
outperform matched MLP-pair or smooth-Jastrow baselines in this CPU exact-feature setting.

The project should proceed only as an **interpretable-NQS / benchmark** study. The next gate
is a trainable NetKet/Flax production study with matched parameter counts and Monte Carlo
budgets. Graph-KAN remains gated until KAN-Jastrow has a non-dominated result or a stronger,
quantified interpretability result.

## Reproduction

```bash
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python scripts/run_exact_benchmark.py --config configs/exact_reference_benchmark.yaml
PYTHONPATH=src python scripts/aggregate_results.py \
  --input results_exact_reference_benchmark_v1 \
  --output results_exact_reference_benchmark_v1/aggregated
```
