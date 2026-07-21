# Codegen and Run Report

Generated: 20 June 2026.

## Source packages inspected

- Proposal package: `GraphKAN_KANJastrow_Proposal_v2_AntiScoop_Overleaf(2).zip`, unpacked under `/mnt/data/graphkan_input_audit/proposal/GraphKAN_KANJastrow_Proposal_v2_AntiScoop_Overleaf`.
- Prototype codebase: `GraphKAN_AMO_NQS_Codebase(3).zip`, unpacked under `/mnt/data/graphkan_input_audit/codebase/graphkan_amo_nqs`.
- User instruction file: `Pasted text(107).txt`.

## Baseline audit results

The uploaded prototype repository passed its original test suite:

```text
PYTHONPATH=src python -m pytest -q tests/
15 passed
```

The uploaded prototype smoke script completed:

```text
PYTHONPATH=src python scripts/run_smoke.py --config configs/smoke.yaml
completed
```

The audit concluded that the prototype was useful Level-1 smoke machinery, not final production evidence.

## Production repository generated

Repository: `/mnt/data/graphkan-amo-nqs-real`.

Generated Python source modules: 93.
Generated test files: 27.
Generated figures: 10.
Generated table/report source files: 12.
Generated result files: 19.

## Tests run

```text
cd /mnt/data/graphkan-amo-nqs-real
PYTHONPATH=src python -m pytest -q
...........................                                              [100%]
27 passed
```

## Experiments and scripts run

Executed:

```text
PYTHONPATH=src python scripts/run_quick_ci.py --config configs/quick_ci.yaml
PYTHONPATH=src python scripts/compute_ed_references.py --config configs/ed_reference.yaml
PYTHONPATH=src python scripts/run_exact_fullsum_pilot.py --config configs/exact_fullsum_pilot.yaml
PYTHONPATH=src python scripts/run_rydberg_pilot.py --config configs/rydberg_chain_pilot.yaml
PYTHONPATH=src python scripts/aggregate_results.py --input results --output results/aggregated
PYTHONPATH=src python scripts/make_figures.py --config configs/plotting.yaml
PYTHONPATH=src python scripts/make_tables.py
PYTHONPATH=src python scripts/make_analysis_report.py
PYTHONPATH=src python scripts/create_release_manifest.py
PYTHONPATH=src python scripts/validate_claims.py
PYTHONPATH=src python scripts/check_top_journal_readiness.py
```

Recorded as not run because they require the next production stage, optional dependencies, or more compute:

```text
PYTHONPATH=src python scripts/run_pra_core.py --config configs/rydberg_pra_core.yaml
PYTHONPATH=src python scripts/run_geometry_transfer.py --config configs/rydberg_geometry_transfer.yaml
PYTHONPATH=src python scripts/run_graphkan.py --config configs/graphkan_full.yaml
PYTHONPATH=src python scripts/run_ablation_core.py --config configs/ablation_core.yaml
PYTHONPATH=src python scripts/run_robustness_core.py --config configs/robustness_core.yaml
PYTHONPATH=src python scripts/run_bose_optional.py --config configs/bose_hubbard_optional.yaml
PYTHONPATH=src python scripts/run_ibm_optional_sanity.py --config configs/ibm_optional_sanity.yaml
```

Unexecuted profiles are retained in `results/raw/failed_runs.csv`; they are not silently discarded.

## Optional dependency status

- jax: available (0.9.0.1)
- flax: not available
- optax: not available
- netket: not available
- quspin: not available
- tenpy: not available
- qiskit: not available
- qiskit_ibm_runtime: not available
- torch: available (2.10.0+cpu)

NetKet, Flax, Optax, QuSpin, TeNPy, and Qiskit were not available in the execution environment. The production VMC and DMRG profiles are therefore implemented as import-safe optional modules but not executed. No NetKet/DMRG/IBM result is claimed.

## Result paths

- ED references: `results/references/ed_references.csv`
- ED validation: `results/references/ed_validation.csv`
- Raw metrics: `results/raw/run_metrics.csv`
- Training histories: `results/raw/training_history.csv`
- Failed/not-run profiles: `results/raw/failed_runs.csv`
- Model comparison: `results/aggregated/model_comparison.csv`
- Paired comparisons: `results/aggregated/paired_comparisons.csv`
- Readiness gates: `results/aggregated/readiness_gates.csv`
- Gauge-fixed functions: `results/interpretability/gauge_fixed_functions.csv`
- Function stability: `results/interpretability/function_stability.csv`
- Scale collapse: `results/interpretability/scale_collapse.csv`
- Correlation matching: `results/interpretability/correlation_matching.csv`

## Figures generated

- `figures/fig_claim_ladder_decision.pdf`
- `figures/fig_compute_pareto.pdf`
- `figures/fig_ed_validation.pdf`
- `figures/fig_failure_map.pdf`
- `figures/fig_gauge_fixed_radial_functions.pdf`
- `figures/fig_geometry_transfer.pdf`
- `figures/fig_kan_vs_mlp_pair_ablation.pdf`
- `figures/fig_pilot_energy_correlation.pdf`
- `figures/fig_reproducibility_workflow.pdf`
- `figures/fig_scale_collapse.pdf`

## Tables generated

- `tables/.gitkeep`
- `tables/baseline_fairness_matrix.csv`
- `tables/compute_budget_table.csv`
- `tables/parameter_count_table.csv`
- `tables/table_ablation_summary.tex`
- `tables/table_ed_validation.tex`
- `tables/table_failure_modes.tex`
- `tables/table_interpretability_metrics.tex`
- `tables/table_model_comparison.tex`
- `tables/table_paired_statistics.tex`
- `tables/table_readiness_gates.tex`
- `tables/table_safe_and_unsafe_claims.tex`

## Report and manifest

- LaTeX source: `paper/production_analysis_report.tex`
- PDF report: `paper/production_analysis_report.pdf`
- Release manifest: `release/manifest.json`

## Scientific status

The CPU pilot supports Hamiltonian correctness and exact-fullsum pilot behaviour. It does not support method-superiority, Graph-KAN, NetKet-VMC, DMRG, hardware-validation, or top-journal-readiness claims.

## Hostile self-review answers

1. Results are generated by scripts from CSV/NPZ files; no manual result editing was used.
2. Quick-CI smoke rows are separated by profile and paper plotting rejects smoke-only data unless explicitly allowed.
3. Rydberg dense/sparse and analytic-limit tests pass.
4. Exact references are generated for the configured small instances; large-system truth is not claimed.
5. Baseline fairness tables are generated, but final fair VMC baselines remain future work.
6. MLP-pair receives the same blockade-normalised feature family as KAN-Jastrow in the CPU pilot.
7. KAN learned functions are gauge-fixed before export.
8. Pilot gauge-fixed curves are seed-stable, but this is not final interpretability proof.
9. Correlation errors are reported, not only energies.
10. Failed and unexecuted runs are retained.
11. In the Rydberg CPU pilot, KAN-Jastrow is not convincingly better than MLP-pair or smooth Jastrow.
12. Graph-KAN was not run and no Graph-KAN value is claimed.
13. Optional dependencies are handled gracefully and logged.
14. IBM results are absent and explicitly supplementary-only.
15. Figures are regenerated by `scripts/make_figures.py`.
16. The project is worth continuing as a benchmark/interpretable-NQS study, not as a superiority claim.
17. Strongest rejection argument: KAN-Jastrow is dominated or matched by simpler feature baselines in the CPU pilot.
18. Evidence that would overturn rejection: NetKet/Flax VMC with stronger baselines, more seeds, OOD transfer, and stable correlation-matched functions.
