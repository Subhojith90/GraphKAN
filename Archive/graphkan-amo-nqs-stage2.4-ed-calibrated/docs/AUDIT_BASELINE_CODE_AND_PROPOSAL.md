# Baseline code and proposal audit

## Source packages inspected

- `/mnt/data/GraphKAN_AMO_NQS_Codebase(3).zip`
- `/mnt/data/GraphKAN_KANJastrow_Proposal_v2_AntiScoop_Overleaf(2).zip`
- Prior proposal PDF and prior package copies in `/mnt/data` were also available, but the current production run uses the two newest uploaded packages.

## Exact codebase inventory

```text
.gitignore
.pytest_cache/.gitignore
.pytest_cache/CACHEDIR.TAG
.pytest_cache/README.md
.pytest_cache/v/cache/nodeids
LICENSE
Makefile
README.md
configs/ablation.yaml
configs/full.yaml
configs/laptop.yaml
configs/robustness.yaml
configs/smoke.yaml
docs/SCIENTIFIC_VALIDITY.md
environment.yml
figures/.gitkeep
notebooks/00_quickstart.ipynb
notebooks/01_results_analysis.ipynb
paper_assets/figures/.gitkeep
paper_assets/tables/.gitkeep
pyproject.toml
reproducibility_manifest.json
requirements.txt
results/.gitkeep
results/20260620_164323_smoke_rydberg_kan_smoke/20260620_164324_smoke_rydberg_kan_kan_jastrow_rydberg_chain_N5_733f7934fd28_seed0/config.yaml
results/20260620_164323_smoke_rydberg_kan_smoke/20260620_164324_smoke_rydberg_kan_kan_jastrow_rydberg_chain_N5_733f7934fd28_seed0/environment_snapshot.txt
results/20260620_164323_smoke_rydberg_kan_smoke/20260620_164324_smoke_rydberg_kan_kan_jastrow_rydberg_chain_N5_733f7934fd28_seed0/experiment.log
results/20260620_164323_smoke_rydberg_kan_smoke/20260620_164324_smoke_rydberg_kan_kan_jastrow_rydberg_chain_N5_733f7934fd28_seed0/history.csv
results/20260620_164323_smoke_rydberg_kan_smoke/20260620_164324_smoke_rydberg_kan_kan_jastrow_rydberg_chain_N5_733f7934fd28_seed0/instance.json
results/20260620_164323_smoke_rydberg_kan_smoke/20260620_164324_smoke_rydberg_kan_kan_jastrow_rydberg_chain_N5_733f7934fd28_seed0/metrics.csv
results/20260620_164323_smoke_rydberg_kan_smoke/20260620_164324_smoke_rydberg_kan_kan_jastrow_rydberg_chain_N5_733f7934fd28_seed0/metrics.json
results/20260620_164323_smoke_rydberg_kan_smoke/20260620_164324_smoke_rydberg_kan_kan_jastrow_rydberg_chain_N5_733f7934fd28_seed0/model.pt
results/20260620_164323_smoke_rydberg_kan_smoke/20260620_164324_smoke_rydberg_kan_kan_jastrow_rydberg_chain_N5_733f7934fd28_seed0/observables.npz
results/20260620_164323_smoke_rydberg_kan_smoke/20260620_164324_smoke_rydberg_kan_kan_jastrow_rydberg_chain_N5_733f7934fd28_seed0/pair_function.csv
results/20260620_164323_smoke_rydberg_kan_smoke/20260620_164324_smoke_rydberg_kan_kan_jastrow_rydberg_chain_N5_733f7934fd28_seed0/reproducibility_manifest.json
results/20260620_164323_smoke_rydberg_kan_smoke/20260620_164324_smoke_rydberg_kan_kan_jastrow_rydberg_chain_N5_733f7934fd28_seed0/run_metadata.json
results/20260620_164323_smoke_rydberg_kan_smoke/20260620_164324_smoke_rydberg_kan_mlp_pair_rydberg_chain_N5_733f7934fd28_seed0/config.yaml
results/20260620_164323_smoke_rydberg_kan_smoke/20260620_164324_smoke_rydberg_kan_mlp_pair_rydberg_chain_N5_733f7934fd28_seed0/environment_snapshot.txt
results/20260620_164323_smoke_rydberg_kan_smoke/20260620_164324_smoke_rydberg_kan_mlp_pair_rydberg_chain_N5_733f7934fd28_seed0/experiment.log
results/20260620_164323_smoke_rydberg_kan_smoke/20260620_164324_smoke_rydberg_kan_mlp_pair_rydberg_chain_N5_733f7934fd28_seed0/history.csv
results/20260620_164323_smoke_rydberg_kan_smoke/20260620_164324_smoke_rydberg_kan_mlp_pair_rydberg_chain_N5_733f7934fd28_seed0/instance.json
results/20260620_164323_smoke_rydberg_kan_smoke/20260620_164324_smoke_rydberg_kan_mlp_pair_rydberg_chain_N5_733f7934fd28_seed0/metrics.csv
results/20260620_164323_smoke_rydberg_kan_smoke/20260620_164324_smoke_rydberg_kan_mlp_pair_rydberg_chain_N5_733f7934fd28_seed0/metrics.json
results/20260620_164323_smoke_rydberg_kan_smoke/20260620_164324_smoke_rydberg_kan_mlp_pair_rydberg_chain_N5_733f7934fd28_seed0/model.pt
results/20260620_164323_smoke_rydberg_kan_smoke/20260620_164324_smoke_rydberg_kan_mlp_pair_rydberg_chain_N5_733f7934fd28_seed0/observables.npz
results/20260620_164323_smoke_rydberg_kan_smoke/20260620_164324_smoke_rydberg_kan_mlp_pair_rydberg_chain_N5_733f7934fd28_seed0/pair_function.csv
results/20260620_164323_smoke_rydberg_kan_smoke/20260620_164324_smoke_rydberg_kan_mlp_pair_rydberg_chain_N5_733f7934fd28_seed0/reproducibility_manifest.json
results/20260620_164323_smoke_rydberg_kan_smoke/20260620_164324_smoke_rydberg_kan_mlp_pair_rydberg_chain_N5_733f7934fd28_seed0/run_metadata.json
results/20260620_164323_smoke_rydberg_kan_smoke/20260620_164324_smoke_rydberg_kan_product_rydberg_chain_N5_733f7934fd28_seed0/config.yaml
results/20260620_164323_smoke_rydberg_kan_smoke/20260620_164324_smoke_rydberg_kan_product_rydberg_chain_N5_733f7934fd28_seed0/environment_snapshot.txt
results/20260620_164323_smoke_rydberg_kan_smoke/20260620_164324_smoke_rydberg_kan_product_rydberg_chain_N5_733f7934fd28_seed0/experiment.log
results/20260620_164323_smoke_rydberg_kan_smoke/20260620_164324_smoke_rydberg_kan_product_rydberg_chain_N5_733f7934fd28_seed0/history.csv
results/20260620_164323_smoke_rydberg_kan_smoke/20260620_164324_smoke_rydberg_kan_product_rydberg_chain_N5_733f7934fd28_seed0/instance.json
results/20260620_164323_smoke_rydberg_kan_smoke/20260620_164324_smoke_rydberg_kan_product_rydberg_chain_N5_733f7934fd28_seed0/metrics.csv
results/20260620_164323_smoke_rydberg_kan_smoke/20260620_164324_smoke_rydberg_kan_product_rydberg_chain_N5_733f7934fd28_seed0/metrics.json
results/20260620_164323_smoke_rydberg_kan_smoke/20260620_164324_smoke_rydberg_kan_product_rydberg_chain_N5_733f7934fd28_seed0/model.pt
results/20260620_164323_smoke_rydberg_kan_smoke/20260620_164324_smoke_rydberg_kan_product_rydberg_chain_N5_733f7934fd28_seed0/observables.npz
results/20260620_164323_smoke_rydberg_kan_smoke/20260620_164324_smoke_rydberg_kan_product_rydberg_chain_N5_733f7934fd28_seed0/reproducibility_manifest.json
results/20260620_164323_smoke_rydberg_kan_smoke/20260620_164324_smoke_rydberg_kan_product_rydberg_chain_N5_733f7934fd28_seed0/run_metadata.json
results/20260620_164323_smoke_rydberg_kan_smoke/all_metrics.csv
results/20260620_164323_smoke_rydberg_kan_smoke/config.yaml
results/20260620_164323_smoke_rydberg_kan_smoke/dataset_instances.json
results/dataset_instances.json
scripts/aggregate_results.py
scripts/make_figures.py
scripts/run_ablation.py
scripts/run_full.py
scripts/run_laptop.py
scripts/run_robustness.py
scripts/run_smoke.py
src/graphkan_amo_nqs/__init__.py
src/graphkan_amo_nqs/__pycache__/__init__.cpython-313.pyc
src/graphkan_amo_nqs/__pycache__/config.cpython-313.pyc
src/graphkan_amo_nqs/__pycache__/device.cpython-313.pyc
src/graphkan_amo_nqs/__pycache__/seed.cpython-313.pyc
src/graphkan_amo_nqs/config.py
src/graphkan_amo_nqs/data/__init__.py
src/graphkan_amo_nqs/data/__pycache__/__init__.cpython-313.pyc
src/graphkan_amo_nqs/data/__pycache__/datasets.cpython-313.pyc
src/graphkan_amo_nqs/data/__pycache__/preprocessing.cpython-313.pyc
src/graphkan_amo_nqs/data/__pycache__/synthetic.cpython-313.pyc
src/graphkan_amo_nqs/data/datasets.py
src/graphkan_amo_nqs/data/preprocessing.py
src/graphkan_amo_nqs/data/synthetic.py
src/graphkan_amo_nqs/device.py
src/graphkan_amo_nqs/evaluation/__init__.py
src/graphkan_amo_nqs/evaluation/__pycache__/__init__.cpython-313.pyc
src/graphkan_amo_nqs/evaluation/__pycache__/evaluate.cpython-313.pyc
src/graphkan_amo_nqs/evaluation/__pycache__/metrics.cpython-313.pyc
src/graphkan_amo_nqs/evaluation/calibration.py
src/graphkan_amo_nqs/evaluation/evaluate.py
src/graphkan_amo_nqs/evaluation/metrics.py
src/graphkan_amo_nqs/evaluation/robustness.py
src/graphkan_amo_nqs/experiments/__init__.py
src/graphkan_amo_nqs/experiments/__pycache__/__init__.cpython-313.pyc
src/graphkan_amo_nqs/experiments/__pycache__/run_experiment.cpython-313.pyc
src/graphkan_amo_nqs/experiments/run_ablation.py
src/graphkan_amo_nqs/experiments/run_experiment.py
src/graphkan_amo_nqs/experiments/run_robustness.py
src/graphkan_amo_nqs/models/__init__.py
src/graphkan_amo_nqs/models/__pycache__/__init__.cpython-313.pyc
src/graphkan_amo_nqs/models/__pycache__/baseline_models.cpython-313.pyc
src/graphkan_amo_nqs/models/__pycache__/proposed_model.cpython-313.pyc
src/graphkan_amo_nqs/models/baseline_models.py
src/graphkan_amo_nqs/models/proposed_model.py
src/graphkan_amo_nqs/plotting/__init__.py
src/graphkan_amo_nqs/plotting/__pycache__/__init__.cpython-313.pyc
src/graphkan_amo_nqs/plotting/__pycache__/paper_figures.cpython-313.pyc
src/graphkan_amo_nqs/plotting/__pycache__/plots.cpython-313.pyc
src/graphkan_amo_nqs/plotting/paper_figures.py
src/graphkan_amo_nqs/plotting/plots.py
src/graphkan_amo_nqs/seed.py
src/graphkan_amo_nqs/training/__init__.py
src/graphkan_amo_nqs/training/__pycache__/__init__.cpython-313.pyc
src/graphkan_amo_nqs/training/__pycache__/losses.cpython-313.pyc
src/graphkan_amo_nqs/training/__pycache__/optim.cpython-313.pyc
src/graphkan_amo_nqs/training/__pycache__/train.cpython-313.pyc
src/graphkan_amo_nqs/training/losses.py
src/graphkan_amo_nqs/training/optim.py
src/graphkan_amo_nqs/training/train.py
src/graphkan_amo_nqs/utils/__pycache__/io.cpython-313.pyc
src/graphkan_amo_nqs/utils/__pycache__/logging_utils.cpython-313.pyc
src/graphkan_amo_nqs/utils/__pycache__/tables.cpython-313.pyc
src/graphkan_amo_nqs/utils/io.py
src/graphkan_amo_nqs/utils/logging_utils.py
src/graphkan_amo_nqs/utils/tables.py
src/graphkan_amo_nqs/utils/timing.py
tests/__pycache__/conftest.cpython-313-pytest-9.0.2.pyc
tests/__pycache__/test_config.cpython-313-pytest-9.0.2.pyc
tests/__pycache__/test_data_pipeline.cpython-313-pytest-9.0.2.pyc
tests/__pycache__/test_device.cpython-313-pytest-9.0.2.pyc
tests/__pycache__/test_imports.cpython-313-pytest-9.0.2.pyc
tests/__pycache__/test_loss_function.cpython-313-pytest-9.0.2.pyc
tests/__pycache__/test_metrics.cpython-313-pytest-9.0.2.pyc
tests/__pycache__/test_model_shapes.cpython-313-pytest-9.0.2.pyc
tests/__pycache__/test_plotting.cpython-313-pytest-9.0.2.pyc
tests/__pycache__/test_seed_reproducibility.cpython-313-pytest-9.0.2.pyc
tests/__pycache__/test_smoke_run.cpython-313-pytest-9.0.2.pyc
tests/conftest.py
tests/test_config.py
tests/test_data_pipeline.py
tests/test_device.py
tests/test_imports.py
tests/test_loss_function.py
tests/test_metrics.py
tests/test_model_shapes.py
tests/test_plotting.py
tests/test_seed_reproducibility.py
tests/test_smoke_run.py
```

## Exact proposal package inventory

```text
ANTI_SCOOP_LITERATURE_MATRIX.csv
BUILD_INSTRUCTIONS.md
README_OVERLEAF.md
appendices/appendix_a_literature_matrix.tex
appendices/appendix_b_mathematical_details.tex
appendices/appendix_c_software_spec.tex
appendices/appendix_d_experiment_configs.tex
appendices/appendix_e_reviewer_response_bank.tex
build.log
docs/ANTI_SCOOP_LITERATURE_REVIEW.md
docs/AUDIT_CURRENT_PROJECT_STATE.md
docs/CLAIM_AUDIT.md
docs/COMPILE_LOG.txt
docs/EXPERIMENT_EXECUTION_PLAN.md
docs/IBM_QUANTUM_OPTIONAL_VALUE_DECISION.md
docs/PRODUCTION_CODE_CONTRACT.md
docs/PROJECT_GO_NO_GO.md
docs/REPRODUCIBILITY_PLAN.md
docs/TOP_JOURNAL_READINESS.md
docs/VENUE_STRATEGY.md
figures/fig_benchmark_splits.pdf
figures/fig_blockade_scaled_kan.pdf
figures/fig_claim_ladder.pdf
figures/fig_execution_graph.pdf
figures/fig_gauge_fixed_interpretability.pdf
figures/fig_project_pipeline.pdf
macros.tex
main.bbl
main.pdf
main.tex
references.bib
sections/01_executive_summary.tex
sections/02_research_question_and_claim_ladder.tex
sections/03_updated_literature_and_anti_scoop.tex
sections/04_repaired_invention_kernel.tex
sections/05_physical_and_mathematical_framework.tex
sections/06_methods_and_ansatzes.tex
sections/07_benchmark_design_and_splits.tex
sections/08_baselines_and_fairness_protocol.tex
sections/09_interpretability_and_gauge_fixing.tex
sections/10_experiment_matrix.tex
sections/11_statistics_and_uncertainty.tex
sections/12_software_architecture_blueprint.tex
sections/13_ibm_quantum_optional_decision.tex
sections/14_venue_strategy_and_go_no_go.tex
sections/15_risk_register_and_reviewer_attacks.tex
sections/16_execution_plan.tex
sections/17_open_science_and_artifact_plan.tex
sections/18_conclusion.tex
tables/ablation_matrix.tex
tables/anti_scoop_matrix.tex
tables/baseline_hierarchy.tex
tables/benchmark_tasks.tex
tables/claim_ladder.tex
tables/code_generation_contract.tex
tables/experiment_matrix.tex
tables/literature_matrix.tex
tables/metric_suite.tex
tables/reviewer_objections.tex
tables/risk_register.tex
tables/software_module_spec.tex
tables/venue_strategy.tex
```

## Baseline tests

Command:

```bash
cd /mnt/data/graphkan_input_audit/codebase/graphkan_amo_nqs
PYTHONPATH=src python -m pytest -q tests/
```

Result excerpt:

```text
...............                                                          [100%]
```

## Baseline smoke run

Command:

```bash
PYTHONPATH=src python scripts/run_smoke.py --config configs/smoke.yaml
```

Result excerpt:

```text
2026-06-20 16:43:24,920 | INFO | Starting run: seed=0 instance=rydberg_chain_N5_733f7934fd28 model=product device=cpu
2026-06-20 16:43:24,923 | INFO | Exact training with 32 basis states and 160 transitions
2026-06-20 16:43:24,938 | INFO | Completed run in 0.31 seconds
2026-06-20 16:43:24,953 | INFO | Starting run: seed=0 instance=rydberg_chain_N5_733f7934fd28 model=mlp_pair device=cpu
2026-06-20 16:43:24,954 | INFO | Exact training with 32 basis states and 160 transitions
2026-06-20 16:43:24,969 | INFO | Completed run in 0.03 seconds
2026-06-20 16:43:24,980 | INFO | Starting run: seed=0 instance=rydberg_chain_N5_733f7934fd28 model=kan_jastrow device=cpu
2026-06-20 16:43:24,981 | INFO | Exact training with 32 basis states and 160 transitions
2026-06-20 16:43:24,993 | INFO | Completed run in 0.02 seconds
Smoke run completed. Results root: results/20260620_164323_smoke_rydberg_kan_smoke
```

## Current model/backend status

The uploaded codebase is a useful PyTorch-style exact/smoke scaffold with deterministic synthetic data, simple baseline/proposed models, exact small-basis training, metrics, plotting, and smoke scripts. It does not contain a production NetKet/JAX/Flax VMC engine, QuSpin cross-checks, TeNPy references, serious graph transfer splits, full baseline fairness, sampler diagnostics, or a journal-grade statistical pipeline.

## What is real

- A passing unit-test suite.
- A runnable smoke script producing small exact-basis runs.
- Basic product, MLP-pair, and KAN-Jastrow code paths.
- Simple configs, metrics, plots, and reproducibility manifest.

## What is scaffold

- Scientific result claims.
- Final Rydberg benchmark evidence.
- Full VMC and sampler diagnostics.
- Strong baselines such as production RBM, trainable MLP-pair, RNN, transformer, Graph-MPNN, Graph-KAN.
- ED/DMRG cross-reference hierarchy beyond tiny smoke tasks.

## Missing relative to the v2 proposal

- NetKet/JAX/Flax production VMC.
- Custom dense/sparse Rydberg Hamiltonian validation across task families.
- QuSpin optional cross-check and TeNPy/DMRG reference integration.
- Blockade-scale-normalised feature policies and gauge-fixed learned-function export.
- Long-format result schemas with per-row reproducibility metadata.
- Paired statistics, bootstrap intervals, Holm correction, failure retention.
- Claim validation and readiness gates.

## Supported claims

The baseline code can support only that the prototype imports, passes 15 tests, and executes a tiny smoke path. It can also support internal development planning.

## Unsupported claims

It cannot support KAN-Jastrow competitiveness, Graph-KAN value, Rydberg benchmark conclusions, interpretability, geometry transfer, PRA/MLST/CPC readiness, or top-journal claims.

## Modules replaced or upgraded in this repository

This production artifact adds deterministic geometry, Rydberg Hamiltonians, exact diagonalisation, exact full-sum variational training, observables, interpretability, paired statistics, schema validation, plotting, LaTeX reporting, claim auditing, and release manifests.

## Production-coding readiness

The v2 proposal is production-coding ready as a specification. The original uploaded code is not journal-ready and should not be cosmetically expanded. It should be upgraded into a real benchmark engine, which is what this repository starts.

## Journal readiness

Not journal-ready. The project is at a CPU exact-reference pilot stage after this run. PRA/MLST/CPC readiness requires final VMC/DMRG baselines, more seeds, geometry transfer, robustness, and final release hardening.
