# GraphKAN AMO-NQS — Stage 4F-R Publication Repository

This repository contains the complete code, frozen protocol, execution evidence,
analysis, and supervisory handoff material for the GraphKAN AMO neural quantum
state Stage 4F-R campaign.

## Validation status

- Frozen campaign: `results_stage4f_r/frozen`
- Registered runs: **960/960 COMPLETE**
- Core lane: **840 COMPLETE**
- Control lane: **120 COMPLETE**
- Strict registry/filesystem reconciliation: **PASS**
- Frozen protocol hash:
  `327f66318ee93c3365f5a12ba4d64fddd20a29ee288cc30330d437f514328ff9`

The supplied evidence supports completion and integrity of the registered
Stage 4F-R N=10 benchmark. It does not establish generic KAN, endpoint-feature,
Graph-KAN, or top-journal superiority. Descriptive rankings apply only to the
frozen registered grid.

## Repository map

- `src/graphkan_amo_nqs/` — research implementation
- `scripts/` — execution, validation, analysis, and finalization tools
- `configs/` — frozen and historical experiment configurations
- `tests/` and `release_tests/` — regression and release checks
- `results_stage4f_r/frozen/` — canonical Stage 4F-R campaign evidence
- `results_stage4f_r/publication_analysis/` — publication-stage tables and plots
- `results_stage4f1c/` — scheduler, recovery, replay, and microcampaign evidence
- `output/pdf/` — supervisory report
- `docs/` — scientific reports and supervision material
- `docs/tracking/` — stage history, changelogs, repair logs, and execution records
- `reproducibility/` and `requirements/` — portable environments and dependency locks

Earlier `results_stage*` directories are retained as traceable research history.
Generated ZIP handoffs and duplicate extracted release folders are deliberately
excluded from the active repository.

## Validate the frozen campaign

From the repository root:

```bash
export PYTHONPATH="$PWD/src"

python scripts/stage4f1c_campaign.py validate \
  --campaign results_stage4f_r/frozen \
  --strict-integrity \
  --reconcile-registry \
  --require-terminal \
  --require-complete
```

Confirm the registry totals:

```bash
sqlite3 results_stage4f_r/frozen/registry.sqlite \
  "SELECT lane,status,COUNT(*) FROM runs GROUP BY lane,status ORDER BY lane,status;"
```

Expected result:

```text
control|COMPLETE|120
core|COMPLETE|840
```

Run the test suite:

```bash
export PYTHONPATH="$PWD/src"
python -m pytest -q
```

## Rebuild the analysis and review directory

```bash
export PYTHONPATH="$PWD/src"

python scripts/finalize_stage4fr_publication.py \
  --campaign results_stage4f_r/frozen \
  --analysis-output results_stage4f_r/publication_analysis \
  --report output/pdf/GraphKAN_Stage4FR_Final_Validation_Handoff_20260720.pdf \
  --package stage4fr_review_directory
```

The finalizer now creates a normal review directory and does not create a ZIP.

## Primary review documents

- `output/pdf/GraphKAN_Stage4FR_Final_Validation_Handoff_20260720.pdf`
- `results_stage4f_r/publication_analysis/final_gate_summary.json`
- `docs/tracking/stage4f/README_STAGE4F1C.md`
- `REPRODUCIBILITY.md`
- `LIMITATIONS_AND_UNSAFE_CLAIMS.md` is maintained under
  `docs/LIMITATIONS_AND_UNSAFE_CLAIMS.md`

## Reproducibility boundary

The included replay is technically clean but was executed on the same Mac as
the main campaign. It must not be represented as genuine second-host evidence.
Any public-release or next-stage claim should preserve that limitation unless a
separate host replay is completed and documented.

