# GraphKAN AMO-NQS — Stage 4F-R Publication Repository

This repository contains the complete code, frozen protocol, execution evidence,
analysis, and supervisory handoff material for the GraphKAN AMO neural quantum
state Stage 4F-R campaign.

## Clone and verify

```bash
git clone https://github.com/Subhojith90/GraphKAN.git
cd GraphKAN
bash verify_repository.sh
```

Run all commands from the cloned repository root. The verifier checks the full
960-run campaign, regenerated report checksums, and repository test suite.

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

### Stage 4F-R.1 integrity reconstruction

The hardened reconstruction is available at
`releases/GraphKAN_Stage4FR1b_Hardened_20260722.zip` with its adjacent
`.zip.sha256` sidecar. It closes 11,520/11,520 raw payload hashes, 960/960
completion/registry hashes, and the exact authoritative 105/105 campaign
source ledger. The post-campaign release tools have a distinct ledger. The
builder performs a ZIP CRC test and runs the complete verifier from a clean
extraction.

The hardened independent 20-run Ubuntu replay is intentionally marked
**PENDING**. Run the manual GitHub Actions workflow `Stage 4F-R.1b hardened independent Linux replay` and
return its artifact before claiming publication-grade integrity closure. See
`docs/STAGE4FR1_RELEASE_STATUS.md` for the exact boundary.

## Repository map

- `src/graphkan_amo_nqs/` — research implementation
- `scripts/` — execution, validation, analysis, and finalization tools
- `configs/` — frozen and historical experiment configurations
- `tests/` and `release_tests/` — regression and release checks
- `results_stage4f_r/frozen/` — canonical Stage 4F-R campaign evidence
- `reports/stage4fr_final_20260721/` — regenerated tables, figures, and validation ledger
- `output/pdf/` — visually verified supervisory report
- `results_stage4f1c/` — scheduler, recovery, replay, and microcampaign evidence
- `docs/` — scientific reports and supervision material
- `docs/tracking/` — stage history, changelogs, repair logs, and execution records
- `reproducibility/` and `requirements/` — portable environments and dependency locks

Earlier `results_stage*` directories are retained as traceable research history.
Duplicate extracted release folders remain excluded; the single audited
Stage 4F-R.1 archive and its checksum sidecar are versioned deliberately.

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
  --analysis-output reports/stage4fr_final_20260721/analysis \
  --report output/pdf/GraphKAN_Stage4FR_Final_Validation_Report_20260721.pdf \
  --skip-package
```

The finalizer now creates a normal review directory and does not create a ZIP.

## Primary review documents

- `output/pdf/GraphKAN_Stage4FR_Final_Validation_Report_20260721.pdf`
- `reports/stage4fr_final_20260721/analysis/final_gate_summary.json`
- `docs/tracking/stage4f/README_STAGE4F1C.md`
- `REPRODUCIBILITY.md`
- `LIMITATIONS_AND_UNSAFE_CLAIMS.md` is maintained under
  `docs/LIMITATIONS_AND_UNSAFE_CLAIMS.md`

## Reproducibility boundary

The included replay is technically clean but was executed on the same Mac as
the main campaign. It must not be represented as genuine second-host evidence.
Any public-release or next-stage claim should preserve that limitation unless a
separate host replay is completed and documented.
