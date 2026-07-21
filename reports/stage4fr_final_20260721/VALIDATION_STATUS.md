# Stage 4F-R regenerated report validation

Generated: 21 July 2026

## Canonical campaign audit

- Campaign: `results_stage4f_r/frozen`
- Schema: `stage4f-native-v1c`
- Complete bundles checked: **960**
- Registry total: **960 COMPLETE**
- Core lane: **840 COMPLETE**
- Control lane: **120 COMPLETE**
- Duplicate registry keys: **0**
- Registry/filesystem errors: **0**
- Blocking errors: **0**
- SQLite integrity check: **ok**
- Strict validation: **PASS**
- Repository test suite: **PASS**

No experiment reruns were required because no canonical bundle was damaged or
missing. Empty `stdout.log`/`stderr.log` files are valid logs from runs that
produced no output on that stream and are not evidence of corruption.

## Regenerated deliverables

The PDF, tables, figures, and JSON summaries were regenerated from the canonical
campaign after the audit. The PDF is stored under `../../output/pdf/`; it is an
unencrypted, JavaScript-free, eight-page A4 document.

Verify the report files from this directory with:

```bash
shasum -a 256 -c SHA256SUMS.txt
```

Verify the canonical campaign from the repository root with:

```bash
export PYTHONPATH="$PWD/src"
python scripts/stage4f1c_campaign.py validate \
  --campaign results_stage4f_r/frozen \
  --strict-integrity \
  --reconcile-registry \
  --require-terminal \
  --require-complete
```
