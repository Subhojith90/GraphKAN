# Stage 4G execution guide

For the full campaign, prefer the sharded Ubuntu workflow documented in
`GITHUB_ACTIONS_STAGE4G.md`; it does not require the Mac to remain powered on.
The commands below remain the local fallback and diagnostic route.

Run from the repository root. Keep open and periodic work in separate terminals;
each command is resumable and removes only an interrupted attempt before retry.

## 1. Preflight, protocol freeze, gauge gate and smoke

```bash
cd "/Users/subhojithalder/Desktop/Research Papers/Graph Kan/GraphKAN_Publication_Validated"
export PYTHONPATH="$PWD/src:$PWD/scripts"
export MPLCONFIGDIR="/tmp/graphkan-matplotlib-$USER"

bash verify_repository.sh
pytest -q tests/test_stage4g_protocol.py tests/test_stage4d_model_builds.py tests/test_periodic_flax_features_optional.py

python scripts/stage4g_protocol.py validate
python scripts/stage4g_protocol.py freeze --output results_stage4g/protocol
python scripts/stage4g_protocol.py verify-freeze --output results_stage4g/protocol
python scripts/stage4g_gauge.py

python -u scripts/run_stage4g.py --config configs/stage4g_smoke_open.yaml --overwrite
python -u scripts/run_stage4g.py --config configs/stage4g_smoke_periodic.yaml --overwrite
python scripts/validate_stage4g_smoke.py
```

Do not start the campaign unless the repository, protocol, gauge and smoke
checks all report `PASS`.

## 2. G2 convergence matching (two terminals)

Terminal A:

```bash
cd "/Users/subhojithalder/Desktop/Research Papers/Graph Kan/GraphKAN_Publication_Validated"
export PYTHONPATH="$PWD/src:$PWD/scripts"
caffeinate -dimsu python -u scripts/run_stage4g.py \
  --config results_stage4g/protocol/runner_configs/convergence_open.yaml --resume
```

Terminal B:

```bash
cd "/Users/subhojithalder/Desktop/Research Papers/Graph Kan/GraphKAN_Publication_Validated"
export PYTHONPATH="$PWD/src:$PWD/scripts"
caffeinate -dimsu python -u scripts/run_stage4g.py \
  --config results_stage4g/protocol/runner_configs/convergence_periodic.yaml --resume
```

Expected: 60 completed metrics rows and 180 exact-sentinel rows per boundary.
Check progress at any time with:

```bash
python scripts/stage4g_status.py --phase convergence
```

## 3. G3 disjoint control calibration (two terminals)

Terminal A uses `calibration_open.yaml`; Terminal B uses
`calibration_periodic.yaml`, with the same command form and `--resume`.
After both finish:

```bash
python scripts/stage4g_status.py --phase calibration
python scripts/stage4g_protocol.py select-controls \
  --output results_stage4g/protocol \
  --metrics \
    results_stage4g/calibration/open/raw/stage4g_metrics.csv \
    results_stage4g/calibration/periodic/raw/stage4g_metrics.csv
```

This writes `control_selection.json`, the immutable 180-row confirmation roster,
and the final open/periodic confirmation configurations. It refuses incomplete
calibration data.

## 4. G4 N=12 confirmation (two terminals)

Use the same resumable two-terminal pattern with:

```text
results_stage4g/protocol/runner_configs/confirmation_open.yaml
results_stage4g/protocol/runner_configs/confirmation_periodic.yaml
```

Expected: 90 completed runs per boundary. Never use `--overwrite` for a real
campaign; use `--resume` after any interruption.

```bash
python scripts/stage4g_status.py --phase confirmation
```

## 5. G5 final decision

```bash
python scripts/analyze_stage4g.py
```

The authoritative decision is
`results_stage4g/analysis/stage4g_final_decision.json`. A failed preregistered
gate remains a scientific result and must not be hidden by changing cells,
seeds, thresholds or model labels after execution.
