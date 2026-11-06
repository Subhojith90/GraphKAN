# Stage 4H radial-warm-start optimisation discriminator

Stage 4H is the sole approved post-Stage-4G VMC campaign. It tests whether the
N=12 endpoint result was caused by optimisation/conditioning or persists after
the expanded models are placed exactly in the matched radial solution.

The protocol is prospective and immutable: N=12 open chains, three registered
cells, seeds 0--4, four continuation arms, 4096 training samples, exact
all-state primary observables, and sentinels at +3000 and +6000 steps. Every
expanded parameter absent from the radial model is initialised to literal zero.

## Local-only execution

```bash
export PYTHONPATH="$PWD/src:$PWD/scripts"
export JAX_ENABLE_X64=true
python scripts/run_stage4h_local.py prepare
caffeinate -dimsu python -u scripts/run_stage4h_local.py run --max-workers 2
```

The launcher is restart-safe. A repeated `run` skips completed bundles and
creates a new immutable attempt for interrupted or failed work. It never
deletes failed attempts. To inspect progress from another terminal:

```bash
python scripts/run_stage4h_local.py status
tail -f results_stage4h/local_campaign_20260807/logs/*.log
```

After 12/12 shards and 60/60 runs complete:

```bash
python scripts/run_stage4h_local.py finalize
python scripts/package_stage4h.py
shasum -a 256 -c releases/GraphKAN_Stage4H_Complete_Evidence_Local_20260807.zip.sha256
unzip -tq releases/GraphKAN_Stage4H_Complete_Evidence_Local_20260807.zip
```

Use `--max-workers 1` on an 8 GB machine or whenever macOS reports sustained
memory pressure. Two workers are the intended default for a plugged-in Apple
Silicon laptop. Keep the lid open: `caffeinate` prevents idle sleep but cannot
override macOS clamshell sleep without an external display.

## Decision semantics

- `SCIENTIFIC_RESCUE_PASS`: an energy-safe endpoint arm restores joint density
  and correlation repair in at least two cells with at least four positive
  paired seeds per passing cell.
- `NEGATIVE_CLOSURE_PASS`: energy safety closes, but joint repair does not;
  freeze the optimisation-controlled finite-size no-benefit conclusion.
- `OPTIMISATION_UNRESOLVED`: neither endpoint arm meets 14/15 energy safety;
  only one prospectively frozen orthogonalised-feature pilot is authorised.

The local launcher preserves all successful, failed, interrupted, and partial
evidence before enforcing the scientific gate.
