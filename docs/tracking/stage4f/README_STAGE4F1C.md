# Stage 4F.1c Schema-Fenced Real-Runner and Crash-Recovery Gate

Stage 4F.1c repairs the completion invariant. It does not by itself supply a new
physics result or authorise manuscript claims.

Implemented controls:

- expected-work specifications stored in the registry and bound into every bundle;
- path-invariant protocol identity covering physics, scheduler policy, base configs,
  environment, complete source tree, and runner/transaction scripts;
- versioned typed bundle validation with exact row cardinalities and finite values;
- actual `jax-tree-leaves-npz-v1` checkpoint deserialization with shape summaries;
- complete, safe artifact-manifest coverage and a bound `COMPLETE.json` hash;
- active worker heartbeats and cryptographic attempt/claim fencing;
- configured retry, backoff, terminal failure, and quarantine behavior;
- same-host no-replace publication locks with dead-owner recovery;
- preservation of invalid or stale canonical bundles under `quarantine/`;
- bidirectional registry/filesystem/hash reconciliation;
- truthful nonzero worker exits;
- a real single-run adapter around `scripts/run_stage4e.py`;
- deterministic SIGKILL hooks for training, checkpoint, promotion, and
  post-promotion recovery tests.

## Environment

Run all commands from the repository root in the environment that contains JAX,
Flax, Optax, NetKet, NumPy, pandas, PyYAML, and pytest.

```bash
export PYTHONPATH="$PWD/scripts:$PWD/src"
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export XDG_CACHE_HOME="$PWD/.cache"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
```

## Tier 0: code and semantic gates

```bash
python -m py_compile scripts/stage4f1b_tx.py scripts/stage4f1b_worker.py scripts/stage4f1b_campaign.py scripts/stage4f1c_crash_probe.py
python -m pytest -q tests/test_stage4f1c_closure.py tests/test_stage4f1b_semantic_lease.py tests/test_stage4f1_transactional_scheduler.py
python scripts/stage4f1c_campaign.py semantic-test --config configs/stage4f1c_campaign.yaml --output results_stage4f1c/semantic_tests
python scripts/stage4f1c_campaign.py lease-test --config configs/stage4f1c_campaign.yaml --output results_stage4f1c/lease_tests --wait 1.2
python scripts/stage4f1c_crash_probe.py --config configs/stage4f1c_crash_probe.yaml --output results_stage4f1c/crash_probe
```

All commands must exit zero. The semantic summary must report zero false accepts,
and all four crash phases must report `pass: true` with a fault return code of `-9`.

## Tier 1: four-run real adapter smoke

```bash
python scripts/stage4f1c_campaign.py init --config configs/stage4f1c_adapter_smoke.yaml
python scripts/stage4f1c_worker.py --config configs/stage4f1c_adapter_smoke.yaml --lane core --worker-id "$(hostname)-core-smoke" --real-run --max-runs 100
python scripts/stage4f1c_worker.py --config configs/stage4f1c_adapter_smoke.yaml --lane control --worker-id "$(hostname)-control-smoke" --real-run --max-runs 100
python scripts/stage4f1c_campaign.py validate --campaign results_stage4f1c/adapter_smoke_final --strict-integrity --reconcile-registry --require-complete
```

The two workers may run in separate terminals on the same host. Do not put one
SQLite WAL registry on a network filesystem.

## Tier 2: real 16-run kill/restart microcampaign

The roster covers open/periodic boundaries, two registered physics cells, radial
RBF, endpoint-product RBF, a strong MLP control, and an RBM control.

The preferred execution is the guarded one-command orchestrator. It refuses to
reuse an existing registry, performs all four SIGKILL/recovery phases, drains both
lanes, requires 16/16 COMPLETE, and writes `kill_microcampaign_summary.json`.

```bash
python scripts/run_stage4f1c_kill_microcampaign.py \
  --config configs/stage4f1c_real_microcampaign_killtest.yaml
```

Use the manual commands below only for debugging the orchestrator.

```bash
python scripts/stage4f1c_campaign.py init --config configs/stage4f1c_real_microcampaign_killtest.yaml
```

Before ordinary workers, perform four planned kills. Each command is expected to
terminate with SIGKILL; wait at least 31 seconds after each kill and then run one
normal recovery attempt in the same lane.

```bash
python scripts/stage4f1c_worker.py --config configs/stage4f1c_real_microcampaign_killtest.yaml --lane core --worker-id kill-training --real-run --max-runs 1 --fault-phase during-training --fault-delay 2
sleep 31
python scripts/stage4f1c_worker.py --config configs/stage4f1c_real_microcampaign_killtest.yaml --lane core --worker-id recover-training --real-run --max-runs 1

python scripts/stage4f1c_worker.py --config configs/stage4f1c_real_microcampaign_killtest.yaml --lane core --worker-id kill-checkpoint --real-run --max-runs 1 --fault-phase during-checkpoint
sleep 31
python scripts/stage4f1c_worker.py --config configs/stage4f1c_real_microcampaign_killtest.yaml --lane core --worker-id recover-checkpoint --real-run --max-runs 1

python scripts/stage4f1c_worker.py --config configs/stage4f1c_real_microcampaign_killtest.yaml --lane control --worker-id kill-promotion --real-run --max-runs 1 --fault-phase during-promotion
sleep 31
python scripts/stage4f1c_worker.py --config configs/stage4f1c_real_microcampaign_killtest.yaml --lane control --worker-id recover-promotion --real-run --max-runs 1

python scripts/stage4f1c_worker.py --config configs/stage4f1c_real_microcampaign_killtest.yaml --lane core --worker-id kill-after-promotion --real-run --max-runs 1 --fault-phase after-promotion
sleep 31
python scripts/stage4f1c_worker.py --config configs/stage4f1c_real_microcampaign_killtest.yaml --lane core --worker-id recover-after-promotion --real-run --max-runs 1
```

Then drain both lanes, preferably in separate same-host terminals:

```bash
python scripts/stage4f1c_worker.py --config configs/stage4f1c_real_microcampaign_killtest.yaml --lane core --worker-id "$(hostname)-core-0" --real-run --max-runs 100
python scripts/stage4f1c_worker.py --config configs/stage4f1c_real_microcampaign_killtest.yaml --lane control --worker-id "$(hostname)-control-0" --real-run --max-runs 100
python scripts/stage4f1c_campaign.py validate --campaign results_stage4f1c/real_microcampaign_killtest --strict-integrity --reconcile-registry --require-complete
```

The final command must show `COMPLETE: 16`, zero reconciliation errors, and
`validation: PASS`.

## Independent replay and Stage 4F-R

Copy a clean, renamed extraction to a second host and replay four stratified rows
using a separate local registry. Compare the resulting metrics within the frozen
numeric tolerances and retain the complete stdout/stderr, manifests, and machine
environment. Do not start the 960-run Stage 4F-R campaign until the independent
replay is reviewed and signed off.

On a clean extraction on the second host:

```bash
export PYTHONPATH="$PWD/scripts:$PWD/src"
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export XDG_CACHE_HOME="$PWD/.cache"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

python -m pytest -q
python scripts/stage4f1c_campaign.py init --config configs/stage4f1c_independent_replay.yaml
python scripts/stage4f1c_worker.py --config configs/stage4f1c_independent_replay.yaml --lane core --worker-id "$(hostname)-replay-core" --real-run --max-runs 100
python scripts/stage4f1c_worker.py --config configs/stage4f1c_independent_replay.yaml --lane control --worker-id "$(hostname)-replay-control" --real-run --max-runs 100
python scripts/stage4f1c_campaign.py validate --campaign results_stage4f1c/independent_replay --strict-integrity --reconcile-registry --require-complete
```

The publication package remains blocked until the frozen 960-run campaign is fully
terminal and reconciled, the preregistered statistical tables and figures are
generated without post-hoc roster changes, and the independent replay is included.
