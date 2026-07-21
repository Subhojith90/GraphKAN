#!/usr/bin/env bash
set -euo pipefail
mkdir -p results_stage4/clean_replay
LOG="results_stage4/clean_replay/stage4_clean_replay.log"
{
  echo "=== Stage 4 clean replay ==="
  date -u +"%Y-%m-%dT%H:%M:%SZ"
  python --version
  python -m pip install -e '.[nqs,dev]'
  PYTHONPATH=src python -m pytest -q -ra
  PYTHONPATH=src python scripts/preflight_nqs.py
  PYTHONPATH=src python scripts/check_netket_conventions.py \
    --config configs/netket_stage2_6_conventions.yaml
  python - <<'PY'
import importlib.metadata as md
for p in ['jax','jaxlib','flax','optax','netket','numpy','scipy']:
    print(f'{p}={md.version(p)}')
PY
} 2>&1 | tee "$LOG"
echo "Wrote $LOG"
