#!/usr/bin/env bash
set -euo pipefail
mkdir -p artifacts/clean_replay
LOG=artifacts/clean_replay/stage3f_clean_replay.log
{
  echo "=== Stage 3F clean replay ==="
  date -u +"%Y-%m-%dT%H:%M:%SZ"
  python --version
  python -m pip install -e '.[nqs,dev]'
  PYTHONPATH=src python -m pytest -q -ra
  PYTHONPATH=src python scripts/preflight_nqs.py
  PYTHONPATH=src python scripts/check_netket_conventions.py --config configs/netket_stage2_6_conventions.yaml
  python - <<'PY'
import jax, flax, netket, numpy, optax, scipy
print({"jax":jax.__version__,"flax":flax.__version__,"netket":netket.__version__,"numpy":numpy.__version__,"optax":optax.__version__,"scipy":scipy.__version__})
PY
} 2>&1 | tee "$LOG"
