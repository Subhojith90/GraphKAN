#!/usr/bin/env bash
set -euo pipefail
mkdir -p results_stage4c/clean_replay
LOG=results_stage4c/clean_replay/stage4c_clean_replay.log
{
  echo "=== Stage 4C clean replay ==="
  date -u +"%Y-%m-%dT%H:%M:%SZ"
  python --version
  python -m pip install -e '.[nqs,dev]'
  PYTHONPATH=src python -m pytest -q -ra
  PYTHONPATH=src python scripts/preflight_nqs.py --output results_stage4c/clean_replay/nqs_preflight.json
  PYTHONPATH=src python scripts/check_netket_conventions.py --config configs/netket_stage4c_n8_conventions.yaml
  python - <<'PY'
import jax, flax, netket, numpy, optax, scipy
print({"jax":jax.__version__, "jaxlib":jax.lib.__version__, "flax":flax.__version__, "netket":netket.__version__, "numpy":numpy.__version__, "optax":optax.__version__, "scipy":scipy.__version__})
PY
} 2>&1 | tee "$LOG"
