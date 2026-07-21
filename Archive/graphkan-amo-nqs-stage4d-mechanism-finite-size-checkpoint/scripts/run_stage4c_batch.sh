#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -lt 1 ]; then
  echo "Usage: bash scripts/run_stage4c_batch.sh <config.yaml> [runner options]" >&2
  exit 2
fi
CONFIG="$1"; shift
ROOT=$(python -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["output_root"])' "$CONFIG")
mkdir -p "$ROOT/logs"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG="$ROOT/logs/$(basename "$CONFIG" .yaml)_${STAMP}.log"
{
  echo "=== Stage 4C VMC batch ==="
  echo "UTC: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "Command: PYTHONPATH=src python scripts/run_stage4c_patch.py --config $CONFIG $*"
  python --version
  PYTHONPATH=src python scripts/run_stage4c_patch.py --config "$CONFIG" "$@"
} 2>&1 | tee "$LOG"
echo "Wrote $LOG"
