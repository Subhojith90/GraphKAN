#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/run_stage4d_batch.sh <config.yaml> [runner args...]" >&2
  exit 2
fi
CONFIG="$1"; shift
OUTROOT="$(python - "$CONFIG" <<'PY'
import sys,yaml
print(yaml.safe_load(open(sys.argv[1]))['output_root'])
PY
)"
mkdir -p "$OUTROOT/logs"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$OUTROOT/logs/$(basename "$CONFIG" .yaml)_${STAMP}.log"
echo "=== Stage 4D VMC batch ===" | tee "$LOG"
echo "UTC: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
echo "Command: PYTHONPATH=src python scripts/run_stage4d.py --config $CONFIG $*" | tee -a "$LOG"
PYTHONPATH=src python scripts/run_stage4d.py --config "$CONFIG" "$@" 2>&1 | tee -a "$LOG"
echo "Wrote $LOG"
