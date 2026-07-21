#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/run_stage4e_batch.sh <config.yaml> [runner args...]" >&2
  exit 2
fi
CONFIG="$1"; shift
OUTROOT="$(python - "$CONFIG" <<'PY'
import sys,yaml
print(yaml.safe_load(open(sys.argv[1]))['output_root'])
PY
)"
mkdir -p "$OUTROOT/logs"
REQUIRES_SMOKE="$(python - "$CONFIG" <<'PY2'
import sys, yaml
print(str(bool(yaml.safe_load(open(sys.argv[1])).get("requires_smoke_gate", False))).lower())
PY2
)"
SMOKE_PATH="$(python - "$CONFIG" <<'PY2'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1])).get("smoke_gate_path", ""))
PY2
)"
if [[ "$REQUIRES_SMOKE" == "true" ]]; then
  if [[ ! -f "$SMOKE_PATH" ]]; then
    echo "ERROR: Stage 4E mechanism batch is blocked until full-budget smoke gate exists: $SMOKE_PATH" >&2
    exit 3
  fi
  STATUS="$(python - "$SMOKE_PATH" <<'PY2'
import sys, json
print(json.load(open(sys.argv[1])).get("status", "MISSING"))
PY2
)"
  if [[ "$STATUS" != "PASS" ]]; then
    echo "ERROR: Stage 4E mechanism batch is blocked because smoke gate status is $STATUS" >&2
    exit 3
  fi
fi
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$OUTROOT/logs/$(basename "$CONFIG" .yaml)_${STAMP}.log"
echo "=== Stage 4E VMC batch ===" | tee "$LOG"
echo "UTC: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
echo "Command: PYTHONPATH=src python scripts/run_stage4e.py --config $CONFIG $*" | tee -a "$LOG"
PYTHONPATH=src python scripts/run_stage4e.py --config "$CONFIG" "$@" 2>&1 | tee -a "$LOG"
echo "Wrote $LOG"
