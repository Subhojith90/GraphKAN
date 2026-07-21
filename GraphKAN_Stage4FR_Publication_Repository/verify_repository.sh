#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

echo "[1/3] Validating the frozen 960-run campaign"
python scripts/stage4f1c_campaign.py validate \
  --campaign results_stage4f_r/frozen \
  --strict-integrity \
  --reconcile-registry \
  --require-terminal \
  --require-complete

echo "[2/3] Verifying regenerated report checksums"
if command -v sha256sum >/dev/null 2>&1; then
  (cd reports/stage4fr_final_20260721 && sha256sum -c SHA256SUMS.txt)
elif command -v shasum >/dev/null 2>&1; then
  (cd reports/stage4fr_final_20260721 && shasum -a 256 -c SHA256SUMS.txt)
else
  echo "Neither sha256sum nor shasum is installed" >&2
  exit 1
fi

echo "[3/3] Running the repository test suite"
python -m pytest -q

echo "GraphKAN Stage 4F-R repository verification: PASS"

