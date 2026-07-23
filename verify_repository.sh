#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

echo "[1/4] Verifying raw payload, completion, registry and frozen-source hashes"
python scripts/verify_stage4fr1_release.py --root .
python scripts/verify_campaign_ledger_identity.py --root .

echo "[2/4] Validating the frozen 960-run campaign"
python scripts/stage4f1c_campaign.py validate \
  --campaign results_stage4f_r/frozen \
  --strict-integrity \
  --reconcile-registry \
  --require-terminal \
  --require-complete \
  --no-write

echo "[3/4] Verifying regenerated report checksums"
if command -v sha256sum >/dev/null 2>&1; then
  (cd reports/stage4fr_final_20260721 && sha256sum -c SHA256SUMS.txt)
elif command -v shasum >/dev/null 2>&1; then
  (cd reports/stage4fr_final_20260721 && shasum -a 256 -c SHA256SUMS.txt)
else
  echo "Neither sha256sum nor shasum is installed" >&2
  exit 1
fi

echo "[4/4] Running the repository test suite"
python -m pytest -q

echo "GraphKAN Stage 4F-R repository verification: PASS"
