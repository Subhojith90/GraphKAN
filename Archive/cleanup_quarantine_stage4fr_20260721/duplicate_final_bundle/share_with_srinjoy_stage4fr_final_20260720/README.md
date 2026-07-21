# GraphKAN Stage 4F-R final supervisory handoff

## Current status

- Frozen campaign: **PASS**, 960/960 COMPLETE.
- Strict registry/filesystem/bundle validation: **PASS**.
- Frozen execution-source ledger: **PASS**, 105/105 hashes match.
- Four-phase SIGKILL microcampaign: **PASS**.
- Four-run replay: **technical PASS**, but recorded on the same Mac and therefore **not independent second-host evidence**.

## Requested supervisory decision

CONDITIONAL GO: accept Stage 4F.1c and Stage 4F-R campaign integrity; hold public release/next-stage escalation until second-host replay is supplied or waived by the supervisor.

## Primary entry points

- `report/GraphKAN_Stage4FR_Final_Validation_Handoff_20260720.pdf` - concise supervisory report.
- `evidence/final_gate_summary.json` - machine-readable gate status.
- `analysis/stage4fr_analysis_summary.json` - machine-readable science summary.
- `analysis/tables/` and `analysis/figures/` - registered-outcome outputs.
- `results_stage4f_r/frozen/` - complete canonical 960-run campaign and quarantined late attempts.
- `evidence/real_microcampaign_killtest/` - kill/recovery evidence.
- `evidence/replay_same_host/` - replay evidence with explicit provenance limitation.
- `reproducibility/` - frozen execution sources/configs plus post-processing script.

## Verification

```bash
sha256sum -c SHA256SUMS.txt
export PYTHONPATH="$PWD/reproducibility/src:$PWD/reproducibility/scripts"
python reproducibility/scripts/stage4f1c_campaign.py validate \
  --campaign results_stage4f_r/frozen \
  --strict-integrity --reconcile-registry --require-terminal --require-complete
```

## Claim boundary

This package supports completion and integrity of the registered Stage 4F-R N=10 benchmark. It does not claim generic KAN, endpoint-feature, Graph-KAN, or top-journal superiority. Descriptive rankings apply only to the frozen registered grid. Public release/next-stage escalation remains conditional on supervisory treatment of the second-host replay requirement.
