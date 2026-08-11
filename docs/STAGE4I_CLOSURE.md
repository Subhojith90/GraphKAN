# Stage 4I: finite-size failure-boundary and manuscript closure

Stage 4H is frozen as `NEGATIVE_CLOSURE_PASS`. Stage 4I does not attempt to
rescue Graph-KAN or establish architecture superiority. It closes repository
test parity, performs a checkpoint-only gauge/scale analysis, and runs one
span-preserving orthogonalised endpoint counterfactual at the preregistered
hard cell.

## Scientific design

- G0: explicit base and full-NQS test profiles with bounded transcripts,
  dependency locks, and clean-Git-extraction verification.
- G1: N=10 and N=12 checkpoint analysis using complete-Boolean-design
  minimum-norm gauge fixing, a uniform unordered-pair environment measure,
  weighted centring/RMS normalisation, seed alignment, confidence bands,
  conditioning spectra, and held-out-size physical-scale tests.
- G2: N=12 open `(delta, C6)=(2,5)`, five matched seeds, radial/raw
  endpoint-full/orthogonal endpoint-full, radial warm start, 6000 continuation
  steps. The orthogonal model has exactly the same sufficient-statistic column
  span as raw endpoint-full.
- G3: N=14/DMRG remains conditional and is not launched by these commands.

G1 failure is scientifically valid: it restricts the manuscript to numerical
seed stability and forbids physical-interpretability wording. G2 negative
closure is also a terminal pass when correlation remains worse in at least
four of five seeds.

## Local execution

All commands run locally. No GitHub Actions workflow is supplied.

```bash
cd "/Users/subhojithalder/Desktop/Research Papers/Graph Kan/GraphKAN_Publication_Validated"

JAX_ENABLE_X64=1 PYTHONPATH="$PWD/src:$PWD/scripts" \
  python -u scripts/run_stage4i_test_profiles.py \
  --output results_stage4i/test_profiles --profile all

JAX_ENABLE_X64=1 PYTHONPATH="$PWD/src:$PWD/scripts" \
  python -u scripts/run_stage4i_local.py prepare \
  --config configs/stage4i_frozen.yaml \
  --output results_stage4i/local_campaign_20260810

PYTHONPATH="$PWD/src:$PWD/scripts" caffeinate -dimsu \
  python -u scripts/run_stage4i_local.py run \
  --config configs/stage4i_frozen.yaml \
  --stage4g-root results_stage4g/final_local_20260807/confirmation/open \
  --output results_stage4i/local_campaign_20260810 \
  --max-workers 2

PYTHONPATH="$PWD/src:$PWD/scripts" \
  python scripts/run_stage4i_local.py finalize \
  --config configs/stage4i_frozen.yaml \
  --output results_stage4i/local_campaign_20260810
```

After committing the Stage 4I source, verify the exact clean Git extraction:

```bash
PYTHONPATH="$PWD/src:$PWD/scripts" \
  python scripts/verify_stage4i_release.py \
  --protocol results_stage4i/local_campaign_20260810/protocol \
  --output results_stage4i/test_profiles/clean_extraction_gate.json
```

Then package and test the evidence:

```bash
PYTHONPATH="$PWD/src:$PWD/scripts" \
  python scripts/package_stage4i.py \
  --campaign results_stage4i/local_campaign_20260810 \
  --output releases/GraphKAN_Stage4I_G0_G2_Evidence_Local_20260810.zip

cd releases
shasum -a 256 -c GraphKAN_Stage4I_G0_G2_Evidence_Local_20260810.zip.sha256
unzip -tq GraphKAN_Stage4I_G0_G2_Evidence_Local_20260810.zip
```

Interrupted campaigns are resumed by repeating the `run` command. Completed
run bundles and immutable checkpoints are never overwritten; failures remain
visible under each run's `attempts` directory.
