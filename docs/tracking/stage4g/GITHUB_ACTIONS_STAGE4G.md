# Stage 4G on GitHub Actions

The recommended execution route is the manually dispatched workflow
`.github/workflows/stage4g-confirmation.yml`. The local Mac is not an execution
dependency after the dispatch commit is pushed.

## Workflow structure

1. `preflight` verifies the immutable dispatch commit, the complete frozen Stage
   4F campaign, the Stage 4G protocol/source ledger, the formal gauge gate and
   the real open/periodic integration smoke.
2. `convergence` runs eight Ubuntu shards: both boundaries times four models.
3. `convergence_gate` merges exactly 120 runs and 360 exact sentinels. A failed
   prospective convergence or exact-health comparison blocks later jobs.
4. `calibration` runs twelve disjoint calibration shards.
5. `select_controls` merges exactly 72 runs, applies the frozen within-family
   fractional-rank rule and emits a dynamic confirmation matrix.
6. `confirmation` runs twelve N=12 shards: both boundaries times six models.
7. `final_gate` merges exactly 180 runs, checks the full observables,
   checkpoints, periodic collapse, health and open-cell mechanism decision, and
   uploads the complete evidence tree for 90 days.

All third-party Actions are pinned to immutable commit SHAs. Python 3.12.7 and
the accepted Stage 4F-R.1b hash-locked Linux environment are reused. Shards use
separate output roots, so no concurrent job writes a shared CSV or checkpoint.

## Dispatch

After the scoped Stage 4G commit is on `main`:

1. Open `https://github.com/Subhojith90/GraphKAN/actions`.
2. Select **Stage 4G finite-size and gauge-fixed confirmation**.
3. Click **Run workflow**, select `main`, and confirm.

The Mac may then be closed or powered off. GitHub retains live logs and all
intermediate shard artifacts. Do not rerun individual jobs with altered source;
use GitHub's failed-job rerun on the same workflow run and immutable commit.

Expected elapsed time is approximately 3--6 hours when the account permits the
configured parallelism. Lower concurrency may increase elapsed time without
changing the scientific outputs.

## Required download

When `final_gate` is green, download
`stage4g-complete-evidence-<run_id>`. Preserve the Actions run URL, artifact
digest and dispatch commit. The final scientific authority inside the artifact
is `analysis/stage4g_final_decision.json`.
