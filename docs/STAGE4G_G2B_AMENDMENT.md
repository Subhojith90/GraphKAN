# Stage 4G G2b convergence amendment

## Status

This amendment was frozen on 4 August 2026, before executing G2b. It does not
replace or relabel the original G2 result.

## Why an amendment is required

The original Stage 4G convergence computation completed all 120 runs and 360
exact sentinels with 120/120 exact-health passes. The gate program then raised a
`KeyError` because the first sentinel schema did not carry `delta` and `c6`.
Reconstructing those identity fields read-only from the one-to-one metrics table
shows that the original frozen per-run 5% relative-change gate genuinely fails:
75/360 comparisons pass and 285/360 fail. The failure is retained in the new
workflow as predecessor evidence.

That rule is also unstable when an exact error approaches zero, because a small
absolute change can produce an arbitrarily large relative change. It is not
retrospectively relaxed. G2b instead uses a new, independently declared rule.

## G2b execution

- Start every trajectory afresh from its frozen seed.
- Run 4,200 VMC iterations.
- Record exact all-state sentinels at 2,800, 3,500, and 4,200 iterations.
- Preserve the same two boundaries, three cells, four models, and five seeds
  used in G2 (120 runs total).
- Split the work into 24 boundary/model/cell shards so no hosted job contains
  more than five trajectories.

## Frozen decision gates

For each boundary/cell/model group, take the median over five seeds and compare
the 3,500- and 4,200-step exact errors. All 72 group/outcome comparisons must
remain within the absolute limits in
`configs/stage4g_g2b_amendment.yaml`. Those limits equal 10% of the already
accepted Stage 4F open-chain endpoint-full/radial median effects and were not
estimated from the failed G2 trajectory.

The endpoint-full and endpoint-hybrid density-and-correlation repair must also
remain positive in at least two of three open cells at both late sentinels. All
periodic endpoint/radial exact-error comparisons must collapse exactly, and all
120 final runs must pass exact health.

Only a passing G2b gate permits control calibration and the N=12 confirmation
jobs to start. A failed G2b gate remains a valid scientific outcome and stops
the workflow.
