# Stage 4G — finite-size and gauge-fixed mechanism confirmation

Status: implementation freeze (3 August 2026). This document is prospective;
it does not report Stage 4G results.

## Scope

Stage 4G is the focused gate requested after acceptance of Stage 4F-R.1b. It
tests whether the open-boundary density/correlation repair survives formal
gauge fixing, matched additional optimisation, stronger tuned controls and a
small increase from N=10 to N=12. It is not a broad Graph-KAN, two-dimensional
or scaling campaign.

The machine-readable authority is `configs/stage4g_frozen.yaml`. The three
confirmation cells are (delta, C6) = (1.25, 3), (1.50, 4), and (2.00, 5).
Calibration uses the disjoint cells (1.25, 5) and (1.75, 3). These choices are
frozen before Stage 4G execution and cannot be changed after outcomes are seen.

## Gates

1. **G0 — repository and protocol.** Re-run the accepted Stage 4F verifier,
   validate the Stage 4G configuration, and write the protocol hash/rosters.
2. **G1 — formal gauge diagnostic.** Project the complete N=10 linear
   radial/endpoint parameterisation onto the minimum-Euclidean-norm
   representative of its full Boolean-state design matrix. Report numerical
   rank, null fraction, seed bands and observable linkage. This replaces the
   earlier curve-centring diagnostic; it does not rewrite checkpoints.
3. **G2 — convergence matching.** Re-run matched deterministic trajectories for
   the three linear primary models and the frozen Stage-4F-sized MLP control
   through 2100 iterations, recording exact all-state sentinels at 700, 1400
   and 2100 iterations. Starting from the same seed avoids checkpoint-resume
   optimiser-state ambiguity and makes the 700-step sentinel a reproduction.
4. **G3 — control calibration.** On two disjoint N=10 cells, compare the frozen
   MLP and RBM candidate grids using three seeds and all three co-primary exact
   errors. Select one member per family by the frozen fractional-rank rule.
5. **G4 — N=12 confirmation.** Run open and periodic boundaries, three frozen
   cells, five seeds, the radial/full/hybrid primary models, the selected MLP
   and RBM controls, and the three-parameter non-geometric capacity control.
6. **G5 — decision.** Endpoint repair must persist for density and connected
   correlation in at least two of three open cells without health collapse.
   Periodic endpoint-disabled models must remain exactly collapsed to radial.
   Energy, density and connected correlation remain co-primary; no architecture
   superiority claim is permitted.

## Capacity control

`nongeometric_b32_p3` adds exactly three trainable pair coefficients to the
32-term radial model, matching endpoint-full's linear pair-feature capacity.
Its three fixed pair-label features are produced from SHA-256 with seed
20260803, centred, orthogonalised and RMS-normalised. They never receive
positions, distances, endpoint coordinates or boundary labels. Thus it tests
extra capacity without encoding the proposed endpoint mechanism.

## Execution rule

Run each gate only after the preceding validation passes. Results belong under
`results_stage4g/`; no command may modify `results_stage4f_r/frozen`. A Stage 4G
manuscript claim is blocked until G5 emits a complete machine-readable decision.
