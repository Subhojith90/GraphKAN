# Stage 4B.1 artifact and gate repair incorporated into Stage 4C baseline

Implemented before Stage 4C execution:

1. **Runtime dominance convention corrected.** A control can dominate only when it is no worse on each exact metric, strictly better in at least one, and its runtime satisfies `control / candidate <= 1.25`. The old Stage 4B output used the inverse orientation in the dominance clause. The Stage 4B observed conclusion should be regenerated under this convention before external use.
2. **Alias handling corrected.** `smooth_endpoint_sum_21p` is treated as an implementation-equivalence check and is excluded from the Stage 4C independent control set.
3. **Release-specific tests added.** Trimmed release packages now ship `release_tests/` rather than inherited historical tests requiring omitted older-stage assets.
4. **VMC batch logging added for Stage 4C.** `scripts/run_stage4c_batch.sh` records command, timestamp, Python version, and complete terminal output into the results tree.

Historical Stage 4B terminal logs were not reconstructible from the evidence archive. They must not be fabricated. The release contains the existing clean replay and raw/checkpoint evidence; a literal Stage 4B trajectory replay would require re-running those batches.
