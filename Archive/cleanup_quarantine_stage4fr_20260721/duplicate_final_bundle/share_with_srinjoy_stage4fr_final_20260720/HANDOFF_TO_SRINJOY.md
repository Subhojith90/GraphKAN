# Student handoff for Srinjoy - Stage 4F-R

Date: 20 July 2026

## Short update

Every Stage 4F.1c structural gate and the full frozen Stage 4F-R campaign pass local validation. The campaign is terminal at 960/960 COMPLETE, strict reconciliation checks all 960 canonical bundles, and the 105-entry frozen source ledger has zero drift. The four planned SIGKILL/recovery phases pass and the 16-run real microcampaign is 16/16 COMPLETE.

The four-run replay is technically clean but was executed on the same Mac. I am explicitly reporting that as an open requirement, not as independent-host evidence.

## Known limitations

1. Genuine second-host replay has not been demonstrated in the supplied provenance.
2. Full worker-shell transcripts were not centrally captured with `tee`; canonical per-run `command.json`, `stdout.log`, `stderr.log`, `exit_status.json`, registry events, and quarantined attempts are included instead.
3. Scientific tables are restricted to the frozen N=10 roster and preregistered exact energy, density, and connected-correlation outcomes.
4. No generic method-superiority or top-journal-readiness claim is made.

## Decision requested

CONDITIONAL GO: accept Stage 4F.1c and Stage 4F-R campaign integrity; hold public release/next-stage escalation until second-host replay is supplied or waived by the supervisor.
