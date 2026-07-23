# Stage 4F-R.1b release status

The Stage 4F-R scientific campaign is complete (960/960 registered runs). This repository reconstructs the release without modifying any canonical campaign payload.

## Closed locally

- Raw-byte verification of all 11,520 artifact-manifest payloads.
- Raw-byte verification of 960 completion records and 960 registry hashes.
- Exact reconstruction of the 105-file frozen execution-source ledger, including the original `scripts/stage4f1b_tx.py`.
- Exact equality between the release campaign ledger and `campaign_manifest.json#protocol_identity.source_ledger`.
- Separate post-campaign release-tool ledger and read-only frozen-campaign validation.
- Hash-pinned Python 3.12 Linux environment, commit/host attestation, and prospective five-seed tolerance table.
- Byte-preserving ZIP construction, sidecar generation, structural testing and clean-extraction verification.
- Current repository test collection and results recorded in release metadata.

## External gate closed

The revised 20-run replay executed through
`.github/workflows/stage4fr1b-independent-replay.yml` on a clean GitHub-hosted
Ubuntu 24.04 x86-64 runner. It covered both boundaries, two preregistered
cells, radial RBF, endpoint-full, endpoint-product, MLP-large, and RBM at seed
0.

All 20 scalar/full-vector/health/checkpoint/provenance comparisons and all four
periodic-collapse checks passed with zero blocking errors. The replay archive,
sidecar, machine-readable verification record, and closure report are retained
under `releases/` and `docs/STAGE4FR1B_INDEPENDENT_REPLAY_CLOSURE.md`.

Stage 4F-R.1b therefore closes the requested publication-grade release
integrity gate. Final public-release wording and manuscript claims remain
subject to supervisory approval.

No new scientific claim, hidden exclusion or protocol change is introduced by Stage 4F-R.1.
