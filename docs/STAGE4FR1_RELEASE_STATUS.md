# Stage 4F-R.1 release status

The Stage 4F-R scientific campaign is complete (960/960 registered runs). This repository reconstructs the release without modifying any canonical campaign payload.

## Closed locally

- Raw-byte verification of all 11,520 artifact-manifest payloads.
- Raw-byte verification of 960 completion records and 960 registry hashes.
- Exact reconstruction of the 105-file frozen execution-source ledger, including the original `scripts/stage4f1b_tx.py`.
- Byte-preserving ZIP construction, sidecar generation, structural testing and clean-extraction verification.
- Current repository test collection and results recorded in release metadata.

## Pending external gate

The 16-run replay must execute through `.github/workflows/stage4fr1-independent-replay.yml` on a clean Ubuntu host. Until its validation and comparison records report `PASS`, this release is an integrity-reconstruction candidate—not a publication-grade integrity GO.

No new scientific claim, hidden exclusion or protocol change is introduced by Stage 4F-R.1.
