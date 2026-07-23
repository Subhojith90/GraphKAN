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

## Pending external gate

The revised 20-run replay must execute through `.github/workflows/stage4fr1b-independent-replay.yml` on a clean Ubuntu host. It covers both boundaries, two preregistered cells, radial RBF, endpoint-full, endpoint-product, MLP-large, and RBM at seed 0. Until its scalar, full-vector, health, checkpoint, provenance, and periodic-collapse records all report `PASS`, this release is an integrity-reconstruction candidate—not a publication-grade integrity GO.

No new scientific claim, hidden exclusion or protocol change is introduced by Stage 4F-R.1.
