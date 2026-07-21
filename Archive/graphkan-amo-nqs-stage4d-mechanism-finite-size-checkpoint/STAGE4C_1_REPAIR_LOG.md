# Stage 4C.1 artifact patch

This patch makes the Stage 4C evidence release self-testing:
- release stores the repaired aggregator under `scripts/`, matching its release test;
- release includes `release_manifest.json` at its root;
- raw SHA-256 sidecars are regenerated relative to the release root;
- a `MANIFEST.sha256` covers release contents;
- terminal logs are copied when available.

It does not change frozen Stage 4C raw metrics or repaired results.
