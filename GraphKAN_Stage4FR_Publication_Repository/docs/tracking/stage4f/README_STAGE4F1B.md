# Stage 4F.1b Semantic Completion, Lease Recovery, and Real-Runner Integration Gate

> Superseded by `README_STAGE4F1C.md`. This file describes the historical 4F.1b
> submission and must not be used as evidence that the 4F.1c gate has passed.

This package is an engineering hardening package. It is not a completed Stage 4F-R physics result.

It addresses the Stage 4F.1 audit findings by adding:

- fail-closed semantic bundle validation;
- artifact manifest hash verification;
- checkpoint payload existence and hash checks;
- run-id consistency checks;
- finite CSV/schema checks;
- COMPLETE.json written-last ordering checks;
- immutable protocol-hash campaign identity;
- no-overwrite atomic promotion;
- stale lease recovery;
- bounded retry/quarantine scaffolding;
- balanced deterministic priority ordering;
- Stage 4F.1b configs and tests.

Local gates passed:

- Stage 4F.1b campaign init registered 960 expected runs with zero duplicate registry keys.
- Semantic adversarial validator rejected tampered metrics, false artifact hashes, missing checkpoint payloads, corrupt checkpoint metadata, unreadable checkpoint metadata, wrong run IDs, and stale completion-order cases.
- Lease recovery test passed.
- Release tests passed 4/4.
- Dry-run transactional worker wrote two COMPLETE bundles.
- Campaign validation found zero invalid complete bundles.

The real VMC path remains gated and should be enabled only after this hardening package is accepted.

Stage 4F-R remains blocked until the real 16-run transactional microcampaign and independent replay pass.
