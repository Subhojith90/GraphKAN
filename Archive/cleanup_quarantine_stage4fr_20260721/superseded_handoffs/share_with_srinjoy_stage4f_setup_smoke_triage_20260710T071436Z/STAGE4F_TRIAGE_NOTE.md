# Stage 4F Setup, Smoke, and Triage Note

This bundle is not a completed Stage 4F physics result and is not a public-release artifact.

It contains the Stage 4F setup patch, README/environment cleanup, generated Stage 4F configs, smoke logs, partial open-boundary results, and triage evidence from a runtime stall during the full open-boundary run.

## Status

Stage 4E.1-R was approved as an internal artifact-repair checkpoint.

Stage 4F was initiated as a boundary-patch mechanism gate with embedded release-profile cleanup discipline.

## Smoke Results

The Stage 4F smoke gates completed:

- Open smoke all-models: exact_valid=14/16, sample_valid=14/16.
- Periodic smoke all-models: exact_valid=13/16, sample_valid=13/16.
- The periodic smoke reproduced endpoint-collapse behaviour for radial/endpoint RBF variants.
- RBM remained flagged on density/correlation gates.
- One MLP periodic seed was flagged on correlation.

## Partial Open Full Run

The monolithic open-boundary full run was interrupted after a runtime stall.

The partial open metrics contained 141 rows:

- Delta=1.25, C6=3 completed: 40/40 rows.
- Delta=1.25, C6=4 completed: 40/40 rows.
- Delta=1.25, C6=5 completed: 40/40 rows.
- Delta=1.50, C6=3 partially completed:
  - rbf_radial_b32: 5/5
  - endpoint_sum_b32: 5/5
  - endpoint_product_b32: 5/5
  - endpoint_hybrid_b32: 5/5
  - endpoint_full_b32: 1/5

The observed stall occurred around the open-boundary Delta=1.50, C6=3 endpoint_full_b32 continuation.

## Interpretation

This is an execution/runtime triage point, not a science failure.

The Stage 4F config and smoke gates are valid. The next implementation step should replace the monolithic full run with a stricter cell/model-level scheduler that prints flushed progress before each subprocess, uses shorter timeouts, captures exit codes per chunk, and quarantines slow endpoint_full/RBM rows without blocking endpoint-sum/product/hybrid evidence.

## Forbidden Claims

This bundle does not support:

- generic KAN superiority,
- Graph-KAN value,
- universal endpoint superiority,
- N >= 12 scale-up,
- manuscript readiness,
- public artifact release,
- completed Stage 4F.

