# Stage 4D.1–4E.1 Audit Bundle

## Scope
This bundle contains the controlled evidence for:
1. Stage 4D.1 provenance and quarantine classification.
2. Stage 4E N=10 capacity sweep.
3. Stage 4E.1 open-versus-periodic boundary diagnostic.

## Registered Stage 4E N=10 Cell
- Geometry: N=10 Rydberg chain
- Parameters: Omega=1, Delta=1.5, C6=4
- Seeds: 0–4
- Training: 4096 samples, fixed registered protocol

## Stage 4E Capacity Finding
At the registered N=10 cell, 35–43 parameter endpoint-aware RBF models
form a density-oriented trade-off against larger endpoint MLP controls:
MLPs retain the stronger energy/correlation metrics, while selected endpoint
models improve density reconstruction.

## Stage 4E.1 Boundary Finding
Open and periodic paired batches each completed 30/30 exact-valid and
sample-valid trajectories.

On the open chain, endpoint-aware models improve over same-basis radial
controls. Under periodic boundaries, endpoint-sum and endpoint-product
models collapse exactly to their matched radial forms because endpoint
features are disabled: identical parameter counts and zero differences in
exact energy, density, and correlation across all paired seeds.

## Claim Boundary
This is boundary-conditioned mechanism evidence at one registered N=10
cell. It does not establish universal KAN superiority, universal endpoint
superiority, or general superiority to larger MLP controls.
