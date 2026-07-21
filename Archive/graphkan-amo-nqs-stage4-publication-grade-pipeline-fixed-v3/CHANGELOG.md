# Changelog

## 0.1.0 - 2026-06-20

- Added deterministic Rydberg geometry and Hamiltonian engines.
- Added dense/sparse exact diagonalisation and analytic-limit validation.
- Added exact full-sum CPU variational pilots for product, radial Jastrow, MLP-pair random-feature and KAN-Jastrow models.
- Added gauge-fixed KAN radial-function extraction and seed-stability metrics.
- Added aggregation, figures, tables, claim validation, readiness checks and LaTeX report generation.

## Stage 2.5
- Added fixed occupation-parity phase to represent positive-transverse-field ground-state signs.
- Replaced shared product parameter with site-wise product fields.
- Added phase-aware RBM control and N=6 ED gate.

## Stage 2.6
- Explicitly locked `Spin(inverted_ordering=False)` and defined physical Rydberg occupation as `(1-sigma)/2`.
- Added a NetKet convention checker, exact environment pin file, pip-freeze capture, and ED density/correlation diagnostics.
- Added a frozen-budget N=6,8 observable grid and valid-only paired aggregation.

## Stage 3B
- Added `kan_endpoint_jastrow`, a reflection-symmetric endpoint-distance pair correction.
- Added a six-run, 19-parameter matched gate at the Stage 3A hard instance.
- Stage 3B explicitly tests edge-edge / edge-bulk / bulk-bulk distinction; it is not Graph-KAN.
