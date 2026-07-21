# Stage 4B Changelog

- Replaced legacy Stage 3F/Stage 4 raw output naming with native `stage4b_*` schemas.
- Added separate sampled-health and exact all-state health classifications.
- Made exact validity the headline scientific-gate basis.
- Added native Stage 4B checkpoint `.npz` files with JSON metadata sidecars.
- Added RBF endpoint component replay at N=8.
- Added 21p, 64p, and 128p MLP-pair / endpoint-MLP control configurations.
- Added radial and endpoint-sum smooth/RBF Jastrow controls.
- Added RBF radial vs endpoint-sum capacity/compute sweep.
- Removed legacy singular gate summarizers from the runnable Stage 4B path.
- Added Stage 4B schema/asset tests and release packaging validation.

## v2 hotfix
- Corrected the NetKet sampler construction in `run_stage4b_controls.py`: `MetropolisLocal` now receives the Hilbert space (`hi`), not the `LocalOperatorJax` Hamiltonian. This fixes the runtime `AttributeError: LocalOperatorJax has no attribute _size` observed in v1 before any VMC trajectory began.
- Added explicit RBF basis-family dispatch for `smooth_endpoint_sum_jastrow`.
- Added static regression tests for both fixes.
