# Stage 4B — N=8 Hard-Point Fairness, Gate-Definition, and Endpoint-Sum Control Sweep

## Scope

Stage 4B is the required audit-response package following the Stage 4 N=8 five-seed transfer diagnostic. It is intentionally **not** an N≥10, Graph-KAN, geometry-transfer, or manuscript package.

The purpose is to determine whether the RBF endpoint-sum pair ansatz remains non-dominated when compared at the same N=8 hard point against stronger MLP and smooth/RBF Jastrow controls under explicit sample and exact-health definitions.

## Frozen task

- Open Rydberg chain: `N=8`, spacing `1.0`
- Hamiltonian point: `Omega=1.0`, `Delta=2.0`, `C6=4.0`
- Seeds: `0,1,2,3,4`
- Exact all-state post-training energy, density, and connected correlation observables

## Three batch structure

1. `n8_component_replay`: radial / endpoint-sum / endpoint-difference / endpoint-product / endpoint-full, each with 21 parameters.
2. `n8_controls_core`: matched compact controls plus MLP-pair and endpoint-MLP controls at 21, 64, and 128 parameters.
3. `n8_capacity_compute`: radial and endpoint-sum RBF capacity sweep across basis sizes 8, 12, 16, 24. Larger settings use 8192 samples and SR shift 0.05, making the compute change explicit in raw metadata.

## Health policy

Stage 4B writes both `scientific_validity_sample` and `scientific_validity_exact`. The **headline and non-domination decision use exact all-state validity and exact observables**. The sampled gate remains in the outputs as a sampler-health diagnostic and cannot silently substitute for exact results.

## Checkpoints

Every completed run writes a compressed parameter-tree leaf archive (`.npz`) and JSON sidecar under `results_stage4b/<batch>/checkpoints/`. This is a portable inspection artifact; the JSON records the run ID, model family, parameter count, seed, and config hash.

## Stage 4B decision rule

Endpoint-sum passes only when it is not dominated by an available control under all exact metrics and a compute comparison. A control dominates endpoint-sum only if it is no worse on all exact metrics, strictly better on at least one, and operates within 1.25× endpoint-sum median runtime in the paired seed comparison.

No broad scale-up is allowed from this package alone. Stage 4C must be approved from the Stage 4B evidence archive.
