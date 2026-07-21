# Next Stage — NetKet/Flax Execution Plan

## Objective

Test whether trainable KAN-Jastrow offers either competitive physical accuracy or a measurable
interpretability advantage under a fair VMC protocol. The objective is not to prove that KANs
win generally.

## Non-negotiable protocol

1. Pin a dedicated environment and export a lock or `pip freeze` snapshot.
2. Run product, RBM, smooth Jastrow, trainable MLP-pair and trainable KAN-Jastrow.
3. Match parameter counts within a documented tolerance; report actual counts.
4. Use the same Markov-chain count, samples per iteration, optimiser family, learning-rate
   schedule, iteration cap, stopping rule and seed list.
5. Start with open Rydberg chains \(N=8,10,12,14,16\), five seeds minimum; increase to ten
   where compute permits.
6. Use exact diagonalisation at small sizes and QuSpin/TeNPy references where available.
7. Report energy density, connected correlations, density, structure factor, fidelity where
   exact states exist, convergence failures, runtime and sampling diagnostics.
8. Pre-register geometry-transfer and Graph-KAN only after the chain gate is non-negative.

## Current gate

The exact-reference benchmark does not pass the KAN viability gate. NetKet work is therefore a
confirmation/refinement stage, not a scale-up justified by a claimed KAN advantage.
