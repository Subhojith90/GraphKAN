# Top Journal Readiness

## Current verdict

`NOT_READY`.

## Gate table

| gate                    | status                         | evidence                                                                                                                            |
|:------------------------|:-------------------------------|:------------------------------------------------------------------------------------------------------------------------------------|
| A_hamiltonian_validity  | PASS                           | Dense/sparse and analytic-limit tests passed in pytest and quick_ci.                                                                |
| B_kan_jastrow_viability | PILOT_INCONCLUSIVE_OR_NEGATIVE | KAN-Jastrow is not convincingly better than matched MLP-pair/smooth Jastrow in the CPU pilot; method-superiority claims are unsafe. |
| C_interpretability      | PILOT_PASS                     | Median pilot curve alignment = 0.999.                                                                                               |
| D_graphkan_value        | NOT_RUN                        | Graph-KAN profile is gated on KAN-Jastrow stability and was not executed.                                                           |
| E_top_journal_pursuit   | NOT_READY                      | Current artifact is a CPU pilot, not PRA/Quantum-grade final evidence.                                                              |

## PRA readiness

Not ready. Missing: NetKet/Flax VMC, stronger RBM/MLP-pair baselines, more seeds, larger Rydberg geometries, DMRG/QuSpin cross-checks where applicable, and final correlation/structure-factor evidence.

## PRR readiness

Not ready. The current artifact is a pilot benchmark, not a complete broad physics/computation study.

## MLST readiness

Not ready, but plausible after the benchmark API, documentation, and final software/results pipeline mature.

## CPC readiness

Not ready, but plausible if the reusable software package becomes the main contribution and includes robust documentation, examples, and performance characterisation.

## Quantum / PRX Quantum readiness

Not ready. The current evidence does not show a general NQS principle, state-of-the-art performance, broad quantum-information impact, or a new physical diagnostic.

## Strongest rejection argument

The present KAN-Jastrow result is not convincingly better than simpler physics-informed baselines, and Graph-KAN was not run.

## Evidence required to change the verdict

A larger, fair NetKet/Flax run with 5--10 seeds, strong baselines, exact/DMRG references, OOD transfer, and seed-stable gauge-fixed functions that correlate with physical observables.
