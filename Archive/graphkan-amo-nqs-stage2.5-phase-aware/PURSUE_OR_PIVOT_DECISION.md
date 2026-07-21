# Pursue or Pivot Decision

## Is the project worth pursuing further?

Yes, but only as a disciplined benchmark and interpretability project. The Hamiltonian and exact-reference path is now real enough to support continued development. The KAN method claim is not yet supported.

## Is KAN-Jastrow viable?

Current status: `PILOT_INCONCLUSIVE_OR_NEGATIVE`.

The exact-fullsum pilot is technically healthy, and gauge-fixed curves are stable. However, in the reduced Rydberg chain CPU pilot, KAN-Jastrow has higher median energy-density and correlation errors than matched MLP-pair and smooth Jastrow. KAN-Jastrow should continue only if the next NetKet/Flax VMC and larger exact-fullsum experiments show either comparable accuracy with stronger interpretability, lower correlation error, or better OOD transfer.

## Is Graph-KAN worth continuing?

Not yet. Graph-KAN should remain gated. Implement and run it only after:

1. KAN-Jastrow is stable under a larger pilot;
2. a graph MPNN baseline with the same features is implemented;
3. geometry transfer splits are locked;
4. compute budgets are matched.

## Is the paper likely PRA-worthy now?

No. A PRA paper requires more Rydberg physics observables, stronger baselines, more seeds, and at least one credible physics/interpretable-function result.

## Is MLST/CPC more realistic?

Eventually, yes, if the software benchmark becomes robust. At the current stage it is still a pilot artifact.

## Is this becoming a negative-results benchmark?

Possibly. If larger, fairer runs continue to show that KAN-Jastrow is matched or dominated by MLP-pair and smooth Jastrow, the honest route is a negative-results benchmark: when interpretable KAN-Jastrow NQS does and does not help for Rydberg arrays.

## What would make Quantum or PRX Quantum plausible?

A broad, reproducible result beyond architecture comparison: e.g. a gauge-fixed learned-function diagnostic that reliably tracks blockade/correlation scales across geometries and gives useful physical insight unavailable from black-box baselines, or a strong OOD transfer principle validated against strong NQS baselines.

## Next actions

1. Install NetKet/Flax/Optax in a Python 3.10--3.12 environment.
2. Re-run the Rydberg pilot with 5--10 seeds and trainable Flax models.
3. Add RBM, MLP-pair, smooth Jastrow, and graph MPNN baselines with frozen fairness budgets.
4. Extend exact references to the largest feasible ED sizes and add QuSpin cross-checks.
5. Run geometry transfer only after the chain pilot passes.
6. Keep Graph-KAN and Bose-Hubbard optional until the Rydberg KAN-Jastrow gate improves.
