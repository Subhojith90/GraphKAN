# Software and literature update

Date: 20 June 2026.

This update was performed before coding. It uses official documentation and primary research sources where available. It does not create new scientific claims; it determines what can safely be implemented and what remains optional.

## Software stack

| Tool | Source | Status found | Required? | Implementation consequence |
|---|---|---:|---|---|
| NetKet | https://netket.readthedocs.io/ and https://www.netket.org/ | NetKet is a JAX-based machine-learning toolbox for many-body quantum systems and supports neural-network ansatzes through JAX/Flax-compatible frameworks. | Required for final production VMC; unavailable in this execution environment. | CPU artifact implements exact/full-sum engine and leaves NetKet modules as optional, explicit not-run paths. |
| JAX | https://docs.jax.dev/en/latest/installation.html | JAX installation requires `jax` and `jaxlib`; CPU installs with `pip install -U jax`, accelerator installs use extras such as CUDA wheels. | Required for final VMC route; JAX itself is installed here. | JAX-compatible design is retained; no NetKet/Flax VMC executed. |
| Flax | https://flax.readthedocs.io/ and https://flax-linen.readthedocs.io/ | Flax provides Linen/NNX neural network APIs for JAX. Linen remains usable, while NNX is encouraged for new users. | Required for final trainable KAN/Graph-KAN VMC models; unavailable here. | CPU artifact uses NumPy exact-basis feature models and marks Flax production models as not run. |
| Optax | https://optax.readthedocs.io/ | Optax is a gradient-processing and optimisation library for JAX. | Required for JAX model optimisation in final VMC; unavailable here. | CPU exact/full-sum uses SciPy L-BFGS-B only. |
| SciPy sparse ED | https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html | `scipy.sparse.linalg.eigsh` computes selected eigenpairs of real symmetric or complex Hermitian matrices using ARPACK, with convergence exceptions exposing partial results. | Required for CPU ED. | Implemented as the primary exact-reference path. |
| QuSpin | https://quspin.github.io/QuSpin/ | QuSpin is an open-source exact-diagonalisation and quantum-dynamics package for spin, boson and fermion many-body systems. | Optional cross-check; unavailable here. | Optional builder detects absence and records not-run status. |
| TeNPy | https://tenpy.readthedocs.io/ and https://tenpy.github.io/ | TeNPy is a tensor-network Python library for strongly correlated systems; DMRG is documented for MPS ground-state calculations. | Optional but strongly recommended for 1D final references; unavailable here. | Optional reference module detects absence and records not-run status. |
| Qiskit / IBM Quantum | https://quantum.cloud.ibm.com/docs/guides/plans-overview and https://quantum.cloud.ibm.com/docs/guides/v2-primitives | IBM Open Plan has limited runtime; Qiskit Runtime V2 primitives use Sampler V2 and Estimator V2. | Optional only. | IBM is not used for Rydberg NQS validation; optional toy sanity is not run. |

## Literature and anti-scoop update

| Work | Source | What it changes |
|---|---|---|
| SineKAN neural quantum states | https://arxiv.org/abs/2506.01891 and APS DOI page | KAN-NQS novelty is already gone. The project cannot claim first KAN neural quantum state. It must focus on AMO/Rydberg geometry, blockade scaling, gauge-fixed learned pair functions, and fair baselines. |
| Graph Kolmogorov-Arnold Networks / GKAN / GraphKAN | https://arxiv.org/abs/2406.06470 and https://arxiv.org/abs/2406.18354 | Graph-KAN novelty as a graph-learning architecture is also gone. The only defensible graph contribution is a wavefunction/AMO benchmark with fair Graph-MPNN comparison. |
| Patched transformers for Rydberg VMC | https://www.nature.com/articles/s42005-024-01584-y | Rydberg NQS has strong transformer baselines. This artifact cannot claim state-of-the-art accuracy or ignore transformer/RNN literature. |
| RNN wavefunctions for kagome Rydberg arrays | https://www.nature.com/articles/s42005-025-02226-7 and https://arxiv.org/abs/2405.20384 | Rydberg NQS is already a mature benchmark domain. The KAN contribution must be interpretability/transfer/failure-map oriented rather than accuracy-only. |
| KAN foundation | https://arxiv.org/abs/2404.19756 | KANs use learnable univariate functions on edges, but the theorem is not a performance guarantee. |
| KAN/MLP cautionary comparison | https://doi.org/10.1016/j.cma.2024.117290 | KANs can be slower or less robust. The decision gates must allow failure or negative-results benchmark outcomes. |

## Anti-scoop consequence

The v2 proposal's anti-scoop position remains correct. Generic KAN-NQS and Graph-KAN claims are unsafe. The surviving project is a reproducible Rydberg-array benchmark for blockade-normalised KAN-Jastrow and Graph-KAN models, with gauge-fixed learned-function diagnostics and fair baselines.

## Implementation plan consequence

The primary production route remains NetKet + JAX/Flax. The current environment does not provide NetKet or Flax, so the artifact implements a real CPU exact-reference and exact-fullsum engine, then records production VMC/DMRG profiles as not executed. This is preferable to faking NetKet or DMRG results.
