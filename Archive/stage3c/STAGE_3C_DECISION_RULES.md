# Stage 3C decision rules
1. **Taxonomy gate:** Stage 3B.1 must report sine for both radial and endpoint KAN.
2. **Replay gate:** corrected Stage 3B.1 must retain endpoint density improvement relative to radial KAN.
3. **Ablation gate:** at least one endpoint feature must lower median exact density MAE relative to radial-only sine KAN, and the full endpoint model must not be materially worse in exact energy (>20%).
4. **N=8 Batch A gate:** endpoint sine KAN must not show systematic density-gate failure across two seeds and must not be dominated simultaneously by RBF, MLP-pair, and endpoint-aware MLP across energy, density, correlation, and runtime.
5. **Replication gate:** only then run five-seed N=8 configuration.

Failure of any gate produces a diagnostic/failure-map outcome, not scale-up.
