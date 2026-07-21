# Stage 3E changes
- Added true 19-parameter matched MLP-pair control (`n_basis=10`, `hidden=1`).
- Added true 19-parameter matched endpoint-aware MLP control (`n_basis=7`, `hidden=1`).
- Removed the duplicated `rbf_jastrow_19p` from the core factorial set.
- Preserved explicit basis taxonomy and immediate per-run checkpoints.
- Added parameter-match tests and a summary-time hard stop.
- Added two-seed checkpoint and five-seed replication configs.
