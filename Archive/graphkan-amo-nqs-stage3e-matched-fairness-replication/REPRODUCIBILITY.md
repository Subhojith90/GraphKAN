# Reproducibility

The repository uses deterministic seeds, stable SHA-256 hashes for configs and source files, atomic JSON/CSV writes, and long-format result schemas. Optional dependencies are detected at runtime and unexecuted profiles are explicitly recorded in `results/raw/failed_runs.csv` or the release manifest.

The CPU path does not require GPU, NetKet, QuSpin, TeNPy, IBM credentials, or AMO hardware.
