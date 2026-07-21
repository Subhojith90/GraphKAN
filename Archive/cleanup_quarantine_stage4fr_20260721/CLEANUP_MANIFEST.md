# Stage 4F-R Cleanup Quarantine — 2026-07-21

This directory contains generated handoff material moved out of the active
GraphKAN project during cleanup. Nothing here is part of the canonical
Stage 4F-R result registry.

## Preserved in the active project

- All source code, tests, configurations, documentation, and reports
- All `results_*` directories, including `results_stage4f_r/frozen`
- All historical release directories and release archives
- The validated final handoff ZIP and its SHA-256 sidecar

## Moved here

- Superseded Srinjoy handoff directories and archives from Stage 4E/4F setup
- The extracted Stage 4F-R final handoff directory, which duplicates the
  validated ZIP and canonical results

## Permanently removed as regenerable

- Active-project Python `__pycache__` directories (historical release snapshots
  were left byte-for-byte intact)
- `.pytest_cache`, matplotlib cache, `.DS_Store`
- Python package `*.egg-info` build metadata
- The project `tmp` directory and empty `artifacts`/`build` directories

Restore any moved item by moving it back to the active project root.
