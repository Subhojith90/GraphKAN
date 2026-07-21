#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from _common import load_config, hash_dict, ensure_result_dirs, build_geometries_from_config, parameter_grid, run_model_on_instance, append_or_write, metric_row
from graphkan_amo_nqs.hamiltonians import build_dense_rydberg, build_sparse_rydberg, analytic_diagonal_spectrum, RydbergParams, enumerate_binary_states
from graphkan_amo_nqs.hamiltonians.rydberg import instance_id
from graphkan_amo_nqs.hamiltonians.exact_diag import compute_rydberg_reference
from graphkan_amo_nqs.manifests import atomic_write_dataframe


def main():
    ap = argparse.ArgumentParser(description="Run CPU quick CI for geometry, Hamiltonian and exact-fullsum KAN pilot.")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config); profile = cfg.get("profile", "quick_ci"); config_hash = hash_dict(cfg)
    root = Path(cfg.get("output_root", "results")); ensure_result_dirs(root)
    geoms = build_geometries_from_config(cfg); params = next(parameter_grid(cfg)); g = geoms[0]
    rows = []
    H_dense = build_dense_rydberg(g.positions, params)
    H_sparse = build_sparse_rydberg(g.positions, params)
    rows.append(metric_row(profile, instance_id(g, params), g, params, "quick", "hamiltonian", 0, "dense_sparse", "dense_sparse_max_abs", float(np.max(np.abs(H_dense - H_sparse.toarray()))), config_hash, "abs"))
    p0 = RydbergParams(omega=0.0, delta=params.delta, c6=params.c6, sign_rule=params.sign_rule)
    spec_dense = np.sort(np.linalg.eigvalsh(build_dense_rydberg(g.positions, p0)))
    spec_analytic = analytic_diagonal_spectrum(g.positions, p0)
    rows.append(metric_row(profile, instance_id(g, p0), g, p0, "quick", "hamiltonian", 0, "analytic", "omega0_spectrum_max_abs", float(np.max(np.abs(spec_dense - spec_analytic))), config_hash, "abs"))
    ref = compute_rydberg_reference(g, params)
    for model in cfg.get("models", ["product", "kan_jastrow"]):
        r, h, it = run_model_on_instance(profile, g, params, model, 0, cfg, config_hash, reference=ref)
        rows.extend(r); append_or_write(root / "raw" / "training_history.csv", h); append_or_write(root / "interpretability" / "gauge_fixed_functions.csv", it)
    append_or_write(root / "raw" / "run_metrics.csv", rows)
    print(f"quick_ci completed with {len(rows)} metric rows")

if __name__ == "__main__":
    main()
