#!/usr/bin/env python
"""Run a registered exact-reference benchmark suite without optional NQS dependencies.

This script deliberately keeps the scope narrow: each variational ansatz is optimized
against the same small-system Rydberg Hamiltonian using exact full-basis sums.  It is
not a VMC claim; it is a reproducible screening and validation tier before NetKet.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from _common import (
    append_or_write,
    build_geometries_from_config,
    ensure_result_dirs,
    hash_dict,
    load_config,
    parameter_grid,
    run_model_on_instance,
)
from graphkan_amo_nqs.hamiltonians.exact_diag import compute_rydberg_reference
from graphkan_amo_nqs.manifests import atomic_write_json


def main() -> None:
    ap = argparse.ArgumentParser(description="Run registered exact-reference Rydberg benchmark.")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    profile = cfg.get("profile", "exact_reference_benchmark")
    config_hash = hash_dict(cfg)
    root = Path(cfg.get("output_root", "results"))
    ensure_result_dirs(root)

    # Immutable benchmark declaration stored alongside numerical outputs.
    declaration = {
        "profile": profile,
        "config_hash": config_hash,
        "scope": "exact full-basis variational benchmark; not VMC",
        "models": cfg.get("models", []),
        "seeds": cfg.get("training", {}).get("seeds", []),
        "geometries": cfg.get("chains", cfg.get("geometry", {})),
        "parameters": cfg.get("parameters", cfg.get("hamiltonian", {})),
        "fairness": {
            "same_instances": True,
            "same_optimizer": "SciPy L-BFGS-B exact full-basis objective",
            "same_maxiter": cfg.get("training", {}).get("maxiter"),
            "same_seed_manifest": True,
        },
    }
    atomic_write_json(root / "raw" / "benchmark_declaration.json", declaration)

    all_metrics: list[dict] = []
    all_history: list[dict] = []
    all_interp: list[dict] = []
    total = 0
    for geometry in build_geometries_from_config(cfg):
        for params in parameter_grid(cfg):
            reference = compute_rydberg_reference(geometry, params)
            for model in cfg.get("models", ["product", "smooth_jastrow", "mlp_pair", "kan_jastrow"]):
                for seed in cfg.get("training", {}).get("seeds", [0, 1, 2, 3, 4]):
                    rows, history, interp = run_model_on_instance(
                        profile, geometry, params, model, int(seed), cfg, config_hash, reference=reference
                    )
                    all_metrics.extend(rows)
                    all_history.extend(history)
                    all_interp.extend(interp)
                    total += 1
                    print(f"completed {total}: N={geometry.n_sites} delta={params.delta:g} c6={params.c6:g} {model} seed={seed}")

    append_or_write(root / "raw" / "run_metrics.csv", all_metrics)
    append_or_write(root / "raw" / "training_history.csv", all_history)
    append_or_write(root / "interpretability" / "gauge_fixed_functions.csv", all_interp)
    print(f"{profile} completed: {total} matched optimisation runs, {len(all_metrics)} metric rows")


if __name__ == "__main__":
    main()
