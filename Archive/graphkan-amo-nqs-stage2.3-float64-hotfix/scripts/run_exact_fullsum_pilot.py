#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

from _common import load_config, hash_dict, ensure_result_dirs, build_geometries_from_config, parameter_grid, run_model_on_instance, append_or_write
from graphkan_amo_nqs.hamiltonians.exact_diag import compute_rydberg_reference


def main():
    ap = argparse.ArgumentParser(description="Run exact full-basis CPU pilot for small Rydberg chains.")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config); profile = cfg.get("profile", "exact_fullsum_pilot"); config_hash = hash_dict(cfg)
    root = Path(cfg.get("output_root", "results")); ensure_result_dirs(root)
    all_metrics=[]; all_hist=[]; all_interp=[]
    for g in build_geometries_from_config(cfg):
        for params in parameter_grid(cfg):
            ref = compute_rydberg_reference(g, params)
            for model in cfg.get("models", ["product", "radial_jastrow", "mlp_pair", "kan_jastrow"]):
                for seed in cfg.get("training", {}).get("seeds", [0]):
                    r, h, it = run_model_on_instance(profile, g, params, model, int(seed), cfg, config_hash, reference=ref)
                    all_metrics.extend(r); all_hist.extend(h); all_interp.extend(it)
    append_or_write(root / "raw" / "run_metrics.csv", all_metrics)
    append_or_write(root / "raw" / "training_history.csv", all_hist)
    append_or_write(root / "interpretability" / "gauge_fixed_functions.csv", all_interp)
    print(f"exact_fullsum_pilot completed: {len(all_metrics)} metric rows")

if __name__ == "__main__":
    main()
