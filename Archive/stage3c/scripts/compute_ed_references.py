#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from _common import load_config, hash_dict, ensure_result_dirs, build_geometries_from_config, parameter_grid, metric_row, append_or_write
from graphkan_amo_nqs.hamiltonians import build_dense_rydberg, build_sparse_rydberg, analytic_diagonal_spectrum, RydbergParams
from graphkan_amo_nqs.hamiltonians.rydberg import instance_id
from graphkan_amo_nqs.hamiltonians.exact_diag import compute_rydberg_reference
from graphkan_amo_nqs.manifests import atomic_write_dataframe


def main():
    ap = argparse.ArgumentParser(description="Compute ED references for configured Rydberg instances.")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config); profile = cfg.get("profile", "ed_reference"); config_hash = hash_dict(cfg)
    root = Path(cfg.get("output_root", "results")); ensure_result_dirs(root)
    ref_rows = []; val_rows = []
    npz_payload = {}
    for g in build_geometries_from_config(cfg):
        for params in parameter_grid(cfg):
            inst = instance_id(g, params)
            ref = compute_rydberg_reference(g, params, dense_threshold=2**int(cfg.get("max_dense_n", 8)))
            for metric, value, units in [("reference_energy", ref.energy, "Omega"), ("density_mean", float(np.mean(ref.density)), "dimensionless"), ("connected_abs_mean", float(np.mean(np.abs(ref.connected))), "dimensionless")]:
                ref_rows.append(metric_row(profile, inst, g, params, "reference", "ED", -1, ref.method, metric, value, config_hash, units))
            npz_payload[f"{inst}_density"] = ref.density
            npz_payload[f"{inst}_connected"] = ref.connected
            if g.n_sites <= int(cfg.get("max_dense_n", 8)):
                Hd = build_dense_rydberg(g.positions, params)
                Hs = build_sparse_rydberg(g.positions, params).toarray()
                val_rows.append({"profile": profile, "instance_id": inst, "geometry": g.family, "n_sites": g.n_sites, "omega": params.omega, "delta": params.delta, "c6": params.c6, "metric": "dense_sparse_max_abs", "value": float(np.max(np.abs(Hd - Hs)))})
            if abs(params.omega) < 1e-14:
                spec = np.sort(np.linalg.eigvalsh(build_dense_rydberg(g.positions, params))) if g.n_sites <= int(cfg.get("max_dense_n", 8)) else analytic_diagonal_spectrum(g.positions, params)
                ana = analytic_diagonal_spectrum(g.positions, params)
                val_rows.append({"profile": profile, "instance_id": inst, "geometry": g.family, "n_sites": g.n_sites, "omega": params.omega, "delta": params.delta, "c6": params.c6, "metric": "omega0_spectrum_max_abs", "value": float(np.max(np.abs(spec - ana)))})
    atomic_write_dataframe(root / "references" / "ed_references.csv", pd.DataFrame(ref_rows))
    atomic_write_dataframe(root / "references" / "ed_validation.csv", pd.DataFrame(val_rows))
    np.savez_compressed(root / "references" / "observable_references.npz", **npz_payload)
    print(f"computed {len(ref_rows)} ED reference metrics for {len(npz_payload)//2} instances")

if __name__ == "__main__":
    main()
