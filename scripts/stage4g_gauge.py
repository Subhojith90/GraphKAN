#!/usr/bin/env python3
"""Formal minimum-norm gauge analysis of frozen Stage 4F linear checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from graphkan_amo_nqs.models.flax_nqs import pair_feature_arrays


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("rbf_radial_b32", "endpoint_full_b32", "endpoint_hybrid_b32")
CELLS = ((1.25, 3.0), (1.50, 4.0), (2.00, 5.0))


def minimum_norm_gauge(design: np.ndarray, theta: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Project theta onto row(design), preserving every represented amplitude."""
    design = np.asarray(design, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    if design.ndim != 2 or theta.shape != (design.shape[1],):
        raise ValueError("incompatible design matrix and parameter vector")
    _, singular, vh = np.linalg.svd(design, full_matrices=False)
    tolerance = max(design.shape) * np.finfo(np.float64).eps * (singular[0] if singular.size else 0.0)
    rank = int(np.sum(singular > tolerance))
    row_basis = vh[:rank]
    projected = row_basis.T @ (row_basis @ theta)
    before = design @ theta
    after = design @ projected
    theta_norm = float(np.linalg.norm(theta))
    return projected, {
        "design_rows": int(design.shape[0]), "design_columns": int(design.shape[1]),
        "rank": rank, "nullity": int(design.shape[1] - rank), "svd_tolerance": float(tolerance),
        "max_log_amplitude_difference": float(np.max(np.abs(before - after))),
        "raw_parameter_norm": theta_norm,
        "null_parameter_norm": float(np.linalg.norm(theta - projected)),
        "null_fraction": float(np.linalg.norm(theta - projected) / max(theta_norm, 1.0e-300)),
    }


def boolean_states(n_sites: int) -> np.ndarray:
    integers = np.arange(1 << n_sites, dtype=np.uint64)[:, None]
    return ((integers >> np.arange(n_sites, dtype=np.uint64)) & 1).astype(np.float64)


def endpoint_features(n_sites: int, pair_i: np.ndarray, pair_j: np.ndarray, model: str) -> np.ndarray:
    d = np.minimum(np.arange(n_sites), n_sites - 1 - np.arange(n_sites)).astype(float)
    d /= max(float(d.max()), 1.0)
    full = np.stack([d[pair_i] + d[pair_j], np.abs(d[pair_i] - d[pair_j]), d[pair_i] * d[pair_j]], axis=1)
    if model == "endpoint_hybrid_b32":
        return full[:, [0, 2]]
    if model == "endpoint_full_b32":
        return full
    return np.empty((len(pair_i), 0), dtype=float)


def checkpoint_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_linear_theta(path: Path, model: str) -> tuple[np.ndarray, int, int]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = [np.asarray(archive[name], dtype=float) for name in sorted(archive.files)]
    if model == "rbf_radial_b32" and [a.shape for a in arrays] == [(32,), (10,)]:
        radial, site = arrays
        endpoint = np.empty((0,), dtype=float)
    elif model == "endpoint_full_b32" and [a.shape for a in arrays] == [(3,), (32,), (10,)]:
        endpoint, radial, site = arrays
    elif model == "endpoint_hybrid_b32" and [a.shape for a in arrays] == [(2,), (32,), (10,)]:
        endpoint, radial, site = arrays
    else:
        raise ValueError(f"unexpected checkpoint leaf layout for {model}: {[a.shape for a in arrays]}")
    return np.concatenate([site, radial, endpoint]), len(site), len(radial)


def find_bundle(campaign: Path, boundary: str, delta: float, c6: float, model: str, seed: int) -> Path:
    d = f"{delta:.2f}".replace(".", "p")
    c = f"{c6:g}".replace(".", "p")
    matches = list((campaign / "runs").glob(f"*__B{boundary}__D{d}__C{c}__M{model}__S{seed}"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one bundle, found {len(matches)} for {boundary}/{d}/{c}/{model}/S{seed}")
    return matches[0]


def analyse(campaign: Path, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    states = boolean_states(10)
    pair_i, pair_j, basis = pair_feature_arrays(np.arange(10, dtype=float)[:, None], n_basis=32, family="rbf")
    pair_occ = states[:, pair_i] * states[:, pair_j]
    records: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, Any]] = []
    for delta, c6 in CELLS:
        for seed in range(5):
            for model in MODELS:
                bundle = find_bundle(campaign, "open", delta, c6, model, seed)
                checkpoint = bundle / "checkpoint.npz"
                manifest = json.loads((bundle / "checkpoint_manifest.json").read_text())
                if checkpoint_sha256(checkpoint) != manifest["sha256"]:
                    raise RuntimeError(f"checkpoint hash mismatch: {bundle.name}")
                theta, n_site, n_radial = load_linear_theta(checkpoint, model)
                endpoint = endpoint_features(10, pair_i, pair_j, model)
                pair_design = np.concatenate([basis, endpoint], axis=1)
                design = np.concatenate([states, pair_occ @ pair_design], axis=1)
                gauged, diagnostic = minimum_norm_gauge(design, theta)
                pair_coeff = gauged[n_site:]
                edge = pair_design @ pair_coeff
                record = {
                    "run_id": manifest["run_id"], "delta": delta, "c6": c6, "model": model, "seed": seed,
                    "checkpoint_sha256": manifest["sha256"], "gauge_definition": "full-Boolean-design minimum-Euclidean-norm",
                    "gauge_pair_norm": float(np.linalg.norm(edge)),
                    "gauge_endpoint_coefficient_norm": float(np.linalg.norm(gauged[n_site + n_radial:])),
                    **diagnostic,
                }
                records.append(record)
                for index, value in enumerate(edge):
                    curves.append({
                        "run_id": manifest["run_id"], "delta": delta, "c6": c6, "model": model, "seed": seed,
                        "pair_i": int(pair_i[index]), "pair_j": int(pair_j[index]), "gauge_pair_value": float(value),
                    })
                metric = pd.read_csv(bundle / "metrics.csv").iloc[0].to_dict()
                metrics_rows.append({
                    "delta": delta, "c6": c6, "model": model, "seed": seed,
                    **{name: float(metric[name]) for name in (
                        "exact_model_abs_energy_error_to_ed", "exact_model_density_mae_to_ed",
                        "exact_model_connected_correlation_mae_to_ed")},
                })
    diagnostics = pd.DataFrame(records)
    metric_df = pd.DataFrame(metrics_rows)
    outcomes = ["exact_model_density_mae_to_ed", "exact_model_connected_correlation_mae_to_ed"]
    radial = metric_df[metric_df.model == "rbf_radial_b32"].drop(columns="model")
    endpoint_values = metric_df[metric_df.model != "rbf_radial_b32"]
    linked = diagnostics[diagnostics.model != "rbf_radial_b32"].merge(
        radial, on=["delta", "c6", "seed"]
    ).merge(
        endpoint_values, on=["delta", "c6", "seed", "model"], suffixes=("_radial", "_endpoint")
    )
    for outcome in outcomes:
        linked[f"repair_{outcome}"] = linked[f"{outcome}_radial"] - linked[f"{outcome}_endpoint"]
    linkage = []
    for model, group in linked.groupby("model"):
        for outcome in ("exact_model_density_mae_to_ed", "exact_model_connected_correlation_mae_to_ed"):
            repair = f"repair_{outcome}"
            linkage.append({
                "model": model, "repair_outcome": outcome, "n": len(group),
                "spearman_gauge_endpoint_norm_vs_repair": float(group["gauge_endpoint_coefficient_norm"].corr(group[repair], method="spearman")),
                "pearson_gauge_endpoint_norm_vs_repair": float(group["gauge_endpoint_coefficient_norm"].corr(group[repair], method="pearson")),
            })
    diagnostics.to_csv(output / "gauge_diagnostics.csv", index=False)
    pd.DataFrame(curves).to_csv(output / "gauge_pair_functions.csv", index=False)
    pd.DataFrame(linkage).to_csv(output / "gauge_observable_linkage.csv", index=False)
    summary = {
        "schema_version": "stage4g-formal-gauge-v1", "validation": "PASS",
        "gauge_definition": "minimum Euclidean norm in the complete N=10 Boolean-state linear design",
        "checkpoints_checked": len(diagnostics), "expected_checkpoints": 45,
        "maximum_amplitude_preservation_error": float(diagnostics.max_log_amplitude_difference.max()),
        "models": list(MODELS), "cells": [{"delta": d, "c6": c} for d, c in CELLS],
        "observable_linkage": linkage,
        "interpretation_limit": "association diagnostic; not a causal identification of learned functions",
    }
    (output / "gauge_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, default=ROOT / "results_stage4f_r" / "frozen")
    parser.add_argument("--output", type=Path, default=ROOT / "results_stage4g" / "gauge")
    args = parser.parse_args()
    result = analyse(args.campaign, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
