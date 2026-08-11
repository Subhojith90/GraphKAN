#!/usr/bin/env python3
"""Deterministic checkpoint-only Stage 4I gauge and scale-linkage gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from graphkan_amo_nqs.models.stage4g_flax_nqs import pair_feature_arrays
from graphkan_amo_nqs.training.stage4i_orthogonal import (
    boolean_states,
    endpoint_pair_features,
)
from stage4i_protocol import DEFAULT_CONFIG, read_config, validate_config

MODEL_N10 = {
    "radial": "rbf_radial_b32",
    "endpoint_full": "endpoint_full_b32",
    "endpoint_hybrid": "endpoint_hybrid_b32",
}
ARM_N12 = {
    "radial": "radial_continuation",
    "endpoint_full": "endpoint_full_warmstart",
    "endpoint_hybrid": "endpoint_hybrid_warmstart",
}


def design(
    n: int, c6: float, arm: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    states = boolean_states(n)
    i, j, radial = pair_feature_arrays(
        np.arange(n, dtype=float)[:, None],
        n_basis=32,
        family="rbf",
        r_scale=float(c6) ** (1 / 6),
    )
    pair_occ = states[:, i] * states[:, j]
    extra = endpoint_pair_features(n, i, j)
    if arm == "endpoint_hybrid":
        extra = extra[:, [0, 2]]
    pair = radial if arm == "radial" else np.concatenate([radial, extra], axis=1)
    return np.concatenate([states, pair_occ @ pair], axis=1), pair, i, j


def n10_dir(root: Path, cell: dict[str, Any], model: str, seed: int) -> Path:
    delta = f"D{float(cell['delta']):.2f}".replace(".", "p")
    token = f"__Bopen__{delta}__C{int(cell['c6'])}__M{model}__S{seed}"
    found = [p for p in (root / "runs").iterdir() if token in p.name]
    if len(found) != 1:
        raise FileNotFoundError(f"N10 run lookup {token}: {len(found)} matches")
    return found[0]


def n12_dir(root: Path, cell: dict[str, Any], arm: str, seed: int) -> Path:
    run = root / "runs" / f"stage4h__{cell['tag']}__{ARM_N12[arm]}__S{seed}"
    completion = json.loads((run / "completion.json").read_text())
    return run / "attempts" / completion["attempt"]


def load_record(
    n: int, root: Path, cell: dict[str, Any], arm: str, seed: int
) -> dict[str, Any]:
    if n == 10:
        run = n10_dir(root, cell, MODEL_N10[arm], seed)
        with np.load(run / "checkpoint.npz", allow_pickle=False) as z:
            arrays = [np.asarray(z[k], float) for k in sorted(z.files)]
        if arm == "radial":
            params = {"kan_coeff": arrays[0], "site_bias": arrays[1]}
        else:
            params = {
                "endpoint_coeff": arrays[0],
                "kan_coeff": arrays[1],
                "site_bias": arrays[2],
            }
        site, corr = (
            pd.read_csv(run / "site_observables.csv"),
            pd.read_csv(run / "corr_observables.csv"),
        )
        run_id = run.name
    else:
        run = n12_dir(root, cell, arm, seed)
        with np.load(run / "checkpoints/step-06000.npz", allow_pickle=False) as z:
            params = {k: np.asarray(z[k], float) for k in z.files}
        site, corr = (
            pd.read_csv(run / "site_observables.csv"),
            pd.read_csv(run / "correlation_observables.csv"),
        )
        run_id = json.loads((run / "metrics.json").read_text())["run_id"]
    return {"run_id": run_id, "params": params, "site": site, "corr": corr}


def gauge_one(
    n: int,
    cell: dict[str, Any],
    arm: str,
    seed: int,
    record: dict[str, Any],
    matrix: np.ndarray,
    pair: np.ndarray,
    i: np.ndarray,
    j: np.ndarray,
    row_basis: np.ndarray,
    tolerance: float,
) -> tuple[dict, list[dict]]:
    p = record["params"]
    theta = np.concatenate(
        [p["site_bias"], p["kan_coeff"]]
        + ([] if arm == "radial" else [p["endpoint_coeff"]])
    )
    gauged = row_basis.T @ (row_basis @ theta)
    before, after = matrix @ theta, matrix @ gauged
    theta_norm = float(np.linalg.norm(theta))
    diagnostic = {
        "design_rows": int(matrix.shape[0]),
        "design_columns": int(matrix.shape[1]),
        "rank": int(len(row_basis)),
        "nullity": int(matrix.shape[1] - len(row_basis)),
        "svd_tolerance": float(tolerance),
        "max_log_amplitude_difference": float(np.max(np.abs(before - after))),
        "raw_parameter_norm": theta_norm,
        "null_parameter_norm": float(np.linalg.norm(theta - gauged)),
        "null_fraction": float(
            np.linalg.norm(theta - gauged) / max(theta_norm, 1e-300)
        ),
    }
    one_body = gauged[:n]
    pair_value = pair @ gauged[n:]
    pair_value = pair_value - pair_value.mean()
    rms = float(np.sqrt(np.mean(pair_value**2)))
    normalised = pair_value / max(rms, 1e-300)
    separation = j - i
    landmark = float(
        np.sum(separation * np.abs(normalised)) / np.sum(np.abs(normalised))
    )
    summary = {
        "run_id": record["run_id"],
        "n_sites": n,
        "cell_tag": cell["tag"],
        "delta": cell["delta"],
        "c6": cell["c6"],
        "arm": arm,
        "seed": seed,
        "one_body_rms": float(np.sqrt(np.mean(one_body**2))),
        "pair_rms_before_normalisation": rms,
        "curve_landmark_separation": landmark,
        **diagnostic,
    }
    rows = [
        {
            **{
                k: summary[k]
                for k in ("run_id", "n_sites", "cell_tag", "delta", "c6", "arm", "seed")
            },
            "pair_i": int(i[x]),
            "pair_j": int(j[x]),
            "separation": int(separation[x]),
            "gauge_pair_value": float(pair_value[x]),
            "normalised_pair_value": float(normalised[x]),
        }
        for x in range(len(i))
    ]
    return summary, rows


def correlation_length(corr: pd.DataFrame, n: int) -> float:
    upper = corr[corr.i.astype(int) < corr.j.astype(int)].copy()
    upper["separation"] = upper.j.astype(int) - upper.i.astype(int)
    weight = upper.ed_connected.astype(float).abs().to_numpy()
    return float(
        np.sum(upper.separation.to_numpy() * weight) / max(np.sum(weight), 1e-300)
    )


def observed_scale_rows(
    records: dict[tuple, dict], cfg: dict[str, Any]
) -> pd.DataFrame:
    rows = []
    for n in cfg["gauge"]["sizes"]:
        for cell in cfg["gauge"]["cells"]:
            radial = [
                records[(n, cell["tag"], "radial", s)] for s in cfg["gauge"]["seeds"]
            ]
            corr_scale = correlation_length(radial[0]["corr"], n)
            for arm in cfg["gauge"]["arms"]:
                site_repairs, sep_repairs = [], []
                for seed in cfg["gauge"]["seeds"]:
                    r, a = (
                        records[(n, cell["tag"], "radial", seed)],
                        records[(n, cell["tag"], arm, seed)],
                    )
                    rs, asite = (
                        r["site"].sort_values("site"),
                        a["site"].sort_values("site"),
                    )
                    site_repairs.append(
                        rs.exact_model_abs_error.to_numpy(float)
                        - asite.exact_model_abs_error.to_numpy(float)
                    )
                    rc, ac = r["corr"].copy(), a["corr"].copy()
                    rc["sep"] = (rc.j - rc.i).abs()
                    ac["sep"] = (ac.j - ac.i).abs()
                    sep_repairs.append(
                        [
                            float(
                                rc.loc[rc.sep == d, "exact_model_abs_error"].mean()
                                - ac.loc[ac.sep == d, "exact_model_abs_error"].mean()
                            )
                            for d in range(1, n)
                        ]
                    )
                site_median = np.median(np.asarray(site_repairs), axis=0)
                depth = np.minimum(np.arange(n), n - 1 - np.arange(n))
                depth_repair = {
                    int(d): float(np.mean(site_median[depth == d]))
                    for d in sorted(set(depth))
                }
                positive = [
                    d
                    for d in sorted(depth_repair)
                    if all(depth_repair[x] > 0 for x in range(d + 1))
                ]
                width = float(max(positive) + 1 if positive else 0)
                rows.append(
                    {
                        "n_sites": n,
                        "cell_tag": cell["tag"],
                        "delta": cell["delta"],
                        "c6": cell["c6"],
                        "arm": arm,
                        "blockade_radius": float(cell["c6"]) ** (1 / 6),
                        "exact_correlation_length": corr_scale,
                        "boundary_layer_width": width,
                        "site_repair_by_depth_json": json.dumps(
                            depth_repair, sort_keys=True
                        ),
                        "separation_correlation_repair_json": json.dumps(
                            {
                                d: float(x)
                                for d, x in enumerate(
                                    np.median(np.asarray(sep_repairs), axis=0), 1
                                )
                            },
                            sort_keys=True,
                        ),
                    }
                )
    return pd.DataFrame(rows)


def linkage(
    scales: pd.DataFrame, diagnostics: pd.DataFrame, cfg: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    curve = (
        diagnostics[diagnostics.arm != "radial"]
        .groupby(["n_sites", "cell_tag", "delta", "c6", "arm"], as_index=False)
        .curve_landmark_separation.median()
    )
    data = scales.merge(
        curve, on=["n_sites", "cell_tag", "delta", "c6", "arm"], validate="one_to_one"
    )
    rows = []
    for arm in cfg["gauge"]["arms"]:
        group = data[data.arm == arm]
        for outcome in ("curve_landmark_separation", "boundary_layer_width"):
            for scale in ("blockade_radius", "exact_correlation_length"):
                rho = (
                    float(spearmanr(group[scale], group[outcome]).statistic)
                    if group[outcome].nunique() > 1
                    else 0.0
                )
                train, test = group[group.n_sites == 10], group[group.n_sites == 12]
                coefficient = np.polyfit(train[scale], train[outcome], 1)
                predicted = np.polyval(coefficient, test[scale])
                denom = max(float(group[outcome].max() - group[outcome].min()), 1.0)
                nrmse = float(
                    np.sqrt(np.mean((predicted - test[outcome]) ** 2)) / denom
                )
                passed = abs(rho) >= float(
                    cfg["gauge"]["held_out_cell_min_abs_spearman"]
                ) and nrmse <= float(cfg["gauge"]["held_out_size_max_normalised_rmse"])
                rows.append(
                    {
                        "arm": arm,
                        "outcome": outcome,
                        "physical_scale": scale,
                        "spearman_all_six": rho,
                        "held_out_size_normalised_rmse": nrmse,
                        "pass": bool(passed),
                    }
                )
    table = pd.DataFrame(rows)
    passed = table[table["pass"]]
    decision = {
        "schema_version": "stage4i-g1-gauge-scale-gate-v1",
        "validation": "PASS",
        "interpretability_gate": "PASS" if len(passed) else "FAIL",
        "passing_associations": int(len(passed)),
        "tested_associations": int(len(table)),
        "claim_if_pass": "a preregistered curve landmark or boundary-layer association survives held-out size",
        "claim_if_fail": "numerical seed stability only; physical-interpretability language is forbidden",
    }
    return table, decision


def analyse(
    cfg: dict[str, Any], n10_root: Path, n12_root: Path, output: Path
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    records: dict[tuple, dict] = {}
    diagnostics, functions, spectra = [], [], []
    for n, root in ((10, n10_root), (12, n12_root)):
        for cell in cfg["gauge"]["cells"]:
            for arm in ("radial", *cfg["gauge"]["arms"]):
                matrix, pair, pair_i, pair_j = design(n, float(cell["c6"]), arm)
                _, singular, vh = np.linalg.svd(matrix, full_matrices=False)
                tolerance = max(matrix.shape) * np.finfo(float).eps * singular[0]
                rank = int(np.sum(singular > tolerance))
                row_basis = vh[:rank]
                spectra.extend(
                    {
                        "n_sites": n,
                        "cell_tag": cell["tag"],
                        "arm": arm,
                        "singular_index": k,
                        "singular_value": float(x),
                    }
                    for k, x in enumerate(singular)
                )
                reference = None
                for seed in cfg["gauge"]["seeds"]:
                    record = load_record(n, root, cell, arm, seed)
                    records[(n, cell["tag"], arm, seed)] = record
                    summary, rows = gauge_one(
                        n,
                        cell,
                        arm,
                        seed,
                        record,
                        matrix,
                        pair,
                        pair_i,
                        pair_j,
                        row_basis,
                        tolerance,
                    )
                    values = np.array([r["normalised_pair_value"] for r in rows])
                    if reference is None:
                        reference = values.copy()
                    elif float(np.dot(values, reference)) < 0:
                        for row in rows:
                            row["normalised_pair_value"] *= -1
                    diagnostics.append(summary)
                    functions.extend(rows)
    diag, funcs = pd.DataFrame(diagnostics), pd.DataFrame(functions)
    bands = funcs.groupby(
        ["n_sites", "cell_tag", "delta", "c6", "arm", "pair_i", "pair_j", "separation"],
        as_index=False,
    ).normalised_pair_value.agg(
        mean="mean",
        lower_95=lambda x: np.quantile(x, 0.025),
        upper_95=lambda x: np.quantile(x, 0.975),
        standard_deviation="std",
    )
    scales = observed_scale_rows(records, cfg)
    links, decision = linkage(scales, diag, cfg)
    diag.to_csv(output / "gauge_diagnostics.csv", index=False)
    funcs.to_csv(output / "gauge_fixed_functions.csv", index=False)
    bands.to_csv(output / "gauge_confidence_bands.csv", index=False)
    pd.DataFrame(spectra).to_csv(output / "condition_spectra.csv", index=False)
    scales.to_csv(output / "physical_scale_and_repair.csv", index=False)
    links.to_csv(output / "held_out_scale_linkage.csv", index=False)
    (output / "stage4i_g1_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n"
    )
    return decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--n10-root", type=Path)
    parser.add_argument("--n12-root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results_stage4i/gauge"))
    args = parser.parse_args()
    cfg = read_config(args.config)
    if validate_config(cfg)["validation"] != "PASS":
        raise SystemExit("invalid Stage 4I config")
    n10 = args.n10_root or Path(cfg["inputs"]["stage4f_n10_root"])
    n12 = args.n12_root or Path(cfg["inputs"]["stage4h_root"])
    result = analyse(cfg, n10, n12, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
