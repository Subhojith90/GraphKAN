#!/usr/bin/env python3
"""Strict Stage 4H scientific gate and conditioning/gauge analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from graphkan_amo_nqs.models.stage4g_flax_nqs import (
    nongeometric_pair_features,
    pair_feature_arrays,
)
from stage4g_gauge import endpoint_features, minimum_norm_gauge
from stage4h_protocol import DEFAULT_CONFIG, read_config, validate_config, verify_freeze

ENDPOINT_ARMS = ("endpoint_full_warmstart", "endpoint_hybrid_warmstart")
EXPANDED_ARMS = (*ENDPOINT_ARMS, "nongeometric_warmstart")


def boolean_states(n_sites: int) -> np.ndarray:
    integers = np.arange(1 << n_sites, dtype=np.uint64)[:, None]
    return ((integers >> np.arange(n_sites, dtype=np.uint64)) & 1).astype(np.float64)


def arm_design(
    n_sites: int, r_scale: float, arm: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    states = boolean_states(n_sites)
    i, j, radial = pair_feature_arrays(
        np.arange(n_sites, dtype=float)[:, None],
        n_basis=32,
        family="rbf",
        r_scale=r_scale,
    )
    pair_occ = states[:, i] * states[:, j]
    if arm == "endpoint_full_warmstart":
        extra = endpoint_features(n_sites, i, j, "endpoint_full_b32")
    elif arm == "endpoint_hybrid_warmstart":
        extra = endpoint_features(n_sites, i, j, "endpoint_hybrid_b32")
    elif arm == "nongeometric_warmstart":
        extra = nongeometric_pair_features(i, j)
    else:
        extra = np.empty((len(i), 0))
    pair = np.concatenate([radial, extra], axis=1)
    design = np.concatenate([states, pair_occ @ pair], axis=1)
    return design, pair, i, j


def effective_condition(matrix: np.ndarray) -> tuple[int, float, float]:
    singular = np.linalg.svd(matrix, compute_uv=False)
    tolerance = (
        max(matrix.shape)
        * np.finfo(float).eps
        * (singular[0] if len(singular) else 0.0)
    )
    rank = int(np.sum(singular > tolerance))
    condition = float(singular[0] / singular[rank - 1]) if rank else float("inf")
    return rank, condition, float(singular[rank - 1]) if rank else 0.0


def conditioning_table(cfg: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for cell in cfg["physics"]["cells"]:
        radial_design, _, _, _ = arm_design(
            12, float(cell["c6"]) ** (1 / 6), "radial_continuation"
        )
        q, _ = np.linalg.qr(radial_design, mode="reduced")
        for arm in cfg["arms"]:
            design, pair, _, _ = arm_design(12, float(cell["c6"]) ** (1 / 6), arm)
            rank, condition, smallest = effective_condition(design)
            extra_columns = design[:, radial_design.shape[1] :]
            residual = (
                extra_columns - q @ (q.T @ extra_columns)
                if extra_columns.size
                else extra_columns
            )
            rows.append(
                {
                    "cell_tag": cell["tag"],
                    "delta": cell["delta"],
                    "c6": cell["c6"],
                    "arm": arm,
                    "design_rows": design.shape[0],
                    "design_columns": design.shape[1],
                    "rank": rank,
                    "effective_condition_number": condition,
                    "smallest_nonzero_singular_value": smallest,
                    "extra_feature_columns": extra_columns.shape[1],
                    "extra_residual_rank": int(np.linalg.matrix_rank(residual))
                    if residual.size
                    else 0,
                    "extra_residual_fraction": float(
                        np.linalg.norm(residual)
                        / max(np.linalg.norm(extra_columns), 1e-300)
                    )
                    if residual.size
                    else 0.0,
                }
            )
    return pd.DataFrame(rows)


def gauge_tables(
    merged: Path, cfg: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    diagnostics, functions = [], []
    for run_root in sorted((merged / "runs").iterdir()):
        completion = json.loads((run_root / "completion.json").read_text())
        attempt = run_root / "attempts" / completion["attempt"]
        metrics = json.loads((attempt / "metrics.json").read_text())
        arm = metrics["arm"]
        checkpoint = attempt / "checkpoints" / "step-06000.npz"
        with np.load(checkpoint, allow_pickle=False) as archive:
            params = {
                name: np.asarray(archive[name], dtype=float) for name in archive.files
            }
        design, pair, pair_i, pair_j = arm_design(
            12, float(metrics["c6"]) ** (1 / 6), arm
        )
        extra_name = cfg["arms"][arm]["extra_parameter"]
        theta_parts = [params["site_bias"], params["kan_coeff"]]
        if extra_name:
            theta_parts.append(params[extra_name])
        theta = np.concatenate(theta_parts)
        gauged, diagnostic = minimum_norm_gauge(design, theta)
        pair_coeff = gauged[12:]
        edge = pair @ pair_coeff
        diagnostics.append(
            {
                "run_id": metrics["run_id"],
                "arm": arm,
                "delta": metrics["delta"],
                "c6": metrics["c6"],
                "seed": metrics["seed"],
                "gauge_definition": "full-Boolean-design minimum-Euclidean-norm",
                **diagnostic,
                "gauge_pair_function_norm": float(np.linalg.norm(edge)),
                "gauge_extra_coefficient_norm": float(np.linalg.norm(gauged[44:]))
                if len(gauged) > 44
                else 0.0,
            }
        )
        for k, value in enumerate(edge):
            functions.append(
                {
                    "run_id": metrics["run_id"],
                    "arm": arm,
                    "delta": metrics["delta"],
                    "c6": metrics["c6"],
                    "seed": metrics["seed"],
                    "pair_i": int(pair_i[k]),
                    "pair_j": int(pair_j[k]),
                    "gauge_pair_value": float(value),
                }
            )
    return pd.DataFrame(diagnostics), pd.DataFrame(functions)


def analyse(
    merged: Path, protocol: Path, periodic_alias: Path, cfg: dict[str, Any]
) -> dict[str, Any]:
    analysis = merged / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    merge_summary = json.loads((merged / "merge_summary.json").read_text())
    if merge_summary.get("validation") != "PASS":
        errors.append("G0: shard merge is not valid")
    freeze_result = verify_freeze(cfg, protocol)
    if freeze_result["validation"] != "PASS":
        errors.append("G0: frozen protocol/source ledger does not close")
    periodic = (
        json.loads(periodic_alias.read_text()) if periodic_alias.is_file() else {}
    )
    periodic_pass = (
        periodic.get("validation") == "PASS" and int(periodic.get("passes", -1)) == 30
    )
    if not periodic_pass:
        errors.append("G0: N12 periodic alias sentinel is incomplete")

    metrics = pd.read_csv(merged / "tables/metrics.csv")
    transplant = pd.read_csv(merged / "tables/transplant_attestations.csv")
    health_passes = int((metrics.scientific_validity_exact == "VALID").sum())
    if len(metrics) != 60 or health_passes != 60:
        errors.append(f"G0: exact-health runs are {health_passes}/60")
    transplant_passes = int((transplant.validation == "PASS").sum())
    if len(transplant) != 60 or transplant_passes != 60:
        errors.append(f"G1: transplant attestations are {transplant_passes}/60")

    outcomes = {
        "energy_error": "exact_model_abs_energy_error_to_ed",
        "density_error": "exact_model_density_mae_to_ed",
        "correlation_error": "exact_model_connected_correlation_mae_to_ed",
    }
    key = ["delta", "c6", "seed"]
    radial = metrics[metrics.arm == "radial_continuation"][
        key + ["exact_model_energy", *outcomes.values()]
    ].copy()
    radial = radial.rename(
        columns={
            "exact_model_energy": "radial_exact_energy",
            **{v: f"radial_{k}" for k, v in outcomes.items()},
        }
    )
    paired = metrics[metrics.arm.isin(EXPANDED_ARMS)].merge(
        radial, on=key, validate="many_to_one"
    )
    paired["energy_safe"] = (
        paired.exact_model_energy
        <= paired.radial_exact_energy + float(cfg["acceptance"]["energy_safety_margin"])
    )
    for name, column in outcomes.items():
        paired[f"repair_{name}"] = paired[f"radial_{name}"] - paired[column]
    paired.to_csv(analysis / "paired_warmstart_vs_radial.csv", index=False)

    safety = (
        paired.groupby("arm")
        .agg(
            pairs=("run_id", "count"),
            energy_safe_pairs=("energy_safe", "sum"),
            maximum_energy_excess=("exact_model_energy", lambda x: 0.0),
        )
        .reset_index()
    )
    for index, row in safety.iterrows():
        group = paired[paired.arm == row.arm]
        safety.loc[index, "maximum_energy_excess"] = float(
            (group.exact_model_energy - group.radial_exact_energy).max()
        )
    safety["passes_14_of_15"] = safety.energy_safe_pairs >= int(
        cfg["acceptance"]["energy_safety_minimum_pairs"]
    )
    safety.to_csv(analysis / "nested_energy_safety.csv", index=False)
    safe_endpoint_arms = set(
        safety[(safety.arm.isin(ENDPOINT_ARMS)) & safety.passes_14_of_15].arm
    )
    g2_pass = bool(safe_endpoint_arms)

    cell_rows = []
    for (arm, delta, c6), group in paired[paired.arm.isin(ENDPOINT_ARMS)].groupby(
        ["arm", "delta", "c6"]
    ):
        density_signs = int((group.repair_density_error > 0).sum())
        correlation_signs = int((group.repair_correlation_error > 0).sum())
        density_median = float(group.repair_density_error.median())
        correlation_median = float(group.repair_correlation_error.median())
        passed = (
            density_median > 0
            and correlation_median > 0
            and min(density_signs, correlation_signs)
            >= int(cfg["acceptance"]["paired_seed_sign_minimum"])
        )
        cell_rows.append(
            {
                "arm": arm,
                "delta": delta,
                "c6": c6,
                "density_median_repair": density_median,
                "correlation_median_repair": correlation_median,
                "density_positive_seeds": density_signs,
                "correlation_positive_seeds": correlation_signs,
                "joint_repair_pass": passed,
            }
        )
    cells = pd.DataFrame(cell_rows)
    cells.to_csv(analysis / "joint_repair_by_cell.csv", index=False)
    cell_counts = {
        arm: int(group.joint_repair_pass.sum()) for arm, group in cells.groupby("arm")
    }
    rescued_arms = [
        arm
        for arm in safe_endpoint_arms
        if cell_counts.get(arm, 0)
        >= int(cfg["acceptance"]["scientific_rescue_minimum_cells"])
    ]
    g3_pass = bool(rescued_arms)
    g4_pass = g2_pass and not g3_pass

    conditioning = conditioning_table(cfg)
    conditioning.to_csv(
        analysis / "condition_number_and_residualisation.csv", index=False
    )
    gauge_diagnostics, gauge_functions = gauge_tables(merged, cfg)
    gauge_diagnostics.to_csv(analysis / "gauge_fixed_diagnostics.csv", index=False)
    gauge_functions.to_csv(analysis / "gauge_fixed_pair_functions.csv", index=False)
    regions = pd.read_csv(merged / "tables/region_density.csv")
    radial_regions = regions[regions.arm == "radial_continuation"].rename(
        columns={"exact_model_mae": "radial_region_mae"}
    )
    region_compare = regions[regions.arm.isin(EXPANDED_ARMS)].merge(
        radial_regions[[*key, "region", "radial_region_mae"]], on=[*key, "region"]
    )
    region_compare["region_repair"] = (
        region_compare.radial_region_mae - region_compare.exact_model_mae
    )
    region_compare.to_csv(analysis / "edge_near_edge_bulk_repair.csv", index=False)

    if not g2_pass:
        decision = "OPTIMISATION_UNRESOLVED"
    elif g3_pass:
        decision = "SCIENTIFIC_RESCUE_PASS"
    else:
        decision = "NEGATIVE_CLOSURE_PASS"
    blocking = list(errors)
    if not g2_pass:
        blocking.append(
            "G2: neither endpoint arm satisfies energy safety in 14/15 pairs"
        )
    result = {
        "schema_version": "stage4h-final-gate-v1",
        "validation": "PASS" if not blocking else "FAIL",
        "decision_class": decision,
        "blocking_errors": blocking,
        "G0_release": "PASS" if not errors else "FAIL",
        "G1_transplant": "PASS" if transplant_passes == 60 else "FAIL",
        "G2_energy_safety": "PASS" if g2_pass else "FAIL",
        "G3_scientific_rescue": "PASS" if g3_pass else "FAIL",
        "G4_negative_closure": "PASS" if g4_pass else "NOT_APPLICABLE",
        "runs": len(metrics),
        "exact_health_passes": health_passes,
        "transplant_passes": transplant_passes,
        "periodic_alias_passes": int(periodic.get("passes", 0)),
        "energy_safe_endpoint_arms": sorted(safe_endpoint_arms),
        "rescued_endpoint_arms": sorted(rescued_arms),
        "joint_repair_cells_by_arm": cell_counts,
        "interpretation": (
            "endpoint benefit is recovered under optimisation-controlled radial warm start"
            if g3_pass
            else "finite-size global endpoint benefit is not recovered despite an energy-safe optimisation-controlled protocol"
            if g4_pass
            else "optimisation remains unresolved; only one orthogonalised-feature pilot is authorised"
        ),
    }
    (analysis / "stage4h_final_gate.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--merged", type=Path, default=Path("results_stage4h/merged"))
    parser.add_argument(
        "--protocol", type=Path, default=Path("results_stage4h/protocol")
    )
    parser.add_argument(
        "--periodic-alias",
        type=Path,
        default=Path("results_stage4h/environment/n12_periodic_alias_gate.json"),
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    cfg = read_config(args.config)
    if validate_config(cfg)["validation"] != "PASS":
        raise SystemExit("invalid Stage 4H config")
    result = analyse(args.merged, args.protocol, args.periodic_alias, cfg)
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.strict and result["validation"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
