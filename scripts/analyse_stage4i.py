#!/usr/bin/env python3
"""Merge and decide the Stage 4I G2 orthogonalised counterfactual."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from graphkan_amo_nqs.training.stage4h_transplant import sha256_file
from stage4i_protocol import (
    DEFAULT_CONFIG,
    read_config,
    roster,
    validate_config,
    verify_freeze,
)


def verify_run(run: Path) -> tuple[dict, list[str]]:
    errors: list[str] = []
    completion = (
        json.loads((run / "completion.json").read_text())
        if (run / "completion.json").is_file()
        else {}
    )
    attempt = run / "attempts" / str(completion.get("attempt", ""))
    for relative, expected in completion.get("payload_hashes", {}).items():
        path = attempt / relative
        if not path.is_file() or sha256_file(path) != expected:
            errors.append(f"payload mismatch: {run.name}/{relative}")
    if completion.get("status") != "COMPLETED":
        errors.append(f"nonterminal run: {run.name}")
    return completion, errors


def analyse(
    cfg: dict,
    input_root: Path,
    output: Path,
    protocol: Path,
    test_gate: Path = Path(
        "results_stage4i/test_profiles/stage4i_test_profile_gate.json"
    ),
) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    canonical = output / "runs"
    canonical.mkdir(exist_ok=True)
    expected = {x["run_id"]: x for x in roster(cfg)}
    observed = {
        p.parent.name: p.parent for p in input_root.glob("*/runs/*/completion.json")
    }
    errors = []
    if set(expected) != set(observed):
        errors.append(
            f"roster mismatch: missing={len(set(expected) - set(observed))}, foreign={len(set(observed) - set(expected))}"
        )
    metrics = []
    for run_id, source in sorted(observed.items()):
        completion, run_errors = verify_run(source)
        errors.extend(run_errors)
        if run_errors or run_id not in expected:
            continue
        target = canonical / run_id
        if not target.exists():
            shutil.copytree(source, target)
        elif sha256_file(target / "completion.json") != sha256_file(
            source / "completion.json"
        ):
            errors.append(f"canonical collision: {run_id}")
            continue
        metrics.append(
            json.loads(
                (
                    target / "attempts" / completion["attempt"] / "metrics.json"
                ).read_text()
            )
        )
    frame = pd.DataFrame(metrics)
    (output / "tables").mkdir(exist_ok=True)
    frame.to_csv(output / "tables/metrics.csv", index=False)
    if verify_freeze(cfg, protocol)["validation"] != "PASS":
        errors.append("frozen Stage 4I source/protocol ledger does not close")
    if (
        len(frame) != 15
        or int((frame.scientific_validity_exact == "VALID").sum()) != 15
    ):
        errors.append("15/15 exact-health runs not present")
    key = ["delta", "c6", "seed"]
    radial = frame[frame.arm == "stage4i_radial"][
        key
        + [
            "exact_model_energy",
            "exact_model_density_mae_to_ed",
            "exact_model_connected_correlation_mae_to_ed",
        ]
    ].rename(
        columns={
            "exact_model_energy": "radial_energy",
            "exact_model_density_mae_to_ed": "radial_density",
            "exact_model_connected_correlation_mae_to_ed": "radial_correlation",
        }
    )
    paired = frame[frame.arm != "stage4i_radial"].merge(
        radial, on=key, validate="many_to_one"
    )
    paired["energy_excess"] = paired.exact_model_energy - paired.radial_energy
    paired["density_repair"] = (
        paired.radial_density - paired.exact_model_density_mae_to_ed
    )
    paired["correlation_repair"] = (
        paired.radial_correlation - paired.exact_model_connected_correlation_mae_to_ed
    )
    paired.to_csv(output / "tables/paired_counterfactual.csv", index=False)
    summary = paired.groupby("arm", as_index=False).agg(
        pairs=("seed", "count"),
        energy_safe_seeds=(
            "energy_excess",
            lambda x: int(
                (
                    x
                    <= float(
                        cfg["counterfactual"]["acceptance"]["energy_safety_margin"]
                    )
                ).sum()
            ),
        ),
        correlation_positive_seeds=("correlation_repair", lambda x: int((x > 0).sum())),
        correlation_worse_seeds=("correlation_repair", lambda x: int((x < 0).sum())),
        maximum_density_degradation=(
            "density_repair",
            lambda x: float(max(0.0, -x.min())),
        ),
        median_density_repair=("density_repair", "median"),
        median_correlation_repair=("correlation_repair", "median"),
        maximum_energy_excess=("energy_excess", "max"),
    )
    summary.to_csv(output / "tables/g2_arm_summary.csv", index=False)
    orth = (
        summary[summary.arm == "stage4i_endpoint_full_orthogonal"].iloc[0]
        if len(summary)
        else None
    )
    gate = cfg["counterfactual"]["acceptance"]
    passed = (
        orth is not None
        and int(orth.energy_safe_seeds) >= gate["energy_safe_seeds_required"]
        and int(orth.correlation_positive_seeds)
        >= gate["correlation_positive_seeds_required"]
        and float(orth.maximum_density_degradation)
        <= float(gate["maximum_density_degradation"])
    )
    negative = (
        orth is not None
        and int(orth.correlation_worse_seeds)
        >= gate["negative_closure_correlation_worse_seeds"]
    )
    decision = (
        "ORTHOGONALISED_RESCUE_PASS"
        if passed
        else "ORTHOGONALISED_NEGATIVE_CLOSURE"
        if negative
        else "ORTHOGONALISED_COUNTERFACTUAL_INCONCLUSIVE"
    )
    g0 = json.loads(test_gate.read_text()) if test_gate.is_file() else {}
    if g0.get("validation") != "PASS":
        errors.append("G0 base/full-NQS test-profile parity is absent or failed")
    g1_path = output.parent / "gauge/stage4i_g1_decision.json"
    g1 = json.loads(g1_path.read_text()) if g1_path.is_file() else {}
    if g1.get("validation") != "PASS":
        errors.append("G1 checkpoint-only analysis is absent or invalid")
    if decision == "ORTHOGONALISED_COUNTERFACTUAL_INCONCLUSIVE":
        errors.append("G2 neither rescue nor negative-closure threshold was reached")
    result = {
        "schema_version": "stage4i-final-gate-v1",
        "validation": "PASS" if not errors else "FAIL",
        "blocking_errors": errors,
        "decision_class": decision,
        "runs": len(frame),
        "exact_health_passes": int((frame.scientific_validity_exact == "VALID").sum())
        if len(frame)
        else 0,
        "G0_release_test_parity": g0.get("validation", "NOT_RUN"),
        "G1_gauge_scale_linkage": g1.get("interpretability_gate", "NOT_RUN"),
        "G1_claim_boundary": g1.get("claim_if_pass")
        if g1.get("interpretability_gate") == "PASS"
        else g1.get("claim_if_fail", "physical interpretability is not authorised"),
        "G2_orthogonalised_counterfactual": "PASS"
        if passed
        else "NEGATIVE_CLOSURE_PASS"
        if negative
        else "INCONCLUSIVE",
        "endpoint_architecture_promotion": "AUTHORISED_FOR_CONDITIONAL_G3"
        if passed
        else "STOP",
        "conditional_n14_dmrg": "SUPERVISOR_REVIEW_REQUIRED"
        if not errors
        else "NOT_AUTHORISED",
        "stage4h_frozen_decision": "NEGATIVE_CLOSURE_PASS",
    }
    (output / "stage4i_g2_decision.json").write_text(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k
                not in {
                    "G0_release_test_parity",
                    "G1_gauge_scale_linkage",
                    "G1_claim_boundary",
                }
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (output / "stage4i_final_gate.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("results_stage4i/merged"))
    p.add_argument("--protocol", type=Path, default=Path("results_stage4i/protocol"))
    p.add_argument(
        "--test-gate",
        type=Path,
        default=Path("results_stage4i/test_profiles/stage4i_test_profile_gate.json"),
    )
    a = p.parse_args()
    cfg = read_config(a.config)
    if validate_config(cfg)["validation"] != "PASS":
        raise SystemExit("invalid Stage 4I config")
    result = analyse(cfg, a.input, a.output, a.protocol, a.test_gate)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["validation"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
