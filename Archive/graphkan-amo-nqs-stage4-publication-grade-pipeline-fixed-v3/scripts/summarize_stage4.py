#!/usr/bin/env python
"""Stage 4 summary and dual endpoint-transfer gates.

Produces generic model summaries plus separate N=8 gates for the endpoint-sum
and endpoint-full RBF variants against the matched RBF-radial baseline.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

METRICS = [
    "exact_model_abs_energy_error_to_ed",
    "exact_model_density_mae_to_ed",
    "exact_model_connected_correlation_mae_to_ed",
    "wall_seconds",
]


def paired_gate(df: pd.DataFrame, baseline: str, candidate: str) -> dict:
    rad = df[df.model_label == baseline].set_index("seed")
    end = df[df.model_label == candidate].set_index("seed")
    common = rad.index.intersection(end.index)
    if len(common) == 0:
        return {
            "candidate": candidate,
            "baseline": baseline,
            "status": "NOT_APPLICABLE",
            "reason": "Required paired labels missing",
        }
    dE = end.loc[common, METRICS[0]] - rad.loc[common, METRICS[0]]
    dD = end.loc[common, METRICS[1]] - rad.loc[common, METRICS[1]]
    dC = end.loc[common, METRICS[2]] - rad.loc[common, METRICS[2]]
    corr_base = rad.loc[common, METRICS[2]]
    valid = bool((end.loc[common, "scientific_validity"] == "VALID").all())
    energy_ok = bool((dE <= 0).all())
    density_ok = bool((dD <= 0).all())
    corr_ok = bool((dC <= 0.01).all() and (dC <= 0.25 * corr_base).all())
    return {
        "candidate": candidate,
        "baseline": baseline,
        "status": "PASS" if valid and energy_ok and density_ok and corr_ok else "FAIL",
        "pairs": int(len(common)),
        "candidate_all_valid": valid,
        "energy_not_worse_all_seeds": energy_ok,
        "density_not_worse_all_seeds": density_ok,
        "correlation_penalty_within_abs_0_01_and_rel_25pct": corr_ok,
        "median_energy_delta": float(dE.median()),
        "median_density_delta": float(dD.median()),
        "median_correlation_delta": float(dC.median()),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    a = p.parse_args()
    root = Path(a.results)
    raw, out = root / "raw", root / "aggregated"
    out.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(raw / "stage3f_metrics.csv")

    identity = ["model_label", "model_kind", "basis_family", "n_parameters"]
    summary = m.groupby(identity, dropna=False).agg(
        runs=("run_id", "count"),
        valid=("scientific_validity", lambda x: (x == "VALID").sum()),
        **{f"median_{k}": (k, "median") for k in METRICS},
    ).reset_index()
    summary["valid_rate"] = summary["valid"] / summary["runs"]
    summary.to_csv(out / "model_summary.csv", index=False)
    summary[["model_label", "n_parameters"]].to_csv(out / "parameter_audit.csv", index=False)

    contrasts = []
    comparisons = [
        ("sine_radial_21p", "sine_endpoint_full_21p", "endpoint effect under sine basis at N=8"),
        ("rbf_radial_21p", "rbf_endpoint_sum_21p", "endpoint-sum effect under RBF basis at N=8"),
        ("rbf_radial_21p", "rbf_endpoint_full_21p", "full endpoint effect under RBF basis at N=8"),
        ("rbf_endpoint_full_21p", "rbf_endpoint_sum_21p", "sum-only versus full endpoint at N=8"),
    ]
    for base, treated, question in comparisons:
        a_df = m[m.model_label == base].set_index("seed")
        b_df = m[m.model_label == treated].set_index("seed")
        common = a_df.index.intersection(b_df.index)
        if len(common) == 0:
            continue
        for metric in METRICS:
            d = b_df.loc[common, metric] - a_df.loc[common, metric]
            contrasts.append({
                "comparison": f"{treated} minus {base}",
                "question": question,
                "metric": metric,
                "pairs": len(common),
                "median_delta": d.median(),
                "mean_delta": d.mean(),
                "treated_lower_fraction": float((d < 0).mean()),
            })
    pd.DataFrame(contrasts).to_csv(out / "paired_contrasts.csv", index=False)

    site_path = raw / "stage3f_site_observables.csv"
    if site_path.exists():
        site = pd.read_csv(site_path)
        site.groupby(["model_label", "basis_family", "site"])["exact_model_abs_error"].median().reset_index().to_csv(
            out / "site_density_error_summary.csv", index=False
        )

    gates = pd.DataFrame([
        paired_gate(m, "rbf_radial_21p", "rbf_endpoint_sum_21p"),
        paired_gate(m, "rbf_radial_21p", "rbf_endpoint_full_21p"),
    ])
    gates.to_csv(out / "n8_joint_gates.csv", index=False)
    print("\nModel summary:\n", summary.to_string(index=False))
    print("\nStage 4 joint gates:\n", gates.to_string(index=False))
    print("\nWrote:", out)


if __name__ == "__main__":
    main()
