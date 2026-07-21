#!/usr/bin/env python3
"""Summarise Stage 4B controls with dual-gate reporting and non-domination checks."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

EXACT = [
    "exact_model_abs_energy_error_to_ed",
    "exact_model_density_mae_to_ed",
    "exact_model_connected_correlation_mae_to_ed",
]


def paired_rows(frame: pd.DataFrame, baseline: str, candidate: str, metric: str) -> pd.Series:
    base = frame[frame.model_label == baseline].set_index("seed")
    cand = frame[frame.model_label == candidate].set_index("seed")
    common = base.index.intersection(cand.index)
    return cand.loc[common, metric] - base.loc[common, metric]


def generic_contrasts(frame: pd.DataFrame) -> pd.DataFrame:
    out = []
    labels = sorted(frame.model_label.dropna().unique())
    for candidate in labels:
        for baseline in labels:
            if candidate == baseline:
                continue
            for metric in EXACT + ["wall_seconds"]:
                try:
                    delta = paired_rows(frame, baseline, candidate, metric)
                except KeyError:
                    continue
                if len(delta) == 0:
                    continue
                out.append({
                    "comparison": f"{candidate} minus {baseline}", "candidate": candidate, "baseline": baseline,
                    "metric": metric, "pairs": len(delta), "median_delta": float(delta.median()),
                    "mean_delta": float(delta.mean()), "candidate_lower_fraction": float((delta < 0).mean()),
                })
    return pd.DataFrame(out)


def non_domination_table(frame: pd.DataFrame, candidate: str, controls: list[str]) -> pd.DataFrame:
    rows = []
    for control in controls:
        if control not in set(frame.model_label):
            rows.append({"candidate": candidate, "control": control, "status": "NOT_AVAILABLE"})
            continue
        c = frame[frame.model_label == candidate].set_index("seed")
        b = frame[frame.model_label == control].set_index("seed")
        common = c.index.intersection(b.index)
        if len(common) == 0:
            rows.append({"candidate": candidate, "control": control, "status": "NOT_APPLICABLE"})
            continue
        d = {metric: c.loc[common, metric] - b.loc[common, metric] for metric in EXACT}
        runtime = c.loc[common, "wall_seconds"] / b.loc[common, "wall_seconds"].replace(0, np.nan)
        control_dominates = all((d[m] >= 0).all() for m in EXACT) and any((d[m] > 0).any() for m in EXACT) and bool((runtime <= 1.25).all())
        rows.append({
            "candidate": candidate, "control": control, "pairs": len(common),
            "candidate_exact_valid_all": bool((c.loc[common, "scientific_validity_exact"] == "VALID").all()),
            "candidate_not_worse_energy_all": bool((d[EXACT[0]] <= 0).all()),
            "candidate_not_worse_density_all": bool((d[EXACT[1]] <= 0).all()),
            "candidate_not_worse_corr_all": bool((d[EXACT[2]] <= 0).all()),
            "candidate_runtime_ratio_median": float(runtime.median()),
            "control_dominates_candidate_under_1_25x_compute": control_dominates,
            "status": "DOMINATED" if control_dominates else "NOT_DOMINATED",
            "median_energy_delta": float(d[EXACT[0]].median()),
            "median_density_delta": float(d[EXACT[1]].median()),
            "median_corr_delta": float(d[EXACT[2]].median()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--candidate", default="rbf_endpoint_sum_21p")
    parser.add_argument("--controls", nargs="*", default=None)
    args = parser.parse_args()
    root = Path(args.results)
    raw = root / "raw"
    out = root / "aggregated"
    out.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(raw / "stage4b_metrics.csv")
    completed = metrics[metrics.status == "COMPLETED"].copy()
    if completed.empty:
        raise SystemExit("No completed Stage 4B rows found.")

    identity = ["model_label", "model_kind", "basis_family", "n_parameters"]
    summary = completed.groupby(identity, dropna=False).agg(
        runs=("run_id", "count"),
        exact_valid=("scientific_validity_exact", lambda x: int((x == "VALID").sum())),
        sample_valid=("scientific_validity_sample", lambda x: int((x == "VALID").sum())),
        median_exact_energy_error=(EXACT[0], "median"),
        median_exact_density_mae=(EXACT[1], "median"),
        median_exact_corr_mae=(EXACT[2], "median"),
        median_wall_seconds=("wall_seconds", "median"),
    ).reset_index()
    summary["exact_valid_rate"] = summary.exact_valid / summary.runs
    summary["sample_valid_rate"] = summary.sample_valid / summary.runs
    summary.to_csv(out / "model_summary.csv", index=False)

    dual = completed.groupby("model_label", dropna=False).agg(
        rows=("run_id", "count"),
        exact_valid=("scientific_validity_exact", lambda x: int((x == "VALID").sum())),
        sample_valid=("scientific_validity_sample", lambda x: int((x == "VALID").sum())),
    ).reset_index()
    dual["duality_agrees_all_rows"] = dual.exact_valid == dual.sample_valid
    dual.to_csv(out / "health_gate_comparison.csv", index=False)

    contrasts = generic_contrasts(completed)
    contrasts.to_csv(out / "paired_contrasts.csv", index=False)

    default_controls = [label for label in sorted(completed.model_label.unique()) if label != args.candidate]
    controls = args.controls if args.controls else default_controls
    non_dom = non_domination_table(completed, args.candidate, controls)
    non_dom.to_csv(out / "endpoint_sum_non_domination.csv", index=False)

    site_path = raw / "stage4b_site_observables.csv"
    if site_path.exists():
        site = pd.read_csv(site_path)
        site.groupby(["model_label", "site"])["exact_model_abs_error"].median().reset_index().to_csv(
            out / "site_density_error_summary.csv", index=False
        )
    corr_path = raw / "stage4b_corr_observables.csv"
    if corr_path.exists():
        corr = pd.read_csv(corr_path)
        corr.groupby(["model_label", "i", "j"])["exact_model_abs_error"].median().reset_index().to_csv(
            out / "correlation_error_summary.csv", index=False
        )

    gate = {
        "candidate": args.candidate,
        "candidate_present": args.candidate in set(completed.model_label),
        "candidate_exact_valid_all": bool((completed[completed.model_label == args.candidate]["scientific_validity_exact"] == "VALID").all()) if args.candidate in set(completed.model_label) else False,
        "no_control_dominates_under_1_25x_compute": bool((non_dom.status != "DOMINATED").all()) if not non_dom.empty else False,
        "stage4b_status": "PASS" if args.candidate in set(completed.model_label) and bool((non_dom.status != "DOMINATED").all()) else "INCOMPLETE_OR_FAIL",
        "interpretation": "Exact metrics and exact validity drive the headline gate; sampled gate is reported separately.",
    }
    pd.DataFrame([gate]).to_csv(out / "stage4b_decision_gate.csv", index=False)
    print("\nStage 4B model summary:\n", summary.to_string(index=False))
    print("\nStage 4B non-domination table:\n", non_dom.to_string(index=False))
    print("\nStage 4B decision gate:\n", pd.DataFrame([gate]).to_string(index=False))
    print("\nWrote:", out)


if __name__ == "__main__":
    main()
