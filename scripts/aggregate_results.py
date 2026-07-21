#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from graphkan_amo_nqs.manifests import atomic_write_dataframe
from graphkan_amo_nqs.stats.bootstrap import bootstrap_ci
from graphkan_amo_nqs.stats.paired_tests import wilcoxon_or_permutation
from graphkan_amo_nqs.stats.multiple_comparison import holm_correction
from graphkan_amo_nqs.stats.decision_gates import evaluate_gates
from graphkan_amo_nqs.interpretability.stability import curve_alignment, stability_score


def load_metrics(root: Path) -> pd.DataFrame:
    path = root / "raw" / "run_metrics.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def model_comparison(df: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for (profile, model, metric), g in df[df["metric"].isin(["energy_density_error","correlation_mae","density_mae","fidelity","runtime_sec","parameter_count"])].groupby(["profile","model","metric"]):
        med, lo, hi = bootstrap_ci(g["value"].to_numpy(), n_resamples=500, seed=0)
        rows.append({"profile":profile,"model":model,"metric":metric,"median":med,"ci_low":lo,"ci_high":hi,"n":len(g)})
    return pd.DataFrame(rows)


def paired_comparisons(df: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    metrics=["energy_density_error","correlation_mae"]
    candidates=["kan_jastrow"]
    for metric in metrics:
        sub=df[df["metric"]==metric].copy()
        if sub.empty: continue
        pivot=sub.pivot_table(index=["profile","instance_id","seed"], columns="model", values="value", aggfunc="first")
        for cand in candidates:
            if cand not in pivot.columns: continue
            for baseline in [c for c in pivot.columns if c != cand]:
                both=pivot[[baseline,cand]].dropna()
                if both.empty: continue
                deltas=both[baseline]-both[cand]
                med,lo,hi=bootstrap_ci(deltas.to_numpy(), n_resamples=500, seed=0)
                test=wilcoxon_or_permutation(deltas.to_numpy())
                rows.append({"comparison":f"{baseline}_minus_{cand}","metric":metric,"median_delta_baseline_minus_candidate":med,"ci_low":lo,"ci_high":hi,"n_pairs":len(deltas),"p_value":test.get("p_value",np.nan),"test":test.get("test")})
    out=pd.DataFrame(rows)
    if not out.empty:
        out["p_holm"] = holm_correction(out["p_value"].tolist())
    return out


def interpretability_stability(root: Path) -> pd.DataFrame:
    path=root/"interpretability"/"gauge_fixed_functions.csv"
    if not path.exists(): return pd.DataFrame(columns=["profile","instance_id","model","curve_align","stability_score","n_seeds"])
    df=pd.read_csv(path)
    rows=[]
    for (profile, inst, model), g in df.groupby(["profile","instance_id","model"]):
        curves=[]
        for seed, gs in g.groupby("seed"):
            curves.append(gs.sort_values("r")["gauge_fixed_value"].to_numpy())
        rows.append({"profile":profile,"instance_id":inst,"model":model,"curve_align":curve_alignment(curves),"stability_score":stability_score(curves),"n_seeds":len(curves)})
    return pd.DataFrame(rows)


def main():
    ap=argparse.ArgumentParser(description="Aggregate raw result CSVs into comparison/statistics tables.")
    ap.add_argument("--input", default="results")
    ap.add_argument("--output", default="results/aggregated")
    args=ap.parse_args(); root=Path(args.input); out=Path(args.output); out.mkdir(parents=True, exist_ok=True)
    df=load_metrics(root)
    mc=model_comparison(df) if not df.empty else pd.DataFrame()
    pc=paired_comparisons(df) if not df.empty else pd.DataFrame()
    st=interpretability_stability(root)
    gates=evaluate_gates(mc, pc, st)
    atomic_write_dataframe(out/"model_comparison.csv", mc)
    atomic_write_dataframe(out/"paired_comparisons.csv", pc)
    atomic_write_dataframe(out/"ablation_summary.csv", pc)
    # Robustness not run: explicit status row.
    atomic_write_dataframe(out/"robustness_summary.csv", pd.DataFrame([{"profile":"robustness_core","status":"not_run","reason":"CPU artifact did not execute robustness profile."}]))
    atomic_write_dataframe(root/"interpretability"/"function_stability.csv", st)
    atomic_write_dataframe(root/"interpretability"/"scale_collapse.csv", pd.DataFrame([{"status":"pilot_data_in_gauge_fixed_functions","source":"results/interpretability/gauge_fixed_functions.csv"}]))
    atomic_write_dataframe(root/"interpretability"/"correlation_matching.csv", df[df["metric"]=="correlation_matching_spearman"] if not df.empty else pd.DataFrame())
    atomic_write_dataframe(out/"readiness_gates.csv", gates)
    print(f"aggregated {len(df)} raw metric rows")

if __name__=="__main__":
    main()
