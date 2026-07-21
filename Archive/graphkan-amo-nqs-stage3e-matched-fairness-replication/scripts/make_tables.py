#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from graphkan_amo_nqs.reporting.tables import dataframe_to_latex, safe_claim_table
from graphkan_amo_nqs.baselines.fairness import fairness_matrix
from graphkan_amo_nqs.baselines.compute_matching import compute_budget_table
from graphkan_amo_nqs.manifests import atomic_write_dataframe


def read(path):
    return pd.read_csv(path) if Path(path).exists() else pd.DataFrame()


def main():
    ap=argparse.ArgumentParser(description="Generate LaTeX tables from result CSVs.")
    ap.parse_args()
    root=Path("results"); tables=Path("tables"); tables.mkdir(exist_ok=True)
    edv=read(root/"references"/"ed_validation.csv")
    mc=read(root/"aggregated"/"model_comparison.csv")
    pc=read(root/"aggregated"/"paired_comparisons.csv")
    ab=read(root/"aggregated"/"ablation_summary.csv")
    st=read(root/"interpretability"/"function_stability.csv")
    fr=read(root/"raw"/"failed_runs.csv")
    gates=read(root/"aggregated"/"readiness_gates.csv")
    rm=read(root/"raw"/"run_metrics.csv")
    models=sorted(rm["model"].unique()) if not rm.empty and "model" in rm else ["product","radial_jastrow","mlp_pair","kan_jastrow"]
    fair=fairness_matrix(models)
    pcnt=rm[rm["metric"]=="parameter_count"].groupby(["profile","model"],as_index=False)["value"].median().rename(columns={"value":"parameter_count"}) if not rm.empty else pd.DataFrame()
    comp=compute_budget_table(rm) if not rm.empty else pd.DataFrame()
    atomic_write_dataframe(tables/"baseline_fairness_matrix.csv", fair)
    atomic_write_dataframe(tables/"parameter_count_table.csv", pcnt)
    atomic_write_dataframe(tables/"compute_budget_table.csv", comp)
    dataframe_to_latex(edv, tables/"table_ed_validation.tex", "Hamiltonian validation checks.", "tab:ed_validation")
    dataframe_to_latex(mc, tables/"table_model_comparison.tex", "Pilot model comparison.", "tab:model_comparison")
    dataframe_to_latex(pc, tables/"table_paired_statistics.tex", "Paired pilot statistics.", "tab:paired_statistics")
    dataframe_to_latex(ab, tables/"table_ablation_summary.tex", "Ablation summary.", "tab:ablation")
    dataframe_to_latex(st, tables/"table_interpretability_metrics.tex", "Interpretability metrics.", "tab:interp")
    dataframe_to_latex(fr, tables/"table_failure_modes.tex", "Failed or unexecuted runs.", "tab:failure")
    dataframe_to_latex(gates, tables/"table_readiness_gates.tex", "Decision gate status.", "tab:gates")
    dataframe_to_latex(safe_claim_table(root), tables/"table_safe_and_unsafe_claims.tex", "Safe and unsafe claims.", "tab:claims")
    print(f"tables written to {tables}")

if __name__=="__main__":
    main()
