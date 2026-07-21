from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .paper_style import apply_paper_style


def _metric_df(results_root: str | Path) -> pd.DataFrame:
    files = list(Path(results_root).glob("raw/*run_metrics*.csv")) + list(Path(results_root).glob("raw/run_metrics.csv"))
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if not df.empty:
                dfs.append(df)
        except Exception:
            pass
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def reject_smoke_only(df: pd.DataFrame, filename: str, allow_smoke: bool = False) -> None:
    if allow_smoke or "smoke" in filename:
        return
    if df.empty:
        return
    profiles = set(df.get("profile", []))
    if profiles and profiles.issubset({"smoke", "quick_ci"}):
        raise ValueError("Paper figure generation rejected smoke-only data")


def fig_ed_validation(results_root: str | Path, out: str | Path):
    apply_paper_style()
    path = Path(results_root) / "references" / "ed_validation.csv"
    df = pd.read_csv(path) if path.exists() else pd.DataFrame()
    fig, ax = plt.subplots()
    if df.empty:
        ax.text(0.5, 0.5, "ED validation not run", ha="center", va="center")
    else:
        for metric, g in df.groupby("metric"):
            ax.plot(g["n_sites"], g["value"], marker="o", label=metric)
        ax.set_yscale("symlog", linthresh=1e-14)
        ax.set_xlabel("N sites")
        ax.set_ylabel("absolute discrepancy")
        ax.legend()
    ax.set_title("Hamiltonian validation")
    fig.savefig(out); plt.close(fig)


def fig_pilot_energy_correlation(results_root: str | Path, out: str | Path, allow_smoke: bool = False):
    apply_paper_style(); df = _metric_df(results_root); reject_smoke_only(df, str(out), allow_smoke)
    sub = df[df["metric"].isin(["energy_density_error", "correlation_mae"])] if not df.empty else pd.DataFrame()
    fig, ax = plt.subplots()
    if sub.empty:
        ax.text(0.5, 0.5, "Pilot metrics not available", ha="center", va="center")
    else:
        piv = sub.groupby(["model", "metric"])["value"].median().unstack("metric")
        piv.plot(kind="bar", ax=ax)
        ax.set_ylabel("median error")
        ax.set_xlabel("model")
        ax.tick_params(axis="x", rotation=30)
    ax.set_title("Pilot energy and correlation errors")
    fig.savefig(out); plt.close(fig)


def fig_kan_vs_mlp_pair_ablation(results_root: str | Path, out: str | Path):
    apply_paper_style(); path = Path(results_root) / "aggregated" / "paired_comparisons.csv"
    df = pd.read_csv(path) if path.exists() else pd.DataFrame()
    fig, ax = plt.subplots()
    sub = df[df.get("comparison", pd.Series(dtype=str)).str.contains("kan_jastrow", na=False)] if not df.empty else pd.DataFrame()
    if sub.empty:
        ax.text(0.5, 0.5, "KAN paired ablation not available", ha="center", va="center")
    else:
        labels = sub["comparison"] + "\n" + sub["metric"]
        y = np.arange(len(sub))
        ax.errorbar(sub["median_delta_baseline_minus_candidate"], y,
                    xerr=[sub["median_delta_baseline_minus_candidate"] - sub["ci_low"], sub["ci_high"] - sub["median_delta_baseline_minus_candidate"]],
                    fmt="o")
        ax.axvline(0, linestyle="--")
        ax.set_yticks(y, labels)
        ax.set_xlabel("baseline error - KAN error (positive favours KAN)")
    ax.set_title("KAN-Jastrow paired ablations")
    fig.savefig(out); plt.close(fig)


def fig_compute_pareto(results_root: str | Path, out: str | Path):
    apply_paper_style(); df = _metric_df(results_root)
    fig, ax = plt.subplots()
    if df.empty:
        ax.text(0.5, 0.5, "No compute data", ha="center", va="center")
    else:
        piv = df[df["metric"].isin(["energy_density_error", "runtime_sec", "parameter_count"])].pivot_table(index=["profile", "instance_id", "model", "seed"], columns="metric", values="value", aggfunc="first").reset_index()
        if {"energy_density_error", "runtime_sec"}.issubset(piv.columns):
            for model, g in piv.groupby("model"):
                ax.scatter(g["runtime_sec"], g["energy_density_error"], label=model, alpha=0.6)
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlabel("runtime per run (s)"); ax.set_ylabel("energy-density error")
            ax.legend(fontsize=6)
        else:
            ax.text(0.5, 0.5, "Compute metrics incomplete", ha="center", va="center")
    ax.set_title("Compute-error Pareto pilot")
    fig.savefig(out); plt.close(fig)


def fig_failure_map(results_root: str | Path, out: str | Path):
    apply_paper_style(); df = _metric_df(results_root)
    fig, ax = plt.subplots()
    sub = df[df["metric"] == "energy_density_error"] if not df.empty else pd.DataFrame()
    if sub.empty:
        ax.text(0.5, 0.5, "No failure-map data", ha="center", va="center")
    else:
        # Show best model label on Delta-C6 grid by median energy error.
        g = sub.groupby(["delta", "c6", "model"])["value"].median().reset_index()
        best = g.loc[g.groupby(["delta", "c6"])["value"].idxmin()]
        xs = sorted(best["delta"].unique()); ys = sorted(best["c6"].unique())
        label_map = {m: i for i, m in enumerate(sorted(best["model"].unique()))}
        mat = np.full((len(ys), len(xs)), np.nan)
        for _, row in best.iterrows():
            mat[ys.index(row["c6"]), xs.index(row["delta"])] = label_map[row["model"]]
        im = ax.imshow(mat, origin="lower", aspect="auto")
        ax.set_xticks(range(len(xs)), xs); ax.set_yticks(range(len(ys)), ys)
        ax.set_xlabel("Delta/Omega"); ax.set_ylabel("C6/(a^6 Omega)")
        handles = [plt.Line2D([0], [0], marker="s", linestyle="", label=f"{v}: {k}") for k, v in label_map.items()]
        ax.legend(handles=handles, fontsize=6, loc="best")
    ax.set_title("Pilot best-model/failure map")
    fig.savefig(out); plt.close(fig)


def fig_claim_ladder_decision(results_root: str | Path, out: str | Path):
    apply_paper_style(); path = Path(results_root) / "aggregated" / "readiness_gates.csv"
    df = pd.read_csv(path) if path.exists() else pd.DataFrame()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.axis("off")
    if df.empty:
        ax.text(0.5, 0.5, "Readiness gates not evaluated", ha="center", va="center")
    else:
        y = 0.9
        for _, row in df.iterrows():
            ax.text(0.02, y, f"{row['gate']}: {row['status']}", weight="bold", transform=ax.transAxes)
            ax.text(0.04, y - 0.06, str(row['evidence'])[:110], transform=ax.transAxes, fontsize=7)
            y -= 0.16
    ax.set_title("Claim ladder and decision gates")
    fig.savefig(out); plt.close(fig)


def fig_reproducibility_workflow(out: str | Path):
    apply_paper_style(); fig, ax = plt.subplots(figsize=(6, 2.2)); ax.axis("off")
    labels = ["YAML configs", "Geometry + ED", "Exact/VMC runs", "Raw CSV/NPZ", "Aggregation", "Figures/Tables", "LaTeX report"]
    xs = np.linspace(0.05, 0.95, len(labels))
    for i, (x, label) in enumerate(zip(xs, labels)):
        ax.text(x, 0.55, label, ha="center", va="center", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black"), fontsize=7)
        if i < len(labels) - 1:
            ax.annotate("", xy=(xs[i+1]-0.055, 0.55), xytext=(x+0.055, 0.55), arrowprops=dict(arrowstyle="->"))
    ax.set_title("Hardware-free reproducibility workflow")
    fig.savefig(out); plt.close(fig)
