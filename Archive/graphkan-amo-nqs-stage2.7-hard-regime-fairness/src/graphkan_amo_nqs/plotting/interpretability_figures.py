from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from .paper_style import apply_paper_style


def fig_gauge_fixed_radial_functions(results_root: str | Path, out: str | Path):
    apply_paper_style(); path = Path(results_root) / "interpretability" / "gauge_fixed_functions.csv"
    df = pd.read_csv(path) if path.exists() else pd.DataFrame()
    fig, ax = plt.subplots()
    if df.empty:
        ax.text(0.5, 0.5, "Gauge-fixed functions not available", ha="center", va="center")
    else:
        sub = df[df["model"] == "kan_jastrow"]
        # Plot up to one representative instance per delta.
        for (delta, seed), g in sub.groupby(["delta", "seed"]):
            if len(ax.lines) >= 15:
                break
            ax.plot(g["r"], g["gauge_fixed_value"], alpha=0.55, label=f"D={delta}, s={seed}")
        ax.set_xlabel("distance r/a")
        ax.set_ylabel("gauge-fixed f(r)")
        ax.legend(fontsize=5, ncol=2)
    ax.set_title("Gauge-fixed KAN radial functions")
    fig.savefig(out); plt.close(fig)


def fig_scale_collapse(results_root: str | Path, out: str | Path):
    apply_paper_style(); path = Path(results_root) / "interpretability" / "gauge_fixed_functions.csv"
    df = pd.read_csv(path) if path.exists() else pd.DataFrame()
    fig, ax = plt.subplots()
    if df.empty:
        ax.text(0.5, 0.5, "Scale-collapse data not available", ha="center", va="center")
    else:
        sub = df[df["model"] == "kan_jastrow"]
        for (c6, delta), g in sub.groupby(["c6", "delta"]):
            gmean = g.groupby("rho_omega", as_index=False)["gauge_fixed_value"].mean().sort_values("rho_omega")
            if len(ax.lines) >= 12:
                break
            ax.plot(gmean["rho_omega"], gmean["gauge_fixed_value"], label=f"C6={c6}, D={delta}")
        ax.set_xlabel("r / R_Omega")
        ax.set_ylabel("mean gauge-fixed f")
        ax.legend(fontsize=5, ncol=2)
    ax.set_title("Pilot scale-collapse diagnostic")
    fig.savefig(out); plt.close(fig)
