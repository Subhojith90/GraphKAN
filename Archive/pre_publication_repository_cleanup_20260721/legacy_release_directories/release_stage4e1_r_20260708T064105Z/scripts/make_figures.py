#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
from graphkan_amo_nqs.config import load_config
from graphkan_amo_nqs.plotting.main_figures import fig_ed_validation, fig_pilot_energy_correlation, fig_kan_vs_mlp_pair_ablation, fig_compute_pareto, fig_failure_map, fig_claim_ladder_decision, fig_reproducibility_workflow
from graphkan_amo_nqs.plotting.interpretability_figures import fig_gauge_fixed_radial_functions, fig_scale_collapse


def main():
    ap=argparse.ArgumentParser(description="Generate all report figures from source CSV/NPZ results.")
    ap.add_argument("--config", required=True)
    args=ap.parse_args(); cfg=load_config(args.config)
    root=Path(cfg.get("results_root","results")); figs=Path(cfg.get("figures_root","figures")); figs.mkdir(exist_ok=True)
    allow=bool(cfg.get("allow_smoke_for_paper", False))
    fig_ed_validation(root, figs/"fig_ed_validation.pdf")
    fig_pilot_energy_correlation(root, figs/"fig_pilot_energy_correlation.pdf", allow_smoke=allow)
    fig_kan_vs_mlp_pair_ablation(root, figs/"fig_kan_vs_mlp_pair_ablation.pdf")
    fig_gauge_fixed_radial_functions(root, figs/"fig_gauge_fixed_radial_functions.pdf")
    fig_scale_collapse(root, figs/"fig_scale_collapse.pdf")
    # Geometry transfer is not run; create an honest status figure from manifest status.
    import matplotlib.pyplot as plt
    from graphkan_amo_nqs.plotting.paper_style import apply_paper_style
    apply_paper_style(); fig, ax=plt.subplots(); ax.axis('off'); ax.text(0.5,0.5,'Geometry-transfer profile not executed in CPU artifact.\nRun scripts/run_geometry_transfer.py after NetKet/Flax profile passes.',ha='center',va='center'); fig.savefig(figs/"fig_geometry_transfer.pdf"); plt.close(fig)
    fig_compute_pareto(root, figs/"fig_compute_pareto.pdf")
    fig_failure_map(root, figs/"fig_failure_map.pdf")
    fig_claim_ladder_decision(root, figs/"fig_claim_ladder_decision.pdf")
    fig_reproducibility_workflow(figs/"fig_reproducibility_workflow.pdf")
    print(f"figures written to {figs}")

if __name__=="__main__":
    main()
