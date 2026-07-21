from __future__ import annotations

import pandas as pd


def fairness_matrix(models: list[str]) -> pd.DataFrame:
    rows = []
    for m in models:
        rows.append({
            "model": m,
            "same_instances": True,
            "same_physical_features": m in {"mlp_pair", "kan_jastrow", "smooth_jastrow", "radial_jastrow", "graph_mpnn", "graph_kan"},
            "same_optimizer_family": True,
            "same_seed_manifest": True,
            "notes": "CPU exact full-sum pilot; full NetKet fairness requires production VMC profile." if m in {"mlp_pair", "kan_jastrow"} else "Baseline included in CPU pilot or as lower-bound sanity check.",
        })
    return pd.DataFrame(rows)
