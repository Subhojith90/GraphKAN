from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("results_stage4e/n10_capacity")
RAW = ROOT / "raw" / "stage4e_metrics.csv"
OUT = ROOT / "aggregated"
OUT.mkdir(parents=True, exist_ok=True)

METRICS = [
    "exact_model_abs_energy_error_to_ed",
    "exact_model_density_mae_to_ed",
    "exact_model_connected_correlation_mae_to_ed",
    "wall_seconds",
]

CANDIDATES = [
    "endpoint_sum_b16",
    "endpoint_sum_b24",
    "endpoint_sum_b32",
    "endpoint_product_b16",
    "endpoint_product_b24",
    "endpoint_product_b32",
]

BASELINES = [
    "mlp_endpoint_medium",
    "mlp_endpoint_large",
]

df = pd.read_csv(RAW)

valid = df[
    (df["status"] == "COMPLETED")
    & (df["scientific_validity_exact"] == "VALID")
    & (df["scientific_validity_sample"] == "VALID")
].copy()

rows = []

for candidate in CANDIDATES:
    cand = valid[valid["model_label"] == candidate].set_index("seed")

    for baseline in BASELINES:
        base = valid[valid["model_label"] == baseline].set_index("seed")

        common = sorted(set(cand.index).intersection(base.index))
        if not common:
            continue

        c = cand.loc[common]
        b = base.loc[common]

        delta_energy = (
            c["exact_model_abs_energy_error_to_ed"]
            - b["exact_model_abs_energy_error_to_ed"]
        )
        delta_density = (
            c["exact_model_density_mae_to_ed"]
            - b["exact_model_density_mae_to_ed"]
        )
        delta_corr = (
            c["exact_model_connected_correlation_mae_to_ed"]
            - b["exact_model_connected_correlation_mae_to_ed"]
        )
        runtime_ratio = c["wall_seconds"] / b["wall_seconds"]

        candidate_better_density_all = bool((delta_density < 0.0).all())
        candidate_better_energy_all = bool((delta_energy < 0.0).all())
        candidate_better_corr_all = bool((delta_corr < 0.0).all())

        baseline_dominates = bool(
            (delta_energy >= 0.0).all()
            and (delta_density >= 0.0).all()
            and (delta_corr >= 0.0).all()
            and (
                (delta_energy > 0.0).any()
                or (delta_density > 0.0).any()
                or (delta_corr > 0.0).any()
            )
            and (runtime_ratio >= 1.0).all()
        )

        rows.append(
            {
                "candidate": candidate,
                "baseline": baseline,
                "paired_seeds": len(common),
                "median_delta_energy": float(np.median(delta_energy)),
                "median_delta_density": float(np.median(delta_density)),
                "median_delta_corr": float(np.median(delta_corr)),
                "median_runtime_ratio_candidate_over_baseline": float(
                    np.median(runtime_ratio)
                ),
                "candidate_better_energy_all_seeds": candidate_better_energy_all,
                "candidate_better_density_all_seeds": candidate_better_density_all,
                "candidate_better_corr_all_seeds": candidate_better_corr_all,
                "baseline_dominates_candidate": baseline_dominates,
                "candidate_pareto_non_dominated": not baseline_dominates,
            }
        )

out = pd.DataFrame(rows)
csv_path = OUT / "stage4e_paired_pareto_gate.csv"
out.to_csv(csv_path, index=False)

summary = {
    "schema": "stage4e-paired-pareto-gate-v1",
    "source": str(RAW),
    "candidate_models": CANDIDATES,
    "baseline_models": BASELINES,
    "valid_rows_used": int(len(valid)),
    "output_csv": str(csv_path),
}

json_path = OUT / "stage4e_paired_pareto_gate.json"
json_path.write_text(json.dumps(summary, indent=2) + "\n")

print(out.to_string(index=False))
print()
print(f"Wrote {csv_path}")
print(f"Wrote {json_path}")
