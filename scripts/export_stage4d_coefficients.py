from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    return parser.parse_args()


def parse_leaf_names(treedef_repr: str, leaf_count: int) -> list[str]:
    match = re.search(r"PyTreeDef\(\{(.*)\}\)", treedef_repr)
    if not match:
        return [f"leaf_{i:04d}" for i in range(leaf_count)]

    raw_keys = re.findall(r"'([^']+)'\s*:", match.group(1))
    keys = sorted(raw_keys)

    if len(keys) != leaf_count:
        return [f"leaf_{i:04d}" for i in range(leaf_count)]

    return keys


def main() -> None:
    args = parse_args()
    root = Path(args.results)
    checkpoint_dir = root / "checkpoints"
    metrics_path = root / "raw" / "stage4d_metrics.csv"
    output_dir = root / "aggregated"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint_dir.exists():
        raise SystemExit(f"Checkpoint directory not found: {checkpoint_dir}")
    if not metrics_path.exists():
        raise SystemExit(f"Metrics file not found: {metrics_path}")

    metrics = pd.read_csv(metrics_path)
    metric_cols = [
        "run_id",
        "model_label",
        "model_kind",
        "seed",
        "n_sites",
        "delta",
        "c6",
        "scientific_validity_exact",
        "scientific_validity_sample",
        "exact_model_abs_energy_error_to_ed",
        "exact_model_density_mae_to_ed",
        "exact_model_connected_correlation_mae_to_ed",
    ]
    metrics = metrics[[c for c in metric_cols if c in metrics.columns]]

    rows: list[dict] = []

    for json_path in sorted(checkpoint_dir.glob("*.json")):
        meta = json.loads(json_path.read_text())
        npz_path = json_path.with_suffix(".npz")

        if not npz_path.exists():
            print(f"WARNING: Missing NPZ for {json_path.name}")
            continue

        leaf_names = parse_leaf_names(
            str(meta.get("treedef_repr", "")),
            int(meta.get("leaf_count", 0)),
        )

        arrays = np.load(npz_path, allow_pickle=False)
        run_id = meta.get("run_id", "")
        metric_match = metrics[metrics["run_id"] == run_id]

        metric_data = (
            metric_match.iloc[0].to_dict()
            if len(metric_match) == 1
            else {}
        )

        for leaf_index, leaf_key in enumerate(sorted(arrays.files)):
            values = np.asarray(arrays[leaf_key]).reshape(-1)
            group = (
                leaf_names[leaf_index]
                if leaf_index < len(leaf_names)
                else leaf_key
            )

            for coefficient_index, value in enumerate(values):
                rows.append(
                    {
                        "run_id": run_id,
                        "checkpoint_json": str(json_path),
                        "checkpoint_npz": str(npz_path),
                        "profile": meta.get("profile"),
                        "stage": meta.get("stage"),
                        "model_label": meta.get("model_label"),
                        "model_kind": meta.get("model_kind"),
                        "basis_family": meta.get("basis_family"),
                        "seed": meta.get("seed"),
                        "n_parameters": meta.get("n_parameters"),
                        "parameter_group": group,
                        "leaf_index": leaf_index,
                        "coefficient_index": coefficient_index,
                        "value": float(value),
                        **metric_data,
                    }
                )

    if not rows:
        raise SystemExit("No coefficients exported.")

    long_df = pd.DataFrame(rows)

    keep_first = [
        "run_id",
        "checkpoint_json",
        "checkpoint_npz",
        "profile",
        "stage",
        "model_label",
        "model_kind",
        "basis_family",
        "seed",
        "n_sites",
        "delta",
        "c6",
        "scientific_validity_exact",
        "scientific_validity_sample",
        "exact_model_abs_energy_error_to_ed",
        "exact_model_density_mae_to_ed",
        "exact_model_connected_correlation_mae_to_ed",
        "n_parameters",
        "parameter_group",
        "leaf_index",
        "coefficient_index",
        "value",
    ]
    long_df = long_df[[c for c in keep_first if c in long_df.columns]]

    long_path = output_dir / "stage4d_coefficients_long.csv"
    long_df.to_csv(long_path, index=False)

    summary = (
        long_df.groupby(
            [
                "model_label",
                "n_sites",
                "delta",
                "c6",
                "parameter_group",
                "coefficient_index",
            ],
            dropna=False,
            as_index=False,
        )["value"]
        .agg(["count", "median", "mean", "std", "min", "max"])
        .reset_index()
    )

    summary_path = output_dir / "stage4d_coefficients_summary.csv"
    summary.to_csv(summary_path, index=False)

    manifest = {
        "schema": "stage4d-coefficient-export-v1",
        "results_root": str(root),
        "checkpoint_count": len(list(checkpoint_dir.glob("*.json"))),
        "coefficient_rows": int(len(long_df)),
        "long_csv": str(long_path),
        "summary_csv": str(summary_path),
    }

    manifest_path = output_dir / "stage4d_coefficients_export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
