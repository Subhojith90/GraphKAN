#!/usr/bin/env python3
"""Prospectively calibrate replay tolerances from the frozen five-seed campaign."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path
from typing import Iterable


MODELS = ["rbf_radial_b32", "endpoint_full_b32", "endpoint_product_b32", "mlp_endpoint_large", "rbm_h32"]
CELLS = [("open", "1.25", "3"), ("open", "2.00", "5"), ("periodic", "1.25", "3"), ("periodic", "2.00", "5")]


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def rms(values: Iterable[float]) -> float:
    values = list(values)
    return (sum(value * value for value in values) / len(values)) ** 0.5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", default="results_stage4f_r/frozen")
    parser.add_argument("--output", default="release_integrity/stage4fr1b_replay_tolerances.json")
    args = parser.parse_args()
    campaign = Path(args.campaign)
    run_dirs = list((campaign / "runs").iterdir())
    indexed: dict[tuple[str, str, str, str, int], Path] = {}
    for run in run_dirs:
        manifest = json.loads((run / "run_manifest.json").read_text())
        indexed[(manifest["boundary"], manifest["delta"], manifest["c6"], manifest["model"], int(manifest["seed"]))] = run

    table: dict[str, dict[str, float | int | str]] = {}
    for boundary, delta, c6 in CELLS:
        for model in MODELS:
            selected = [indexed[(boundary, delta, c6, model, seed)] for seed in range(5)]
            metrics = [rows(run / "metrics.csv")[0] for run in selected]
            energies = [float(row["exact_model_energy"]) for row in metrics]
            density_vectors = [
                [float(row["exact_model_density"]) for row in sorted(rows(run / "site_observables.csv"), key=lambda x: int(x["site"]))]
                for run in selected
            ]
            corr_vectors = [
                [float(row["exact_model_connected"]) for row in sorted(rows(run / "corr_observables.csv"), key=lambda x: (int(x["i"]), int(x["j"])))]
                for run in selected
            ]
            density_errors = [
                [float(row["exact_model_abs_error"]) for row in sorted(rows(run / "site_observables.csv"), key=lambda x: int(x["site"]))]
                for run in selected
            ]
            corr_errors = [
                [float(row["exact_model_abs_error"]) for row in sorted(rows(run / "corr_observables.csv"), key=lambda x: (int(x["i"]), int(x["j"])))]
                for run in selected
            ]
            density_sd = [statistics.stdev(values) for values in zip(*density_vectors)]
            corr_sd = [statistics.stdev(values) for values in zip(*corr_vectors)]
            energy_scientific_scale = statistics.median(float(row["exact_model_abs_energy_error_to_ed"]) for row in metrics)
            density_max_scale = statistics.median(max(values) for values in density_errors)
            density_rms_scale = statistics.median(rms(values) for values in density_errors)
            corr_max_scale = statistics.median(max(values) for values in corr_errors)
            corr_rms_scale = statistics.median(rms(values) for values in corr_errors)
            identity = f"{boundary}|{delta}|{c6}|{model}"
            table[identity] = {
                "reference_seeds": 5,
                "energy_abs": max(1.0e-6, min(3.0 * statistics.stdev(energies), 0.25 * energy_scientific_scale)),
                "energy_relative": 5.0e-4,
                "density_max_abs": max(1.0e-6, min(3.0 * max(density_sd), 0.25 * density_max_scale)),
                "density_rms_abs": max(1.0e-6, min(3.0 * rms(density_sd), 0.25 * density_rms_scale)),
                "density_relative": 5.0e-3,
                "correlation_max_abs": max(1.0e-6, min(3.0 * max(corr_sd), 0.25 * corr_max_scale)),
                "correlation_rms_abs": max(1.0e-6, min(3.0 * rms(corr_sd), 0.25 * corr_rms_scale)),
                "correlation_relative": 5.0e-3,
                "calibration": "min(five-seed 3-sigma, 25% of frozen scientific-error scale) plus prospective relative component; frozen before replay",
            }
    manifest = campaign / "campaign_manifest.json"
    output = {
        "schema_version": "stage4fr1b-prospective-tolerances-v1",
        "status": "FROZEN_BEFORE_REPLAY",
        "source_campaign_protocol_hash": json.loads(manifest.read_text())["protocol_hash"],
        "source_campaign_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "periodic_collapse_absolute": 1.0e-12,
        "thresholds": table,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(destination), "rows": len(table), "status": output["status"]}, indent=2))


if __name__ == "__main__":
    main()
