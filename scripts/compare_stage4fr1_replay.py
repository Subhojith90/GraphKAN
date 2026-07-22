#!/usr/bin/env python3
"""Compare the preregistered Stage 4F-R.1 replay with the frozen campaign."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import yaml


ENERGY = "energy_mean"
ENERGY_ERROR = "energy_error"
DENSITY = "exact_model_density_mae_to_ed"
CORRELATION = "exact_model_connected_correlation_mae_to_ed"
EXACT_ENERGY = "exact_model_energy"


def read_metric_rows(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted((root / "runs").glob("*/metrics.csv")):
        with path.open(newline="") as handle:
            row = next(csv.DictReader(handle))
        row["_metrics_path"] = str(path)
        rows.append(row)
    return rows


def key(row: dict[str, str]) -> tuple[str, str, str, str, int]:
    return (
        row.get("boundary", row.get("boundary_condition", "")),
        f"{float(row['delta']):.2f}",
        str(int(float(row["c6"]))),
        row.get("model", row.get("model_label", "")),
        int(row["seed"]),
    )


def f(row: dict[str, str], column: str) -> float:
    return float(row[column])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", default="results_stage4fr1/independent_replay")
    parser.add_argument("--reference", default="results_stage4f_r/frozen")
    parser.add_argument("--config", default="configs/stage4fr1_second_host_replay.yaml")
    parser.add_argument("--output", default="results_stage4fr1/independent_replay/replay_comparison.json")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    limits = config["completion_requirements"]["comparison_tolerances"]
    replay_rows = read_metric_rows(Path(args.replay))
    reference = {key(row): row for row in read_metric_rows(Path(args.reference))}
    errors: list[str] = []
    comparisons: list[dict[str, Any]] = []

    if len(replay_rows) != int(config["completion_requirements"]["expected_runs"]):
        errors.append(f"expected 16 replay rows, found {len(replay_rows)}")

    for replay in replay_rows:
        identity = key(replay)
        original = reference.get(identity)
        if original is None:
            errors.append(f"no frozen reference for {identity}")
            continue
        combined_se = math.hypot(f(replay, ENERGY_ERROR), f(original, ENERGY_ERROR))
        energy_limit = max(
            float(limits["energy_absolute_floor"]),
            float(limits["energy_combined_standard_errors"]) * combined_se,
        )
        deltas = {
            "energy_mean": abs(f(replay, ENERGY) - f(original, ENERGY)),
            "exact_density_mae": abs(f(replay, DENSITY) - f(original, DENSITY)),
            "exact_connected_correlation_mae": abs(f(replay, CORRELATION) - f(original, CORRELATION)),
        }
        passed = (
            deltas["energy_mean"] <= energy_limit
            and deltas["exact_density_mae"] <= float(limits["exact_density_mae_absolute_difference"])
            and deltas["exact_connected_correlation_mae"] <= float(limits["exact_connected_correlation_mae_absolute_difference"])
        )
        if not passed:
            errors.append(f"replay tolerance failure for {identity}")
        comparisons.append(
            {
                "identity": identity,
                "deltas": deltas,
                "energy_limit": energy_limit,
                "pass": passed,
            }
        )

    replay_by_key = {key(row): row for row in replay_rows}
    periodic_checks: list[dict[str, Any]] = []
    periodic_limit = float(limits["periodic_radial_endpoint_exact_difference"])
    for delta, c6 in (("1.25", "3"), ("2.00", "5")):
        radial = replay_by_key.get(("periodic", delta, c6, "rbf_radial_b32", 0))
        endpoint = replay_by_key.get(("periodic", delta, c6, "endpoint_full_b32", 0))
        if radial is None or endpoint is None:
            errors.append(f"missing periodic collapse pair for {(delta, c6)}")
            continue
        differences = {
            column: abs(f(radial, column) - f(endpoint, column))
            for column in (EXACT_ENERGY, DENSITY, CORRELATION)
        }
        passed = all(value <= periodic_limit for value in differences.values())
        if not passed:
            errors.append(f"periodic radial/endpoint collapse failed for {(delta, c6)}")
        periodic_checks.append({"delta": delta, "c6": c6, "differences": differences, "pass": passed})

    output = {
        "schema_version": "stage4fr1-replay-comparison-v1",
        "validation": "PASS" if not errors else "FAIL",
        "replay_rows": len(replay_rows),
        "comparisons": comparisons,
        "periodic_collapse_checks": periodic_checks,
        "blocking_errors": errors,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

