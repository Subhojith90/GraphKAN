#!/usr/bin/env python3
"""Hardened scalar, vector, health, checkpoint and provenance replay gate."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml


HEALTH_FIELDS = [
    "status", "scientific_validity", "scientific_validity_exact", "scientific_validity_sample",
    "exact_health_flags", "sample_health_flags", "health_flags",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def metric(run: Path) -> dict[str, str]:
    return csv_rows(run / "metrics.csv")[0]


def identity(run: Path) -> tuple[str, str, str, str, int]:
    data = json.loads((run / "run_manifest.json").read_text())
    return data["boundary"], f"{float(data['delta']):.2f}", str(int(float(data["c6"]))), data["model"], int(data["seed"])


def vector(run: Path, filename: str, column: str, sort_columns: tuple[str, ...]) -> np.ndarray:
    rows = sorted(csv_rows(run / filename), key=lambda row: tuple(int(row[item]) for item in sort_columns))
    return np.asarray([float(row[column]) for row in rows], dtype=float)


def difference_gate(actual: np.ndarray, reference: np.ndarray, max_abs: float, rms_abs: float, relative: float) -> dict[str, Any]:
    if actual.shape != reference.shape:
        return {"pass": False, "shape_actual": list(actual.shape), "shape_reference": list(reference.shape)}
    delta = actual - reference
    max_difference = float(np.max(np.abs(delta)))
    rms_difference = float(np.sqrt(np.mean(delta * delta)))
    max_limit = float(max_abs + relative * max(1.0e-15, np.max(np.abs(reference))))
    rms_limit = float(rms_abs + relative * max(1.0e-15, np.sqrt(np.mean(reference * reference))))
    return {
        "pass": max_difference <= max_limit and rms_difference <= rms_limit,
        "max_difference": max_difference,
        "rms_difference": rms_difference,
        "max_limit": max_limit,
        "rms_limit": rms_limit,
        "components": int(actual.size),
    }


def checkpoint_gate(actual_run: Path, reference_run: Path) -> dict[str, Any]:
    actual = json.loads((actual_run / "checkpoint_manifest.json").read_text())
    reference = json.loads((reference_run / "checkpoint_manifest.json").read_text())
    fields = ("schema_version", "model", "array_count", "arrays")
    equality = {field: actual.get(field) == reference.get(field) for field in fields}
    checkpoint = actual_run / actual.get("checkpoint", "checkpoint.npz")
    readable = False
    hash_valid = checkpoint.is_file() and sha256(checkpoint) == actual.get("sha256")
    if hash_valid:
        try:
            with np.load(checkpoint, allow_pickle=False) as arrays:
                readable = sorted(arrays.files) == sorted(actual.get("arrays", {}))
        except Exception:
            readable = False
    return {"pass": all(equality.values()) and hash_valid and readable, "field_equality": equality, "hash_valid": hash_valid, "readable": readable}


def compare_bundle(actual_run: Path, reference_run: Path, limits: dict[str, float]) -> dict[str, Any]:
    actual_metric = metric(actual_run)
    reference_metric = metric(reference_run)
    energy_difference = abs(float(actual_metric["exact_model_energy"]) - float(reference_metric["exact_model_energy"]))
    energy_limit = float(limits["energy_abs"] + limits["energy_relative"] * abs(float(reference_metric["exact_model_energy"])))
    density = difference_gate(
        vector(actual_run, "site_observables.csv", "exact_model_density", ("site",)),
        vector(reference_run, "site_observables.csv", "exact_model_density", ("site",)),
        float(limits["density_max_abs"]), float(limits["density_rms_abs"]), float(limits["density_relative"]),
    )
    correlation = difference_gate(
        vector(actual_run, "corr_observables.csv", "exact_model_connected", ("i", "j")),
        vector(reference_run, "corr_observables.csv", "exact_model_connected", ("i", "j")),
        float(limits["correlation_max_abs"]), float(limits["correlation_rms_abs"]), float(limits["correlation_relative"]),
    )
    health = {field: actual_metric.get(field, "") == reference_metric.get(field, "") for field in HEALTH_FIELDS}
    exact_metadata = {
        field: actual_metric.get(field, "") == reference_metric.get(field, "")
        for field in ("n_parameters", "model_kind", "basis_family", "n_sites", "n_samples", "diag_shift")
    }
    checkpoint = checkpoint_gate(actual_run, reference_run)
    passed = energy_difference <= energy_limit and density["pass"] and correlation["pass"] and all(health.values()) and all(exact_metadata.values()) and checkpoint["pass"]
    return {
        "pass": passed,
        "exact_model_energy": {"difference": energy_difference, "limit": energy_limit, "pass": energy_difference <= energy_limit},
        "site_density_vector": density,
        "connected_correlation_matrix": correlation,
        "health_field_equality": health,
        "metadata_equality": exact_metadata,
        "checkpoint": checkpoint,
    }


def periodic_collapse(replay: dict[tuple[str, str, str, str, int], Path], delta: str, c6: str, limit: float) -> list[dict[str, Any]]:
    radial = replay[("periodic", delta, c6, "rbf_radial_b32", 0)]
    results: list[dict[str, Any]] = []
    for model in ("endpoint_full_b32", "endpoint_product_b32"):
        endpoint = replay[("periodic", delta, c6, model, 0)]
        energy = abs(float(metric(radial)["exact_model_energy"]) - float(metric(endpoint)["exact_model_energy"]))
        density = difference_gate(
            vector(endpoint, "site_observables.csv", "exact_model_density", ("site",)),
            vector(radial, "site_observables.csv", "exact_model_density", ("site",)), limit, limit, 0.0,
        )
        correlation = difference_gate(
            vector(endpoint, "corr_observables.csv", "exact_model_connected", ("i", "j")),
            vector(radial, "corr_observables.csv", "exact_model_connected", ("i", "j")), limit, limit, 0.0,
        )
        results.append({"delta": delta, "c6": c6, "model": model, "energy_difference": energy, "density": density, "correlation": correlation, "pass": energy <= limit and density["pass"] and correlation["pass"]})
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", default="results_stage4fr1b/independent_replay")
    parser.add_argument("--reference", default="results_stage4f_r/frozen")
    parser.add_argument("--config", default="configs/stage4fr1b_second_host_replay.yaml")
    parser.add_argument("--tolerances", default="release_integrity/stage4fr1b_replay_tolerances.json")
    parser.add_argument("--expected-commit")
    parser.add_argument("--output", default="results_stage4fr1b/independent_replay/replay_comparison.json")
    args = parser.parse_args()
    replay_root, reference_root = Path(args.replay), Path(args.reference)
    config_path, tolerance_path = Path(args.config), Path(args.tolerances)
    config = yaml.safe_load(config_path.read_text())
    tolerances = json.loads(tolerance_path.read_text())
    errors: list[str] = []

    reference_manifest = json.loads((reference_root / "campaign_manifest.json").read_text())
    replay_manifest = json.loads((replay_root / "campaign_manifest.json").read_text())
    required = config["completion_requirements"]
    if reference_manifest["protocol_hash"] != required["replay_source_protocol_hash"]:
        errors.append("reference campaign protocol hash does not match preregistration")
    if sha256(reference_root / "campaign_manifest.json") != tolerances["source_campaign_manifest_sha256"]:
        errors.append("tolerance calibration is not bound to this reference campaign manifest")
    if replay_manifest["config_sha256"] != sha256(config_path):
        errors.append("replay campaign config hash mismatch")
    campaign_ledger = json.loads(Path(required["campaign_source_ledger"]).read_text())["source_ledger"]
    if campaign_ledger != reference_manifest["protocol_identity"]["source_ledger"]:
        errors.append("campaign source ledger identity mismatch")
    if args.expected_commit:
        commit_file = replay_root / "environment/git_commit.txt"
        if not commit_file.is_file() or commit_file.read_text().strip() != args.expected_commit:
            errors.append("independent host commit attestation mismatch")
        clean_file = replay_root / "environment/git_status_porcelain.txt"
        if not clean_file.is_file() or clean_file.read_text().strip():
            errors.append("independent host checkout was not clean before replay")

    replay = {identity(run): run for run in (replay_root / "runs").iterdir() if run.is_dir()}
    reference = {identity(run): run for run in (reference_root / "runs").iterdir() if run.is_dir()}
    if len(replay) != int(required["expected_runs"]):
        errors.append(f"expected {required['expected_runs']} replay rows, found {len(replay)}")
    comparisons: list[dict[str, Any]] = []
    for run_identity, actual_run in sorted(replay.items()):
        reference_run = reference.get(run_identity)
        if reference_run is None:
            errors.append(f"missing frozen reference: {run_identity}")
            continue
        run_manifest = json.loads((actual_run / "run_manifest.json").read_text())
        if run_manifest["protocol_hash"] != replay_manifest["protocol_hash"]:
            errors.append(f"replay run protocol mismatch: {run_identity}")
        tolerance_key = "|".join(map(str, run_identity[:4]))
        if tolerance_key not in tolerances["thresholds"]:
            errors.append(f"missing prospective tolerance: {tolerance_key}")
            continue
        result = compare_bundle(actual_run, reference_run, tolerances["thresholds"][tolerance_key])
        result["identity"] = run_identity
        comparisons.append(result)
        if not result["pass"]:
            errors.append(f"observable/checkpoint comparison failed: {run_identity}")

    collapse: list[dict[str, Any]] = []
    try:
        for delta, c6 in (("1.25", "3"), ("2.00", "5")):
            collapse.extend(periodic_collapse(replay, delta, c6, float(tolerances["periodic_collapse_absolute"])))
    except KeyError as exc:
        errors.append(f"missing periodic collapse member: {exc}")
    if any(not item["pass"] for item in collapse):
        errors.append("periodic collapse invariant failed")

    output = {
        "schema_version": "stage4fr1b-replay-comparison-v1",
        "validation": "PASS" if not errors else "FAIL",
        "replay_rows": len(replay),
        "reference_protocol_hash": reference_manifest["protocol_hash"],
        "replay_protocol_hash": replay_manifest["protocol_hash"],
        "config_sha256": sha256(config_path),
        "tolerance_table_sha256": sha256(tolerance_path),
        "comparisons": comparisons,
        "periodic_collapse_checks": collapse,
        "blocking_errors": errors,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

