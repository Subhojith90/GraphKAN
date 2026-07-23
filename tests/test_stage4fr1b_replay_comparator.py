from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

from compare_stage4fr1b_replay import compare_bundle


def reference_run() -> Path:
    return next(Path("results_stage4f_r/frozen/runs").glob("*__Bopen__D1p25__C3__Mendpoint_full_b32__S0"))


def permissive_for_identical_rows() -> dict[str, float]:
    return {
        "energy_abs": 1.0e-6,
        "energy_relative": 0.0,
        "density_max_abs": 1.0e-6,
        "density_rms_abs": 1.0e-6,
        "density_relative": 0.0,
        "correlation_max_abs": 1.0e-6,
        "correlation_rms_abs": 1.0e-6,
        "correlation_relative": 0.0,
    }


def test_hardened_comparator_accepts_unchanged_bundle(tmp_path):
    reference = reference_run()
    actual = tmp_path / "actual"
    shutil.copytree(reference, actual)
    assert compare_bundle(actual, reference, permissive_for_identical_rows())["pass"]


def test_hardened_comparator_rejects_vector_drift(tmp_path):
    reference = reference_run()
    actual = tmp_path / "actual"
    shutil.copytree(reference, actual)
    path = actual / "site_observables.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0])
    rows[0]["exact_model_density"] = str(float(rows[0]["exact_model_density"]) + 0.01)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    result = compare_bundle(actual, reference, permissive_for_identical_rows())
    assert not result["pass"]
    assert not result["site_density_vector"]["pass"]


def test_hardened_comparator_rejects_checkpoint_semantic_drift(tmp_path):
    reference = reference_run()
    actual = tmp_path / "actual"
    shutil.copytree(reference, actual)
    path = actual / "checkpoint_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["array_count"] += 1
    path.write_text(json.dumps(manifest) + "\n")
    result = compare_bundle(actual, reference, permissive_for_identical_rows())
    assert not result["pass"]
    assert not result["checkpoint"]["pass"]
