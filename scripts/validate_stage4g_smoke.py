#!/usr/bin/env python3
"""Validate the cheap open/periodic Stage 4G integration smoke."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def checkpoint(root: Path, run_id: str) -> Path:
    return root / "checkpoints" / f"{run_id.replace(':', '__').replace('/', '_')}.npz"


def identical_npz(left: Path, right: Path) -> bool:
    with np.load(left, allow_pickle=False) as a, np.load(right, allow_pickle=False) as b:
        return sorted(a.files) == sorted(b.files) and all(
            np.array_equal(np.asarray(a[key]), np.asarray(b[key])) for key in sorted(a.files)
        )


def main() -> None:
    errors = []
    summaries = {}
    for boundary in ("open", "periodic"):
        root = ROOT / "results_stage4g" / "smoke" / boundary
        raw = root / "raw"
        metrics = pd.read_csv(raw / "stage4g_metrics.csv")
        sentinel = pd.read_csv(raw / "stage4g_exact_sentinels.csv")
        if len(metrics) != 3 or metrics.run_id.nunique() != 3:
            errors.append(f"{boundary}: expected three unique metrics rows")
        if len(sentinel) != 6 or set(sentinel.step) != {1, 2}:
            errors.append(f"{boundary}: expected six exact sentinel rows at steps 1 and 2")
        if not (metrics.scientific_validity_exact == "VALID").all():
            errors.append(f"{boundary}: an exact smoke health gate failed")
        summaries[boundary] = {"runs": len(metrics), "sentinels": len(sentinel)}
        if boundary == "periodic":
            radial = metrics[metrics.model_label == "rbf_radial_b8"].iloc[0]
            endpoint = metrics[metrics.model_label == "endpoint_full_b8"].iloc[0]
            if not identical_npz(checkpoint(root, radial.run_id), checkpoint(root, endpoint.run_id)):
                errors.append("periodic: endpoint-full checkpoint does not collapse exactly to radial")
            for filename, columns, value in (
                ("stage4g_site_observables.csv", ["site"], "exact_model_density"),
                ("stage4g_corr_observables.csv", ["i", "j"], "exact_model_connected"),
            ):
                frame = pd.read_csv(raw / filename)
                left = frame[frame.run_id == radial.run_id].sort_values(columns)[value].to_numpy()
                right = frame[frame.run_id == endpoint.run_id].sort_values(columns)[value].to_numpy()
                if not np.array_equal(left, right):
                    errors.append(f"periodic: {value} does not collapse exactly")
    result = {
        "schema_version": "stage4g-smoke-validation-v1", "validation": "PASS" if not errors else "FAIL",
        "blocking_errors": errors, "boundaries": summaries,
    }
    output = ROOT / "results_stage4g" / "smoke" / "smoke_validation.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
