from __future__ import annotations

from pathlib import Path

import pandas as pd

from .exceptions import SchemaError

RUN_METRICS_COLUMNS = [
    "profile", "instance_id", "geometry", "n_sites", "omega", "delta", "c6", "split",
    "model", "seed", "reference_method", "metric", "value", "run_status", "code_version",
    "config_hash", "source_file_hash", "units",
]

TRAINING_HISTORY_COLUMNS = [
    "profile", "instance_id", "model", "seed", "iteration", "energy", "gradient_norm",
    "step_time_sec", "run_status", "config_hash",
]

FAILED_RUNS_COLUMNS = [
    "profile", "instance_id", "model", "seed", "stage", "run_status", "reason",
    "config_hash", "source_file_hash",
]

ED_REFERENCE_COLUMNS = [
    "profile", "instance_id", "geometry", "n_sites", "omega", "delta", "c6", "split",
    "model", "seed", "reference_method", "metric", "value", "run_status", "code_version",
    "config_hash", "source_file_hash", "units",
]

INTERPRETABILITY_COLUMNS = [
    "profile", "instance_id", "geometry", "n_sites", "omega", "delta", "c6", "split", "model",
    "seed", "r", "rho_omega", "rho_delta", "raw_value", "gauge_fixed_value", "run_status",
    "code_version", "config_hash", "source_file_hash",
]

SCHEMAS = {
    "run_metrics": RUN_METRICS_COLUMNS,
    "training_history": TRAINING_HISTORY_COLUMNS,
    "failed_runs": FAILED_RUNS_COLUMNS,
    "ed_references": ED_REFERENCE_COLUMNS,
    "gauge_fixed_functions": INTERPRETABILITY_COLUMNS,
}


def validate_dataframe(df: pd.DataFrame, schema_name: str, allow_extra: bool = True) -> None:
    required = SCHEMAS[schema_name]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SchemaError(f"{schema_name} missing columns: {missing}")
    if not allow_extra:
        extra = [c for c in df.columns if c not in required]
        if extra:
            raise SchemaError(f"{schema_name} extra columns: {extra}")


def validate_csv(path: str | Path, schema_name: str) -> None:
    p = Path(path)
    if not p.exists():
        raise SchemaError(f"Missing CSV: {p}")
    validate_dataframe(pd.read_csv(p), schema_name)
