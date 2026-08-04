#!/usr/bin/env python3
"""Merge isolated GitHub Actions Stage 4G shards into canonical phase tables."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = (
    "stage4g_metrics.csv",
    "stage4g_history.csv",
    "stage4g_site_observables.csv",
    "stage4g_corr_observables.csv",
    "stage4g_exact_sentinels.csv",
)


def roster_key(row) -> tuple[str, float, float, str, int]:
    boundary = str(getattr(row, "stage4g_boundary", getattr(row, "boundary", "")))
    model = str(getattr(row, "model_label", getattr(row, "model", "")))
    return boundary, float(row.delta), float(row.c6), model, int(row.seed)


def artifact_identity(path: Path, phase: str) -> tuple[str, str]:
    prefix = f"stage4g-{phase}-"
    if not path.name.startswith(prefix):
        raise ValueError(f"unexpected artifact directory: {path.name}")
    boundary, shard = path.name[len(prefix):].split("-", 1)
    if boundary not in {"open", "periodic"}:
        raise ValueError(f"unexpected boundary in {path.name}")
    return boundary, shard


def resolve_protocol_directory(results: Path, phase: str | None = None) -> Path:
    """Resolve either the amended or original frozen protocol directory."""
    candidates = (
        (results / "g2c_protocol",)
        if phase == "convergence_g2c"
        else (results / "g2b_protocol", results / "protocol")
    )
    for candidate in candidates:
        if (candidate / "preselection_roster.csv").is_file():
            return candidate
    searched = ", ".join(str(path / "preselection_roster.csv") for path in candidates)
    raise FileNotFoundError(f"no frozen Stage 4G roster found; searched: {searched}")


def merge(input_root: Path, results: Path, phase: str) -> dict:
    protocol = resolve_protocol_directory(results, phase)
    roster_path = protocol / ("confirmation_roster.csv" if phase == "confirmation" else "preselection_roster.csv")
    roster = pd.read_csv(roster_path)
    roster = roster[roster.phase == phase].copy()
    artifacts = sorted(path for path in input_root.iterdir() if path.is_dir() and path.name.startswith(f"stage4g-{phase}-"))
    errors: list[str] = []
    merged_counts: dict[str, int] = {}
    frames: dict[tuple[str, str], list[pd.DataFrame]] = {}
    for artifact in artifacts:
        boundary, shard = artifact_identity(artifact, phase)
        raw = artifact / "raw"
        for filename in TABLES:
            path = raw / filename
            if not path.is_file():
                if filename == "stage4g_exact_sentinels.csv" and not phase.startswith("convergence"):
                    continue
                errors.append(f"missing {artifact.name}/raw/{filename}")
                continue
            frame = pd.read_csv(path)
            frame["stage4g_boundary"] = boundary
            if "model_label" in frame:
                labels = set(frame.model_label.astype(str))
                if len(labels) != 1 or not shard.startswith(next(iter(labels))):
                    errors.append(f"foreign model row in {artifact.name}/{filename}")
            frames.setdefault((boundary, filename), []).append(frame)
        source_checkpoints = artifact / "checkpoints"
        target_checkpoints = results / phase / boundary / "checkpoints"
        target_checkpoints.mkdir(parents=True, exist_ok=True)
        for source in source_checkpoints.glob("*"):
            target = target_checkpoints / source.name
            if target.exists() and target.read_bytes() != source.read_bytes():
                errors.append(f"checkpoint collision with different bytes: {source.name}")
            elif not target.exists():
                shutil.copy2(source, target)
        source_environment = artifact / "environment"
        if source_environment.is_dir():
            target_environment = results / phase / boundary / "environment" / "shards" / shard
            if target_environment.exists():
                shutil.rmtree(target_environment)
            shutil.copytree(source_environment, target_environment)
    for (boundary, filename), pieces in frames.items():
        combined = pd.concat(pieces, ignore_index=True)
        output = results / phase / boundary / "raw" / filename
        output.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(output, index=False)
        merged_counts[f"{boundary}/{filename}"] = len(combined)
    metric_frames = [frame for (boundary, filename), values in frames.items() if filename == "stage4g_metrics.csv" for frame in values]
    metrics = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()
    observed = {roster_key(row) for row in metrics.itertuples(index=False)} if len(metrics) else set()
    expected = {roster_key(row) for row in roster.itertuples(index=False)}
    missing = sorted(expected - observed)
    foreign = sorted(observed - expected)
    if missing:
        errors.append(f"{len(missing)} roster rows are missing")
    if foreign:
        errors.append(f"{len(foreign)} foreign rows are present")
    if len(metrics) and metrics.run_id.nunique() != len(metrics):
        errors.append("duplicate run IDs are present")
    summary = {
        "schema_version": "stage4g-shard-merge-v1", "phase": phase,
        "validation": "PASS" if not errors else "FAIL", "blocking_errors": errors,
        "artifacts": len(artifacts), "expected_runs": len(expected), "merged_runs": len(metrics),
        "merged_table_rows": merged_counts,
    }
    path = results / phase / "merge_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--results", type=Path, default=ROOT / "results_stage4g")
    parser.add_argument("--phase", choices=("convergence", "convergence_g2b", "convergence_g2c", "calibration", "confirmation"), required=True)
    args = parser.parse_args()
    result = merge(args.input, args.results, args.phase)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["validation"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
