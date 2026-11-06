#!/usr/bin/env python3
"""Merge and byte-verify the twelve isolated Stage 4H shards."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from graphkan_amo_nqs.training.stage4h_transplant import sha256_file
from stage4h_protocol import DEFAULT_CONFIG, read_config, roster, validate_config


def verify_completion(run_root: Path) -> tuple[dict, list[str]]:
    errors: list[str] = []
    completion_path = run_root / "completion.json"
    if not completion_path.is_file():
        return {}, [f"missing completion: {run_root}"]
    completion = json.loads(completion_path.read_text())
    attempt = run_root / "attempts" / str(completion.get("attempt", ""))
    for relative, expected in completion.get("payload_hashes", {}).items():
        payload = attempt / relative
        if not payload.is_file() or sha256_file(payload) != expected:
            errors.append(f"payload hash mismatch: {run_root.name}/{relative}")
    if completion.get("status") != "COMPLETED":
        errors.append(f"nonterminal run: {run_root.name}")
    return completion, errors


def merge(input_root: Path, output: Path, cfg: dict) -> dict:
    expected = {row["run_id"]: row for row in roster(cfg)}
    summaries = list(input_root.rglob("shard_summary.json"))
    errors: list[str] = []
    observed: dict[str, Path] = {}
    for summary_path in summaries:
        summary = json.loads(summary_path.read_text())
        if summary.get("completed") != 5 or summary.get("failed") != 0:
            errors.append(f"incomplete shard: {summary.get('shard', summary_path)}")
        for completion_path in summary_path.parent.glob("runs/*/completion.json"):
            run_id = completion_path.parent.name
            if run_id in observed:
                errors.append(f"duplicate run bundle: {run_id}")
            observed[run_id] = completion_path.parent
    missing, foreign = (
        sorted(set(expected) - set(observed)),
        sorted(set(observed) - set(expected)),
    )
    if missing:
        errors.append(f"{len(missing)} expected run bundles are missing")
    if foreign:
        errors.append(f"{len(foreign)} foreign run bundles are present")

    canonical = output / "runs"
    canonical.mkdir(parents=True, exist_ok=True)
    metric_rows, sentinel_frames, region_frames, separation_frames = [], [], [], []
    transplant_rows = []
    for run_id, source in sorted(observed.items()):
        completion, run_errors = verify_completion(source)
        errors.extend(run_errors)
        if run_id not in expected or run_errors:
            continue
        target = canonical / run_id
        if target.exists():
            if sha256_file(target / "completion.json") != sha256_file(
                source / "completion.json"
            ):
                errors.append(f"canonical collision: {run_id}")
        else:
            shutil.copytree(source, target)
        attempt = target / "attempts" / completion["attempt"]
        metric_rows.append(json.loads((attempt / "metrics.json").read_text()))
        attestation = json.loads((attempt / "transplant_attestation.json").read_text())
        transplant_rows.append(attestation)
        for filename, collection in (
            ("exact_sentinels.csv", sentinel_frames),
            ("region_density.csv", region_frames),
            ("separation_correlation.csv", separation_frames),
        ):
            collection.append(pd.read_csv(attempt / filename))
    tables = output / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_rows).to_csv(tables / "metrics.csv", index=False)
    pd.DataFrame(transplant_rows).to_csv(
        tables / "transplant_attestations.csv", index=False
    )
    for name, frames in (
        ("exact_sentinels.csv", sentinel_frames),
        ("region_density.csv", region_frames),
        ("separation_correlation.csv", separation_frames),
    ):
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(tables / name, index=False)
    result = {
        "schema_version": "stage4h-merge-v1",
        "validation": "PASS" if not errors else "FAIL",
        "blocking_errors": errors,
        "shards_found": len(summaries),
        "expected_shards": 12,
        "runs_merged": len(metric_rows),
        "expected_runs": 60,
    }
    if len(summaries) != 12:
        result["blocking_errors"].append("expected exactly twelve shard summaries")
    if len(metric_rows) != 60:
        result["blocking_errors"].append("expected exactly sixty verified runs")
    result["validation"] = "PASS" if not result["blocking_errors"] else "FAIL"
    (output / "merge_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results_stage4h/merged"))
    args = parser.parse_args()
    cfg = read_config(args.config)
    if validate_config(cfg)["validation"] != "PASS":
        raise SystemExit("invalid Stage 4H config")
    result = merge(args.input, args.output, cfg)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["validation"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
