from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from stage4f1b_tx import (
    SCHEMA_VERSION,
    RunSpec,
    Stage4F1BRegistry,
    atomic_promote_no_overwrite,
    hash_artifacts_before_complete,
    load_config,
    safe_id,
    sha256_bytes,
    sha256_file,
    token_hash,
    utc_now,
    validate_run_bundle_semantic,
    write_json,
    write_valid_synthetic_bundle,
)


def row_to_runspec(row: dict[str, Any]) -> RunSpec:
    expected = row.get("expected_spec") or json.loads(row["expected_spec_json"])
    return RunSpec(
        campaign_id=expected["campaign_id"],
        protocol_hash=expected["protocol_hash"],
        boundary=expected["boundary"],
        delta=expected["delta"],
        c6=expected["c6"],
        model=expected["model"],
        seed=int(expected["seed"]),
        lane=expected["lane"],
        priority=int(expected["priority"]),
        n_sites=int(expected["n_sites"]),
        expected_history_rows=int(expected["expected_history_rows"]),
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty {path.name}")
    fields = columns or list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _resolved_runner_config(cfg: dict[str, Any], run: RunSpec, output_root: Path, attempt_id: str) -> dict[str, Any]:
    base_rel = cfg["base_configs"][run.boundary]
    base = load_config(Path(base_rel))
    if run.model not in base.get("model_specs", {}):
        raise RuntimeError(f"model {run.model} is absent from {base_rel}")
    base["profile"] = f"stage4f1c__{safe_id(run.run_id)}"
    base["stage"] = "stage4f1c_real_runner"
    base["output_root"] = str(output_root)
    base["parameters"] = {"omega": [float(cfg["physics"]["omega"])], "delta": [float(run.delta)], "c6": [float(run.c6)]}
    base["training"] = dict(base["training"])
    base["training"].update(cfg.get("runner_overrides", {}).get("training", {}))
    base["training"]["seeds"] = [run.seed]
    base["model_specs"] = {run.model: base["model_specs"][run.model]}
    base["geometry"] = {
        "family": "periodic_chain" if run.boundary == "periodic" else "chain",
        "n_sites": run.n_sites,
        "spacing": 1.0,
    }
    base["transaction_identity"] = {**run.to_dict(), "attempt_id": attempt_id}
    return base


def _run_with_heartbeat(
    command: list[str],
    cwd: Path,
    reg: Stage4F1BRegistry,
    row: dict[str, Any],
    worker_id: str,
    lease_seconds: int,
    fault_during_training: bool = False,
    fault_delay: float = 2.0,
) -> tuple[int, str, str]:
    process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    interval = max(0.25, min(30.0, lease_seconds / 3.0, fault_delay if fault_during_training else 30.0))
    stdout = ""
    stderr = ""
    while True:
        try:
            stdout, stderr = process.communicate(timeout=interval)
            break
        except subprocess.TimeoutExpired:
            if fault_during_training:
                import signal
                process.kill()
                os.kill(os.getpid(), signal.SIGKILL)
            if not reg.mark_heartbeat(
                row["run_id"], worker_id, row["attempt_id"], row["claim_token"], lease_seconds
            ):
                process.terminate()
                try:
                    stdout, stderr = process.communicate(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    stdout, stderr = process.communicate()
                raise RuntimeError("claim lost while real runner was active; subprocess terminated")
    return process.returncode, stdout, stderr


def _checkpoint_manifest(path: Path, run: RunSpec) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: list(np.asarray(archive[name]).shape) for name in archive.files}
        if not arrays:
            raise RuntimeError("production checkpoint contains no arrays")
        for name in archive.files:
            value = np.asarray(archive[name])
            if value.dtype.kind not in "biufc" or not np.all(np.isfinite(value)):
                raise RuntimeError(f"production checkpoint array {name} is not finite numeric data")
    return {
        "schema_version": "jax-tree-leaves-npz-v1",
        "run_id": run.run_id,
        "model": run.model,
        "checkpoint": "checkpoint.npz",
        "sha256": sha256_file(path),
        "array_count": len(arrays),
        "arrays": arrays,
    }


def write_real_bundle(
    campaign_root: Path,
    cfg: dict[str, Any],
    run: RunSpec,
    row: dict[str, Any],
    reg: Stage4F1BRegistry,
    worker_id: str,
    lease_seconds: int,
    fault_phase: str | None = None,
    fault_delay: float = 2.0,
) -> Path:
    attempts = campaign_root / "attempts" / safe_id(run.run_id)
    attempts.mkdir(parents=True, exist_ok=True)
    bundle = Path(tempfile.mkdtemp(prefix=f"{row['attempt_id']}__", dir=attempts))
    runner_output = bundle / "_runner_output"
    resolved = _resolved_runner_config(cfg, run, runner_output, row["attempt_id"])
    runner_cfg = bundle / "_runner_config.yaml"
    runner_cfg.write_text(yaml.safe_dump(resolved, sort_keys=False))
    command = [sys.executable, "scripts/run_stage4e.py", "--config", str(runner_cfg), "--overwrite"]
    write_json(bundle / "command.json", {
        "argv": command,
        "cwd": str(Path.cwd().resolve()),
        "worker_id": worker_id,
        "run_id": run.run_id,
        "attempt_id": row["attempt_id"],
        "started_utc": utc_now(),
    })
    started = time.monotonic()
    exit_code, stdout, stderr = _run_with_heartbeat(
        command,
        Path.cwd(),
        reg,
        row,
        worker_id,
        lease_seconds,
        fault_during_training=fault_phase == "during-training",
        fault_delay=fault_delay,
    )
    (bundle / "stdout.log").write_text(stdout)
    (bundle / "stderr.log").write_text(stderr)
    if exit_code != 0:
        raise RuntimeError(f"real runner exited {exit_code}: {stderr[-1000:]}")

    raw = runner_output / "raw"
    metrics = _read_csv(raw / "stage4e_metrics.csv")
    if len(metrics) != 1 or metrics[0].get("status") != "COMPLETED":
        raise RuntimeError(f"real runner did not produce exactly one completed metrics row: {metrics}")
    source_run_id = metrics[0]["run_id"]
    metric = metrics[0]
    metric.update({
        "runner_run_id": source_run_id,
        "run_id": run.run_id,
        "status": "COMPLETED",
        "scientific_validity": metric.get("scientific_validity_exact", metric.get("scientific_validity", "FLAGGED")),
        "model": run.model,
        "boundary": run.boundary,
        "delta": run.delta,
        "c6": run.c6,
        "seed": str(run.seed),
    })
    _write_csv(bundle / "metrics.csv", [metric])

    mappings = (
        ("stage4e_history.csv", "history.csv"),
        ("stage4e_site_observables.csv", "site_observables.csv"),
        ("stage4e_corr_observables.csv", "corr_observables.csv"),
    )
    for source_name, target_name in mappings:
        rows = _read_csv(raw / source_name)
        for item in rows:
            if item.get("run_id") != source_run_id:
                raise RuntimeError(f"{source_name} contains a foreign run_id")
            item["runner_run_id"] = source_run_id
            item["run_id"] = run.run_id
        _write_csv(bundle / target_name, rows)

    checkpoints = sorted((runner_output / "checkpoints").glob("*.npz"))
    if len(checkpoints) != 1:
        raise RuntimeError(f"expected one production checkpoint, found {len(checkpoints)}")
    shutil.copy2(checkpoints[0], bundle / "checkpoint.npz")
    if fault_phase == "during-checkpoint":
        import signal
        os.kill(os.getpid(), signal.SIGKILL)
    write_json(bundle / "checkpoint_manifest.json", _checkpoint_manifest(bundle / "checkpoint.npz", run))

    manifest = {
        "schema_version": SCHEMA_VERSION,
        **run.to_dict(),
        "expected_spec_hash": run.spec_hash,
        "attempt_id": row["attempt_id"],
        "claim_token_hash": token_hash(row["claim_token"]),
        "created_utc": utc_now(),
        "runner_run_id": source_run_id,
    }
    write_json(bundle / "run_manifest.json", manifest)
    (bundle / "resolved_config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=True))
    write_json(bundle / "exit_status.json", {
        "run_id": run.run_id,
        "attempt_id": row["attempt_id"],
        "exit_code": 0,
        "status": "COMPLETED",
        "wall_seconds": time.monotonic() - started,
        "ended_utc": utc_now(),
    })
    shutil.rmtree(runner_output)
    runner_cfg.unlink()
    write_json(bundle / "artifact_manifest.json", {"schema_version": SCHEMA_VERSION, "files": hash_artifacts_before_complete(bundle)})
    time.sleep(0.002)
    write_json(bundle / "COMPLETE.json", {
        "schema_version": SCHEMA_VERSION,
        "run_id": run.run_id,
        "attempt_id": row["attempt_id"],
        "completed_utc": utc_now(),
        "artifact_validation": "PASS",
        "artifact_manifest_sha256": sha256_file(bundle / "artifact_manifest.json"),
    })
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 4F.1c fenced transactional worker")
    parser.add_argument("--config", default="configs/stage4f1c_campaign.yaml")
    parser.add_argument("--lane", choices=["core", "control"], default="core")
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--max-runs", type=int, default=1)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--real-run", action="store_true")
    parser.add_argument(
        "--fault-phase",
        choices=["during-training", "during-checkpoint", "during-promotion", "after-promotion"],
        help="Engineering crash-test hook: sends SIGKILL at the selected phase.",
    )
    parser.add_argument("--fault-delay", type=float, default=2.0)
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    root = Path(cfg["campaign"]["output_root"])
    reg = Stage4F1BRegistry(root / "registry.sqlite")
    scheduler = cfg.get("scheduler", {})
    lease_seconds = int(scheduler.get("lease_seconds", 300))
    backoff = int(scheduler.get("backoff_seconds", 60))
    completed = 0
    failures = 0

    for _ in range(args.max_runs):
        row = reg.claim_next(args.lane, args.worker_id, lease_seconds=lease_seconds)
        if row is None:
            print("NO_PENDING_RUN")
            break
        run = row_to_runspec(row)
        tmp: Path | None = None
        try:
            if args.dry_run:
                attempts = root / "attempts" / safe_id(run.run_id)
                tmp = write_valid_synthetic_bundle(
                    attempts,
                    run,
                    attempt_id=row["attempt_id"],
                    claim_token=row["claim_token"],
                )
            else:
                tmp = write_real_bundle(
                    root,
                    cfg,
                    run,
                    row,
                    reg,
                    args.worker_id,
                    lease_seconds,
                    fault_phase=args.fault_phase,
                    fault_delay=args.fault_delay,
                )

            ok, errors = validate_run_bundle_semantic(
                tmp,
                expected_spec=row["expected_spec"],
                expected_attempt_id=row["attempt_id"],
                expected_claim_token=row["claim_token"],
            )
            if not ok:
                raise RuntimeError("bundle validation failed: " + "; ".join(errors))
            if not reg.owns_claim(run.run_id, args.worker_id, row["attempt_id"], row["claim_token"]):
                raise RuntimeError("claim lost before publication; late bundle quarantined")
            final = root / "runs" / run.run_id
            quarantine_root: Path | None = None
            if final.exists():
                existing_ok, _ = validate_run_bundle_semantic(
                    final,
                    expected_spec=row["expected_spec"],
                    expected_attempt_id=row["attempt_id"],
                )
                if existing_ok:
                    raise FileExistsError(f"valid canonical bundle already exists: {final}")
                quarantine_root = root / "quarantine" / "invalid_canonical"
            atomic_promote_no_overwrite(
                tmp,
                final,
                quarantine_root=quarantine_root,
                fault_after_lock=args.fault_phase == "during-promotion",
            )
            tmp = None
            if args.fault_phase == "after-promotion":
                import signal
                os.kill(os.getpid(), signal.SIGKILL)
            reg.mark_complete(
                run.run_id,
                args.worker_id,
                row["attempt_id"],
                row["claim_token"],
                final,
                sha256_file(final / "COMPLETE.json"),
            )
            completed += 1
            print(f"COMPLETE {run.run_id}")
        except Exception as exc:
            failures += 1
            if tmp and tmp.exists():
                quarantine = root / "quarantine" / safe_id(run.run_id)
                quarantine.mkdir(parents=True, exist_ok=True)
                destination = quarantine / f"{row['attempt_id']}__{safe_id(type(exc).__name__)}"
                if not destination.exists():
                    os.rename(tmp, destination)
            fenced = reg.mark_failed(
                run.run_id,
                args.worker_id,
                row["attempt_id"],
                row["claim_token"],
                str(exc),
                retryable=True,
                backoff_seconds=backoff,
            )
            suffix = "" if fenced else " (stale attempt; registry unchanged)"
            print(f"FAILED {run.run_id}: {exc}{suffix}", file=sys.stderr)

    print(json.dumps({"worker_id": args.worker_id, "completed": completed, "failures": failures}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
