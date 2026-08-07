#!/usr/bin/env python3
"""Resumable, local-only orchestration for the complete Stage 4H campaign."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from stage4h_protocol import (
    DEFAULT_CONFIG,
    freeze,
    read_config,
    roster,
    validate_config,
    verify_freeze,
)
from stage4h_transplant_preflight import run as run_transplant_preflight
from validate_stage4g_periodic_alias import validate as validate_periodic_alias


ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def stage4g_input_summary(root: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    from graphkan_amo_nqs.training.stage4h_transplant import (
        find_radial_checkpoint,
        sha256_file,
    )

    entries = []
    for cell in cfg["physics"]["cells"]:
        for seed in cfg["physics"]["seeds"]:
            checkpoint = find_radial_checkpoint(
                root, delta=cell["delta"], c6=cell["c6"], seed=seed
            )
            entries.append(
                {
                    "cell_tag": cell["tag"],
                    "seed": seed,
                    "path": str(checkpoint.resolve()),
                    "sha256": sha256_file(checkpoint),
                }
            )
    return {
        "schema_version": "stage4h-local-stage4g-input-v1",
        "validation": "PASS" if len(entries) == 15 else "FAIL",
        "entries": entries,
    }


def prepare(cfg: dict[str, Any], stage4g_root: Path, output: Path) -> dict[str, Any]:
    validation = validate_config(cfg)
    if validation["validation"] != "PASS":
        raise RuntimeError(json.dumps(validation, indent=2))
    output.mkdir(parents=True, exist_ok=True)
    manifest = freeze(cfg, output / "protocol")
    input_summary = stage4g_input_summary(stage4g_root, cfg)
    write_json(output / "environment/stage4g_input_ledger.json", input_summary)
    transplant = run_transplant_preflight(cfg, stage4g_root, output / "environment")
    periodic = validate_periodic_alias(12)
    write_json(output / "environment/n12_periodic_alias_gate.json", periodic)
    environment = {
        "schema_version": "stage4h-local-environment-v1",
        "python": sys.version,
        "platform": sys.platform,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "git_worktree_clean": not bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=ROOT,
                text=True,
            ).strip()
        ),
        "stage4g_root": str(stage4g_root.resolve()),
    }
    write_json(output / "environment/local_environment.json", environment)
    result = {
        "schema_version": "stage4h-local-prepare-v1",
        "validation": "PASS"
        if transplant["validation"] == periodic["validation"] == "PASS"
        else "FAIL",
        "protocol_hash": manifest["protocol_hash"],
        "radial_checkpoints": len(input_summary["entries"]),
        "transplant_attestations": transplant["passes"],
        "periodic_alias_passes": periodic["passes"],
    }
    write_json(output / "prepare_summary.json", result)
    return result


def shard_names(cfg: dict[str, Any]) -> list[str]:
    return sorted({row["shard"] for row in roster(cfg)})


def shard_complete(output: Path, shard: str) -> bool:
    summary = output / "shards" / shard / "shard_summary.json"
    if not summary.is_file():
        return False
    value = json.loads(summary.read_text())
    return value.get("completed") == 5 and value.get("failed") == 0


def status(cfg: dict[str, Any], output: Path) -> dict[str, Any]:
    rows = []
    for shard in shard_names(cfg):
        root = output / "shards" / shard
        completed = len(list(root.glob("runs/*/completion.json")))
        failed_attempts = len(list(root.glob("runs/*/attempts/*/failure.json")))
        rows.append(
            {
                "shard": shard,
                "state": "COMPLETE" if shard_complete(output, shard) else "PENDING",
                "completed_runs": completed,
                "failed_attempts_retained": failed_attempts,
                "log": str(output / "logs" / f"{shard}.log"),
            }
        )
    result = {
        "schema_version": "stage4h-local-status-v1",
        "complete_shards": sum(row["state"] == "COMPLETE" for row in rows),
        "total_shards": len(rows),
        "complete_runs": sum(row["completed_runs"] for row in rows),
        "total_runs": 60,
        "rows": rows,
    }
    write_json(output / "local_status.json", result)
    return result


def runner_command(
    cfg_path: Path, stage4g_root: Path, output: Path, shard: str
) -> list[str]:
    return [
        sys.executable,
        "-u",
        str(ROOT / "scripts/run_stage4h_warmstart.py"),
        "--config",
        str(cfg_path),
        "--shard",
        shard,
        "--stage4g-root",
        str(stage4g_root),
        "--output-root",
        str(output / "shards"),
        "--resume",
    ]


def run_campaign(
    cfg: dict[str, Any],
    cfg_path: Path,
    stage4g_root: Path,
    output: Path,
    max_workers: int,
) -> dict[str, Any]:
    if verify_freeze(cfg, output / "protocol")["validation"] != "PASS":
        raise RuntimeError(
            "local frozen protocol is missing or its source ledger changed"
        )
    if max_workers < 1 or max_workers > 4:
        raise ValueError("max-workers must be between 1 and 4")
    pending = [name for name in shard_names(cfg) if not shard_complete(output, name)]
    logs = output / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    active: dict[str, tuple[subprocess.Popen, Any]] = {}
    failures: list[str] = []
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "src"), str(ROOT / "scripts"), environment.get("PYTHONPATH", "")]
    )
    environment.update(
        {
            "JAX_ENABLE_X64": "true",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "PYTHONUNBUFFERED": "1",
            "MPLCONFIGDIR": str(output / "environment/matplotlib"),
        }
    )
    print(
        f"Stage 4H local campaign: {len(pending)} pending shards, "
        f"{max_workers} concurrent workers",
        flush=True,
    )
    try:
        while pending or active:
            while pending and len(active) < max_workers:
                shard = pending.pop(0)
                log_path = logs / f"{shard}.log"
                handle = log_path.open("a")
                handle.write(
                    f"\n=== launch {time.strftime('%Y-%m-%dT%H:%M:%S%z')} ===\n"
                )
                handle.flush()
                process = subprocess.Popen(
                    runner_command(cfg_path, stage4g_root, output, shard),
                    cwd=ROOT,
                    env=environment,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                active[shard] = (process, handle)
                print(f"START {shard} pid={process.pid} log={log_path}", flush=True)
            for shard, (process, handle) in list(active.items()):
                returncode = process.poll()
                if returncode is None:
                    continue
                handle.write(
                    f"=== exit {returncode} {time.strftime('%Y-%m-%dT%H:%M:%S%z')} ===\n"
                )
                handle.close()
                del active[shard]
                if returncode != 0 or not shard_complete(output, shard):
                    failures.append(shard)
                    print(
                        f"FAILED {shard}; evidence retained; inspect {logs / (shard + '.log')}",
                        flush=True,
                    )
                else:
                    print(f"COMPLETE {shard}", flush=True)
            status(cfg, output)
            if active:
                time.sleep(5)
    except KeyboardInterrupt:
        print(
            "Interrupt received; terminating active shards. Completed runs and immutable checkpoints are retained.",
            flush=True,
        )
        for process, _ in active.values():
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for process, handle in active.values():
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
            handle.close()
        status(cfg, output)
        raise SystemExit(130)
    result = status(cfg, output)
    result["failed_shards_this_invocation"] = failures
    write_json(output / "local_run_summary.json", result)
    if failures or result["complete_shards"] != 12:
        raise SystemExit(1)
    return result


def finalize(cfg: dict[str, Any], output: Path) -> dict[str, Any]:
    current = status(cfg, output)
    if current["complete_shards"] != 12 or current["complete_runs"] != 60:
        raise RuntimeError(
            f"Stage 4H is incomplete: {current['complete_shards']}/12 shards, "
            f"{current['complete_runs']}/60 runs"
        )
    merge_command = [
        sys.executable,
        str(ROOT / "scripts/merge_stage4h.py"),
        "--input",
        str(output / "shards"),
        "--output",
        str(output / "merged"),
    ]
    analyse_command = [
        sys.executable,
        str(ROOT / "scripts/analyse_stage4h.py"),
        "--merged",
        str(output / "merged"),
        "--protocol",
        str(output / "protocol"),
        "--periodic-alias",
        str(output / "environment/n12_periodic_alias_gate.json"),
        "--strict",
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "src"), str(ROOT / "scripts"), environment.get("PYTHONPATH", "")]
    )
    subprocess.run(merge_command, cwd=ROOT, env=environment, check=True)
    completed = subprocess.run(analyse_command, cwd=ROOT, env=environment)
    gate_path = output / "merged/analysis/stage4h_final_gate.json"
    gate = json.loads(gate_path.read_text()) if gate_path.is_file() else {}
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    return gate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("prepare", "run", "status", "finalize", "all")
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--stage4g-root",
        type=Path,
        default=ROOT / "results_stage4g/final_local_20260807/confirmation/open",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results_stage4h/local_campaign_20260807",
    )
    parser.add_argument("--max-workers", type=int, default=2)
    args = parser.parse_args()
    cfg = read_config(args.config)
    if args.command in {"prepare", "all"}:
        print(
            json.dumps(
                prepare(cfg, args.stage4g_root, args.output), indent=2, sort_keys=True
            )
        )
    if args.command in {"run", "all"}:
        print(
            json.dumps(
                run_campaign(
                    cfg, args.config, args.stage4g_root, args.output, args.max_workers
                ),
                indent=2,
                sort_keys=True,
            )
        )
    if args.command == "status":
        print(json.dumps(status(cfg, args.output), indent=2, sort_keys=True))
    if args.command in {"finalize", "all"}:
        print(json.dumps(finalize(cfg, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
