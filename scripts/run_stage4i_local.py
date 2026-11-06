#!/usr/bin/env python3
"""Local-only, resumable Stage 4I G0-G2 orchestration."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from analyse_stage4i import analyse
from analyse_stage4i_gauge import analyse as analyse_gauge
from stage4i_protocol import (
    DEFAULT_CONFIG,
    freeze,
    read_config,
    roster,
    validate_config,
    verify_freeze,
)

ROOT = Path(__file__).resolve().parents[1]


def write(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def status(cfg: dict, output: Path) -> dict:
    rows = []
    for shard in sorted({r["shard"] for r in roster(cfg)}):
        summary = output / "shards" / shard / "shard_summary.json"
        value = json.loads(summary.read_text()) if summary.is_file() else {}
        rows.append(
            {
                "shard": shard,
                "state": "COMPLETE"
                if value.get("completed") == 5 and value.get("failed") == 0
                else "PENDING",
                "completed_runs": value.get("completed", 0),
                "failed_attempts_retained": len(
                    list(
                        (output / "shards" / shard).glob(
                            "runs/*/attempts/*/failure.json"
                        )
                    )
                ),
            }
        )
    result = {
        "schema_version": "stage4i-local-status-v1",
        "complete_shards": sum(x["state"] == "COMPLETE" for x in rows),
        "total_shards": 3,
        "complete_runs": sum(x["completed_runs"] for x in rows),
        "total_runs": 15,
        "rows": rows,
    }
    write(output / "local_status.json", result)
    return result


def shard_complete(output: Path, shard: str) -> bool:
    summary = output / "shards" / shard / "shard_summary.json"
    if not summary.is_file():
        return False
    value = json.loads(summary.read_text())
    return value.get("completed") == 5 and value.get("failed") == 0


def prepare(cfg: dict, config: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    manifest = freeze(cfg, output / "protocol")
    gauge = analyse_gauge(
        cfg,
        Path(cfg["inputs"]["stage4f_n10_root"]),
        Path(cfg["inputs"]["stage4h_root"]),
        output / "gauge",
    )
    env = {
        "schema_version": "stage4i-local-environment-v1",
        "python": sys.version,
        "platform": sys.platform,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "git_tracked_worktree_clean": not bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=ROOT,
                text=True,
            ).strip()
        ),
    }
    write(output / "environment.json", env)
    result = {
        "schema_version": "stage4i-local-prepare-v1",
        "validation": "PASS",
        "protocol_hash": manifest["protocol_hash"],
        "G1_interpretability_gate": gauge["interpretability_gate"],
        "note": "G1 failure restricts claims but does not block the preregistered G2 pilot",
    }
    write(output / "prepare_summary.json", result)
    return result


def run(cfg: dict, config: Path, stage4g: Path, output: Path, workers: int) -> dict:
    if verify_freeze(cfg, output / "protocol")["validation"] != "PASS":
        raise RuntimeError("frozen protocol/source ledger missing or changed")
    if workers not in (1, 2):
        raise ValueError("Stage 4I local workers must be 1 or 2")
    pending = [
        x["shard"] for x in roster(cfg) if not shard_complete(output, x["shard"])
    ]
    pending = sorted(set(pending))
    active = {}
    logs = output / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": os.pathsep.join(
                [str(ROOT / "src"), str(ROOT / "scripts"), env.get("PYTHONPATH", "")]
            ),
            "JAX_ENABLE_X64": "true",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "PYTHONUNBUFFERED": "1",
        }
    )
    try:
        while pending or active:
            while pending and len(active) < workers:
                shard = pending.pop(0)
                handle = (logs / f"{shard}.log").open("a")
                cmd = [
                    sys.executable,
                    "-u",
                    str(ROOT / "scripts/run_stage4i_counterfactual.py"),
                    "--config",
                    str(config),
                    "--shard",
                    shard,
                    "--stage4g-root",
                    str(stage4g),
                    "--output-root",
                    str(output / "shards"),
                ]
                process = subprocess.Popen(
                    cmd,
                    cwd=ROOT,
                    env=env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                active[shard] = (process, handle)
                print(f"START {shard} pid={process.pid}", flush=True)
            for shard, (process, handle) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                handle.close()
                del active[shard]
                print(("COMPLETE" if code == 0 else "FAILED"), shard, flush=True)
            status(cfg, output)
            if active:
                time.sleep(5)
    except KeyboardInterrupt:
        for process, handle in active.values():
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            handle.close()
        status(cfg, output)
        raise SystemExit(130)
    result = status(cfg, output)
    if result["complete_shards"] != 3:
        raise SystemExit(1)
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("prepare", "run", "status", "finalize"))
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--stage4g-root", type=Path)
    p.add_argument(
        "--output", type=Path, default=Path("results_stage4i/local_campaign")
    )
    p.add_argument("--max-workers", type=int, default=2)
    a = p.parse_args()
    cfg = read_config(a.config)
    if validate_config(cfg)["validation"] != "PASS":
        raise SystemExit("invalid Stage 4I config")
    stage4g = a.stage4g_root or Path(cfg["inputs"]["stage4g_n12_root"])
    result = (
        prepare(cfg, a.config, a.output)
        if a.command == "prepare"
        else run(cfg, a.config, stage4g, a.output, a.max_workers)
        if a.command == "run"
        else status(cfg, a.output)
        if a.command == "status"
        else analyse(
            cfg, a.output / "shards", a.output / "merged", a.output / "protocol"
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
