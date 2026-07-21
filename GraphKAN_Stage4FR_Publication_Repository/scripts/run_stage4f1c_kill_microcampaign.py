from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

from stage4f1b_tx import load_config, write_json


FAULT_PLAN = (
    ("during-training", "core"),
    ("during-checkpoint", "core"),
    ("during-promotion", "control"),
    ("after-promotion", "core"),
)


def execute(command: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(command), flush=True)
    result = subprocess.run(command, text=True, env=env)
    print(f"exit={result.returncode}", flush=True)
    return result


def registry_evidence(root: Path) -> dict:
    con = sqlite3.connect(root / "registry.sqlite")
    con.row_factory = sqlite3.Row
    attempts = [dict(row) for row in con.execute(
        "SELECT run_id,model,boundary,delta,status,attempt FROM runs ORDER BY priority"
    )]
    events = [dict(row) for row in con.execute(
        "SELECT event_type,COUNT(*) AS count FROM events GROUP BY event_type ORDER BY event_type"
    )]
    return {"runs": attempts, "event_counts": events}


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute the mandatory Stage 4F.1c kill-tested 16-run gate")
    parser.add_argument("--config", default="configs/stage4f1c_real_microcampaign_killtest.yaml")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    root = Path(config["campaign"]["output_root"])
    if (root / "registry.sqlite").exists():
        raise SystemExit(
            f"Refusing to reuse {root}: registry already exists. Preserve it and choose a new campaign name/output root."
        )

    env = dict(os.environ)
    env["PYTHONPATH"] = "scripts:src"
    env.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
    env.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(env["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    python = args.python
    campaign = [python, "scripts/stage4f1c_campaign.py"]
    worker = [python, "scripts/stage4f1c_worker.py"]
    failures: list[str] = []
    fault_results: list[dict] = []

    init = execute(campaign + ["init", "--config", str(config_path)], env)
    if init.returncode != 0:
        raise SystemExit(init.returncode)

    lease_wait = float(config["scheduler"]["lease_seconds"]) + 1.0
    for index, (phase, lane) in enumerate(FAULT_PLAN, start=1):
        fault = execute(worker + [
            "--config", str(config_path), "--lane", lane,
            "--worker-id", f"kill-{index}-{phase}", "--real-run", "--max-runs", "1",
            "--fault-phase", phase, "--fault-delay", "2",
        ], env)
        killed = fault.returncode in {-9, 137}
        print(f"Waiting {lease_wait:.0f}s for fenced lease expiry...", flush=True)
        time.sleep(lease_wait)
        recovery = execute(worker + [
            "--config", str(config_path), "--lane", lane,
            "--worker-id", f"recover-{index}-{phase}", "--real-run", "--max-runs", "1",
        ], env)
        passed = killed and recovery.returncode == 0
        fault_results.append({
            "phase": phase,
            "lane": lane,
            "fault_returncode": fault.returncode,
            "sigkill_observed": killed,
            "recovery_returncode": recovery.returncode,
            "pass": passed,
        })
        if not passed:
            failures.append(phase)
            break

    if not failures:
        for lane in ("core", "control"):
            drain = execute(worker + [
                "--config", str(config_path), "--lane", lane,
                "--worker-id", f"final-{lane}", "--real-run", "--max-runs", "100",
            ], env)
            if drain.returncode != 0:
                failures.append(f"drain-{lane}")

    validation = execute(campaign + [
        "validate", "--campaign", str(root), "--strict-integrity",
        "--reconcile-registry", "--require-complete",
    ], env)
    if validation.returncode != 0:
        failures.append("validation")

    evidence = registry_evidence(root)
    retried_rows = [row for row in evidence["runs"] if int(row["attempt"]) >= 2]
    if len(retried_rows) < 4:
        failures.append("fewer-than-four-retried-work-items")
    summary = {
        "gate": "PASS" if not failures else "FAIL",
        "config": str(config_path),
        "campaign_root": str(root),
        "fault_results": fault_results,
        "retried_work_items": retried_rows,
        "registry_evidence": evidence,
        "validation_returncode": validation.returncode,
        "failures": failures,
    }
    write_json(root / "kill_microcampaign_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
