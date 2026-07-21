from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

from stage4f1b_tx import load_config, write_json


PHASES = ("during-training", "during-checkpoint", "during-promotion", "after-promotion")


def run(command: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic Stage 4F.1c SIGKILL/recovery probes")
    parser.add_argument("--config", default="configs/stage4f1c_crash_probe.yaml")
    parser.add_argument("--output", default="results_stage4f1c/crash_probe")
    parser.add_argument("--phases", nargs="*", choices=PHASES, default=list(PHASES))
    args = parser.parse_args()
    template = load_config(Path(args.config))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = "scripts:src"
    env.setdefault("MPLCONFIGDIR", "/tmp/graphkan-mpl")
    env.setdefault("XDG_CACHE_HOME", "/tmp/graphkan-cache")
    results: dict[str, dict] = {}
    failures = []

    for phase in args.phases:
        cfg = json.loads(json.dumps(template))
        root = output / phase
        cfg["campaign"]["name"] = f"stage4f1c_crash_probe_{phase.replace('-', '_')}"
        cfg["campaign"]["output_root"] = str(root)
        cfg_path = output / f"{phase}.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        init = run([sys.executable, "scripts/stage4f1c_campaign.py", "init", "--config", str(cfg_path)], env)
        fault = run([
            sys.executable, "scripts/stage4f1c_worker.py", "--config", str(cfg_path), "--lane", "core",
            "--worker-id", f"fault-{phase}", "--real-run", "--max-runs", "1",
            "--fault-phase", phase, "--fault-delay", "1",
        ], env)
        expected_kill = fault.returncode in {-9, 137}
        time.sleep(float(cfg["scheduler"]["lease_seconds"]) + 0.5)
        recovery = run([
            sys.executable, "scripts/stage4f1c_worker.py", "--config", str(cfg_path), "--lane", "core",
            "--worker-id", f"recovery-{phase}", "--real-run", "--max-runs", "1",
        ], env)
        validation = run([
            sys.executable, "scripts/stage4f1c_campaign.py", "validate", "--campaign", str(root),
            "--strict-integrity", "--reconcile-registry", "--require-complete",
        ], env)
        passed = init.returncode == 0 and expected_kill and recovery.returncode == 0 and validation.returncode == 0
        results[phase] = {
            "pass": passed,
            "fault_returncode": fault.returncode,
            "expected_sigkill_observed": expected_kill,
            "recovery_returncode": recovery.returncode,
            "validation_returncode": validation.returncode,
            "fault_stdout_tail": fault.stdout[-1000:],
            "fault_stderr_tail": fault.stderr[-1000:],
            "recovery_stdout_tail": recovery.stdout[-1000:],
            "recovery_stderr_tail": recovery.stderr[-1000:],
            "validation_stdout_tail": validation.stdout[-2000:],
            "validation_stderr_tail": validation.stderr[-1000:],
        }
        if not passed:
            failures.append(phase)

    summary = {"crash_recovery": "PASS" if not failures else "FAIL", "failed_phases": failures, "phases": results}
    write_json(output / "crash_probe_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
