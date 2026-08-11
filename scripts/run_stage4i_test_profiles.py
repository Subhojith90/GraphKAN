#!/usr/bin/env python3
"""Run bounded Stage 4I base/full-NQS test profiles with durable transcripts."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROFILES = {
    "base": {"marker": "not nqs and not slow", "timeout": 1200},
    "full-nqs": {"marker": None, "timeout": 3600},
    "nqs-only": {"marker": "nqs", "timeout": 1800},
}


def installed(names: list[str]) -> dict[str, str]:
    result = {}
    for name in names:
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = "MISSING"
    return result


def run_profile(name: str, output: Path) -> dict:
    spec = PROFILES[name]
    command = [sys.executable, "-m", "pytest", "-q"]
    if spec["marker"]:
        command += ["-m", spec["marker"]]
    env = os.environ.copy()
    env.update(
        {
            "JAX_ENABLE_X64": "1",
            "PYTHONPATH": str(ROOT / "src"),
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        }
    )
    started = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=spec["timeout"],
        )
        state = "PASS" if completed.returncode == 0 else "FAIL"
        transcript = completed.stdout
        code = completed.returncode
    except subprocess.TimeoutExpired as exc:
        state = "TIMEOUT"
        code = 124
        transcript = (
            exc.stdout or ""
        ) + f"\nSTAGE4I_EXPLICIT_TIMEOUT after {spec['timeout']} seconds\n"
    (output / f"pytest_{name}.txt").write_text(transcript)
    passed = sum(map(int, re.findall(r"(\d+) passed", transcript)))
    skipped = sum(map(int, re.findall(r"(\d+) skipped", transcript)))
    deselected = sum(map(int, re.findall(r"(\d+) deselected", transcript)))
    return {
        "profile": name,
        "command": command,
        "started_utc": started,
        "timeout_seconds": spec["timeout"],
        "exit_code": code,
        "status": state,
        "passed": passed,
        "skipped": skipped,
        "deselected": deselected,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=Path("results_stage4i/test_profiles"))
    p.add_argument("--profile", choices=("base", "full-nqs", "all"), default="all")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    names = (
        ["base"]
        if a.profile == "base"
        else ["full-nqs", "nqs-only"]
        if a.profile == "full-nqs"
        else ["base", "full-nqs", "nqs-only"]
    )
    rows = [run_profile(x, a.output) for x in names]
    deps = installed(
        [
            "numpy",
            "scipy",
            "pandas",
            "PyYAML",
            "matplotlib",
            "pytest",
            "tqdm",
            "jax",
            "jaxlib",
            "flax",
            "optax",
            "netket",
        ]
    )
    (a.output / "installed_dependencies.json").write_text(
        json.dumps(deps, indent=2, sort_keys=True) + "\n"
    )
    blocking = [
        f"{x['profile']} profile {x['status']}" for x in rows if x["status"] != "PASS"
    ]
    nqs = next((x for x in rows if x["profile"] == "nqs-only"), None)
    if nqs and nqs["skipped"]:
        blocking.append(f"full NQS profile contains {nqs['skipped']} NQS skips")
    result = {
        "schema_version": "stage4i-test-profile-parity-v1",
        "validation": "PASS" if not blocking else "FAIL",
        "blocking_errors": blocking,
        "profiles": rows,
        "dependencies": deps,
    }
    (a.output / "stage4i_test_profile_gate.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if blocking:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
