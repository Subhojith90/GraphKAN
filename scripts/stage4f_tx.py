from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def canonical_delta(x: Any) -> str:
    return f"{float(x):.2f}"


def canonical_c6(x: Any) -> str:
    return str(int(float(x)))


def safe_id(text: str) -> str:
    return (
        text.replace("=", "")
        .replace(".", "p")
        .replace("/", "_")
        .replace(":", "_")
        .replace(" ", "_")
    )


@dataclass(frozen=True)
class RunSpec:
    campaign_id: str
    boundary: str
    delta: str
    c6: str
    model: str
    seed: int
    lane: str

    @property
    def run_id(self) -> str:
        return (
            f"{self.campaign_id}__B{self.boundary}"
            f"__D{safe_id(self.delta)}__C{safe_id(self.c6)}"
            f"__M{self.model}__S{self.seed}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "run_id": self.run_id,
            "boundary": self.boundary,
            "delta": self.delta,
            "c6": self.c6,
            "model": self.model,
            "seed": self.seed,
            "lane": self.lane,
        }


class Stage4FRegistry:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path, timeout=30)
        con.row_factory = sqlite3.Row
        return con

    def init(self) -> None:
        with self.connect() as con:
            con.execute("PRAGMA journal_mode=WAL")
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    campaign_id TEXT NOT NULL,
                    boundary TEXT NOT NULL,
                    delta TEXT NOT NULL,
                    c6 TEXT NOT NULL,
                    model TEXT NOT NULL,
                    seed INTEGER NOT NULL,
                    lane TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempt INTEGER NOT NULL DEFAULT 0,
                    worker_id TEXT,
                    heartbeat_utc TEXT,
                    claimed_utc TEXT,
                    completed_utc TEXT,
                    exit_state TEXT,
                    bundle_path TEXT,
                    complete_hash TEXT,
                    resolved_config_hash TEXT,
                    error TEXT,
                    UNIQUE(campaign_id, boundary, delta, c6, model, seed)
                )
                """
            )
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,
                    event_utc TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    detail TEXT
                )
                """
            )

    def register_expected(self, runs: list[RunSpec]) -> None:
        with self.connect() as con:
            for r in runs:
                con.execute(
                    """
                    INSERT OR IGNORE INTO runs
                    (run_id, campaign_id, boundary, delta, c6, model, seed, lane, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'PENDING')
                    """,
                    (
                        r.run_id,
                        r.campaign_id,
                        r.boundary,
                        r.delta,
                        r.c6,
                        r.model,
                        r.seed,
                        r.lane,
                    ),
                )

    def claim_next(self, lane: str, worker_id: str) -> dict[str, Any] | None:
        with self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute(
                """
                SELECT * FROM runs
                WHERE status IN ('PENDING', 'FAILED_RETRYABLE')
                  AND lane = ?
                ORDER BY boundary, delta, c6, model, seed
                LIMIT 1
                """,
                (lane,),
            ).fetchone()
            if row is None:
                con.commit()
                return None

            now = utc_now()
            attempt = int(row["attempt"]) + 1
            con.execute(
                """
                UPDATE runs
                SET status='RUNNING',
                    attempt=?,
                    worker_id=?,
                    heartbeat_utc=?,
                    claimed_utc=?,
                    error=NULL
                WHERE run_id=?
                """,
                (attempt, worker_id, now, now, row["run_id"]),
            )
            con.execute(
                """
                INSERT INTO events(run_id, event_utc, event_type, detail)
                VALUES (?, ?, 'CLAIMED', ?)
                """,
                (row["run_id"], now, f"worker={worker_id};attempt={attempt}"),
            )
            con.commit()

        return dict(row)

    def mark_heartbeat(self, run_id: str, worker_id: str) -> None:
        with self.connect() as con:
            con.execute(
                """
                UPDATE runs
                SET heartbeat_utc=?
                WHERE run_id=? AND worker_id=? AND status='RUNNING'
                """,
                (utc_now(), run_id, worker_id),
            )

    def mark_complete(
        self,
        run_id: str,
        worker_id: str,
        bundle_path: Path,
        complete_hash: str,
        resolved_config_hash: str,
    ) -> None:
        with self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if row is None:
                raise RuntimeError(f"unknown run_id {run_id}")
            if row["worker_id"] != worker_id:
                raise RuntimeError(f"worker mismatch for {run_id}")
            now = utc_now()
            con.execute(
                """
                UPDATE runs
                SET status='COMPLETE',
                    completed_utc=?,
                    exit_state='SUCCESS',
                    bundle_path=?,
                    complete_hash=?,
                    resolved_config_hash=?,
                    heartbeat_utc=?
                WHERE run_id=? AND worker_id=? AND status='RUNNING'
                """,
                (
                    now,
                    str(bundle_path),
                    complete_hash,
                    resolved_config_hash,
                    now,
                    run_id,
                    worker_id,
                ),
            )
            con.execute(
                """
                INSERT INTO events(run_id, event_utc, event_type, detail)
                VALUES (?, ?, 'COMPLETE', ?)
                """,
                (run_id, now, str(bundle_path)),
            )
            con.commit()

    def mark_failed(self, run_id: str, worker_id: str, error: str, retryable: bool = True) -> None:
        status = "FAILED_RETRYABLE" if retryable else "FAILED_FINAL"
        with self.connect() as con:
            now = utc_now()
            con.execute(
                """
                UPDATE runs
                SET status=?,
                    exit_state='FAILED',
                    error=?,
                    heartbeat_utc=?
                WHERE run_id=? AND worker_id=?
                """,
                (status, error[:2000], now, run_id, worker_id),
            )
            con.execute(
                """
                INSERT INTO events(run_id, event_utc, event_type, detail)
                VALUES (?, ?, 'FAILED', ?)
                """,
                (run_id, now, error[:2000]),
            )

    def summary(self) -> dict[str, Any]:
        with self.connect() as con:
            rows = con.execute(
                "SELECT status, COUNT(*) AS n FROM runs GROUP BY status ORDER BY status"
            ).fetchall()
            total = con.execute("SELECT COUNT(*) AS n FROM runs").fetchone()["n"]
            duplicates = con.execute(
                """
                SELECT COUNT(*) AS n FROM (
                  SELECT campaign_id, boundary, delta, c6, model, seed, COUNT(*) AS c
                  FROM runs
                  GROUP BY campaign_id, boundary, delta, c6, model, seed
                  HAVING c > 1
                )
                """
            ).fetchone()["n"]
        return {
            "total_expected": total,
            "status_counts": {r["status"]: r["n"] for r in rows},
            "duplicate_registry_keys": duplicates,
        }


def load_campaign_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def make_campaign_id(config: dict[str, Any]) -> str:
    base = config["campaign"]["name"]
    schema = config["campaign"].get("schema_version", "stage4f-native-v1")
    return f"{base}__{schema}"


def expected_runs(config: dict[str, Any]) -> list[RunSpec]:
    campaign_id = make_campaign_id(config)
    physics = config["physics"]
    models = config["models"]
    runs = []
    for boundary in physics["boundaries"]:
        for delta in physics["deltas"]:
            for c6 in physics["c6_values"]:
                for model, model_cfg in models.items():
                    lane = model_cfg.get("lane", "core")
                    for seed in physics["seeds"]:
                        runs.append(
                            RunSpec(
                                campaign_id=campaign_id,
                                boundary=boundary,
                                delta=canonical_delta(delta),
                                c6=canonical_c6(c6),
                                model=model,
                                seed=int(seed),
                                lane=lane,
                            )
                        )
    return runs


def campaign_paths(config: dict[str, Any]) -> dict[str, Path]:
    root = Path(config["campaign"]["output_root"])
    return {
        "root": root,
        "registry": root / "registry.sqlite",
        "runs": root / "runs",
        "manifests": root / "manifests",
        "resolved_configs": root / "resolved_configs",
        "logs": root / "logs",
    }


def init_campaign(config_path: Path) -> dict[str, Any]:
    cfg = load_campaign_config(config_path)
    paths = campaign_paths(cfg)
    for p in paths.values():
        if p.suffix:
            p.parent.mkdir(parents=True, exist_ok=True)
        else:
            p.mkdir(parents=True, exist_ok=True)

    reg = Stage4FRegistry(paths["registry"])
    reg.init()
    runs = expected_runs(cfg)
    reg.register_expected(runs)

    manifest = {
        "campaign_id": make_campaign_id(cfg),
        "schema_version": cfg["campaign"].get("schema_version", "stage4f-native-v1"),
        "created_utc": utc_now(),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "expected_runs": len(runs),
        "boundaries": cfg["physics"]["boundaries"],
        "deltas": cfg["physics"]["deltas"],
        "c6_values": cfg["physics"]["c6_values"],
        "models": list(cfg["models"].keys()),
        "seeds": cfg["physics"]["seeds"],
        "claim_boundary": "scheduler/provenance repair only; not a completed Stage 4F physics result",
    }
    write_json(paths["manifests"] / "campaign_manifest.json", manifest)
    return {"manifest": manifest, "registry_summary": reg.summary()}


def validate_run_bundle(bundle: Path, requirements: dict[str, Any]) -> tuple[bool, list[str]]:
    errors = []
    required = [
        "run_manifest.json",
        "resolved_config.yaml",
        "exit_status.json",
        "metrics.csv",
        "checkpoint_manifest.json",
        "site_observables.csv",
        "corr_observables.csv",
        "history.csv",
        "artifact_manifest.json",
    ]
    for name in required:
        if not (bundle / name).exists():
            errors.append(f"missing {name}")

    if not errors:
        try:
            exit_status = read_json(bundle / "exit_status.json")
            if requirements.get("require_exitcode_zero", True) and exit_status.get("exit_code") != 0:
                errors.append("exit code is not zero")
        except Exception as exc:
            errors.append(f"bad exit_status.json: {exc}")

    if (bundle / "history.csv").exists() and requirements.get("require_history", True):
        with (bundle / "history.csv").open() as f:
            n = max(0, sum(1 for _ in f) - 1)
        expected = int(requirements.get("expected_history_points", 1))
        if n < expected:
            errors.append(f"history rows {n} < expected {expected}")

    complete = bundle / "COMPLETE.json"
    if requirements.get("require_complete_json_last", True):
        if not complete.exists():
            errors.append("missing COMPLETE.json")
        else:
            try:
                data = read_json(complete)
                if data.get("artifact_validation") != "PASS":
                    errors.append("COMPLETE.json validation not PASS")
            except Exception as exc:
                errors.append(f"bad COMPLETE.json: {exc}")

    return (len(errors) == 0), errors


def hash_artifacts(bundle: Path) -> dict[str, str]:
    out = {}
    for p in sorted(bundle.rglob("*")):
        if p.is_file() and p.name != "artifact_manifest.json":
            out[str(p.relative_to(bundle))] = sha256_file(p)
    return out


def write_mock_run_bundle(
    root: Path,
    run: RunSpec,
    requirements: dict[str, Any],
    fault: str | None = None,
) -> Path:
    tmp = Path(tempfile.mkdtemp(prefix=f".tmp_{run.run_id}_", dir=root))
    write_json(tmp / "run_manifest.json", run.to_dict())
    (tmp / "resolved_config.yaml").write_text("mock: true\n")
    write_json(tmp / "exit_status.json", {"exit_code": 0, "ended_utc": utc_now()})
    (tmp / "metrics.csv").write_text("run_id,status\n" + f"{run.run_id},VALID\n")
    write_json(tmp / "checkpoint_manifest.json", {"checkpoint": "mock.npz", "readable": True})
    (tmp / "site_observables.csv").write_text("run_id,site,value\n" + f"{run.run_id},0,0.0\n")
    (tmp / "corr_observables.csv").write_text("run_id,i,j,value\n" + f"{run.run_id},0,1,0.0\n")

    if fault != "missing_history":
        n_hist = int(requirements.get("expected_history_points", 14))
        with (tmp / "history.csv").open("w") as f:
            f.write("run_id,step,energy\n")
            for i in range(n_hist):
                f.write(f"{run.run_id},{i},0.0\n")

    if fault == "kill_after_metrics":
        return tmp
    if fault == "kill_during_checkpoint":
        (tmp / "checkpoint_manifest.json").write_text("{bad json")
        return tmp
    if fault == "corrupt_checkpoint":
        write_json(tmp / "checkpoint_manifest.json", {"checkpoint": "mock.npz", "readable": False})
        return tmp

    if fault == "kill_before_complete_marker":
        artifact_hashes = hash_artifacts(tmp)
        write_json(tmp / "artifact_manifest.json", {"files": artifact_hashes})
        return tmp

    artifact_hashes = hash_artifacts(tmp)
    write_json(tmp / "artifact_manifest.json", {"files": artifact_hashes})

    if fault is None:
        write_json(
            tmp / "COMPLETE.json",
            {
                "run_id": run.run_id,
                "completed_utc": utc_now(),
                "artifact_validation": "PASS",
                "artifact_manifest_sha256": sha256_file(tmp / "artifact_manifest.json"),
            },
        )

        ok, errors = validate_run_bundle(tmp, requirements)
        if not ok:
            raise RuntimeError(f"mock bundle unexpectedly invalid: {errors}")

    return tmp


def atomic_promote(tmp: Path, final: Path) -> None:
    final.parent.mkdir(parents=True, exist_ok=True)
    if final.exists():
        shutil.rmtree(final)
    os.replace(tmp, final)


def run_fault_injection(config_path: Path, fault_path: Path) -> dict[str, Any]:
    cfg = load_campaign_config(config_path)
    fault_cfg = yaml.safe_load(fault_path.read_text())
    out_root = Path(fault_cfg["fault_injection"]["output_root"])
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    requirements = cfg["completion_requirements"]
    campaign_id = "stage4f1_fault_injection__stage4f-native-v1"
    run = RunSpec(
        campaign_id=campaign_id,
        boundary="open",
        delta="1.50",
        c6="4",
        model="endpoint_product_b32",
        seed=0,
        lane="core",
    )

    reg = Stage4FRegistry(out_root / "registry.sqlite")
    reg.init()
    reg.register_expected([run])

    results = {}
    invalid_faults = [
        "kill_after_metrics",
        "kill_during_checkpoint",
        "kill_before_complete_marker",
        "corrupt_checkpoint",
        "missing_history",
    ]

    for fault in invalid_faults:
        tmp = write_mock_run_bundle(out_root, run, requirements, fault=fault)
        ok, errors = validate_run_bundle(tmp, requirements)
        results[fault] = {"valid": ok, "errors": errors}
        if ok:
            raise RuntimeError(f"fault case incorrectly validated: {fault}")
        shutil.rmtree(tmp, ignore_errors=True)

    tmp = write_mock_run_bundle(out_root, run, requirements, fault=None)
    ok, errors = validate_run_bundle(tmp, requirements)
    if not ok:
        raise RuntimeError(f"valid complete bundle failed validation: {errors}")
    final = out_root / "runs" / run.run_id
    atomic_promote(tmp, final)

    worker1 = reg.claim_next("core", "worker-A")
    worker2 = reg.claim_next("core", "worker-B")
    duplicate_claim_blocked = worker1 is not None and worker2 is None
    if not duplicate_claim_blocked:
        raise RuntimeError("duplicate-worker claim was not blocked")

    reg.mark_complete(
        run.run_id,
        "worker-A",
        final,
        sha256_file(final / "COMPLETE.json"),
        sha256_file(final / "resolved_config.yaml"),
    )

    summary = {
        "fault_injection": "PASS",
        "invalid_faults_rejected": results,
        "valid_complete_bundle": str(final),
        "duplicate_worker_claim_blocked": duplicate_claim_blocked,
        "registry_summary": reg.summary(),
    }
    write_json(out_root / "fault_injection_summary.json", summary)
    return summary


def write_release_root_manifest(root: Path) -> Path:
    manifest = root / "MANIFEST.sha256"
    rows = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name != "MANIFEST.sha256":
            rows.append(f"{sha256_file(p)}  {p.relative_to(root)}")
    manifest.write_text("\n".join(rows) + "\n")
    return manifest
