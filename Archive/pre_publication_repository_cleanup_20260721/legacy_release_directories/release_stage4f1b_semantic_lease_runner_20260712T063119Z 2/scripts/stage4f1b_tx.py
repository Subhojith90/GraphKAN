from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def now_epoch() -> float:
    return time.time()


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


def canonical_json_bytes(data: Any) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode()


def protocol_hash_from_config(config: dict[str, Any], project_root: Path = Path(".")) -> str:
    identity = {
        "schema_version": config["campaign"].get("schema_version", "stage4f-native-v1"),
        "physics": config.get("physics", {}),
        "models": config.get("models", {}),
        "base_configs": {},
        "completion_requirements": config.get("completion_requirements", {}),
    }

    for key, rel in config.get("base_configs", {}).items():
        p = project_root / rel
        identity["base_configs"][key] = {
            "path": str(rel),
            "exists": p.exists(),
            "sha256": sha256_file(p) if p.exists() and p.is_file() else None,
        }

    env = project_root / "reproducibility/environment_stage4f_portable.yml"
    identity["environment"] = {
        "path": str(env),
        "exists": env.exists(),
        "sha256": sha256_file(env) if env.exists() else None,
    }

    h = hashlib.sha256()
    h.update(canonical_json_bytes(identity))
    return h.hexdigest()


def safe_id(text: Any) -> str:
    return str(text).replace(".", "p").replace("/", "_").replace(":", "_").replace(" ", "_")


@dataclass(frozen=True)
class RunSpec:
    campaign_id: str
    protocol_hash: str
    boundary: str
    delta: str
    c6: str
    model: str
    seed: int
    lane: str
    priority: int

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
            "protocol_hash": self.protocol_hash,
            "run_id": self.run_id,
            "boundary": self.boundary,
            "delta": self.delta,
            "c6": self.c6,
            "model": self.model,
            "seed": self.seed,
            "lane": self.lane,
            "priority": self.priority,
        }


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def make_campaign_id(config: dict[str, Any], project_root: Path = Path(".")) -> str:
    name = config["campaign"]["name"]
    schema = config["campaign"].get("schema_version", "stage4f-native-v1")
    phash = protocol_hash_from_config(config, project_root)[:16]
    return f"{name}__{schema}__P{phash}"


def expected_runs(config: dict[str, Any], project_root: Path = Path(".")) -> list[RunSpec]:
    phash = protocol_hash_from_config(config, project_root)
    cid = make_campaign_id(config, project_root)
    physics = config["physics"]
    models = config["models"]

    boundaries = list(physics["boundaries"])
    deltas = list(physics["deltas"])
    c6s = list(physics["c6_values"])
    seeds = list(physics["seeds"])
    model_names = list(models.keys())

    runs: list[RunSpec] = []
    priority = 0

    for seed in seeds:
        for boundary in boundaries:
            for delta in deltas:
                for c6 in c6s:
                    for model in model_names:
                        lane = models[model].get("lane", "core")
                        runs.append(
                            RunSpec(
                                campaign_id=cid,
                                protocol_hash=phash,
                                boundary=str(boundary),
                                delta=f"{float(delta):.2f}",
                                c6=str(int(float(c6))),
                                model=model,
                                seed=int(seed),
                                lane=lane,
                                priority=priority,
                            )
                        )
                        priority += 1

    return runs


class Stage4F1BRegistry:
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
                CREATE TABLE IF NOT EXISTS campaigns (
                    campaign_id TEXT PRIMARY KEY,
                    protocol_hash TEXT NOT NULL,
                    manifest_sha256 TEXT NOT NULL,
                    created_utc TEXT NOT NULL
                )
                """
            )
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    campaign_id TEXT NOT NULL,
                    protocol_hash TEXT NOT NULL,
                    boundary TEXT NOT NULL,
                    delta TEXT NOT NULL,
                    c6 TEXT NOT NULL,
                    model TEXT NOT NULL,
                    seed INTEGER NOT NULL,
                    lane TEXT NOT NULL,
                    priority INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    attempt INTEGER NOT NULL DEFAULT 0,
                    worker_id TEXT,
                    lease_expires_epoch REAL,
                    heartbeat_utc TEXT,
                    claimed_utc TEXT,
                    completed_utc TEXT,
                    next_retry_epoch REAL DEFAULT 0,
                    max_attempts INTEGER DEFAULT 3,
                    bundle_path TEXT,
                    complete_hash TEXT,
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

    def register_campaign(self, config_path: Path, config: dict[str, Any], project_root: Path = Path(".")) -> dict[str, Any]:
        self.init()
        cid = make_campaign_id(config, project_root)
        phash = protocol_hash_from_config(config, project_root)
        runs = expected_runs(config, project_root)

        manifest = {
            "campaign_id": cid,
            "protocol_hash": phash,
            "created_utc": utc_now(),
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "expected_runs": len(runs),
            "claim_boundary": "Stage 4F.1b engineering gate only; not Stage 4F-R science.",
        }

        manifest_hash = hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()

        with self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            existing = con.execute("SELECT * FROM campaigns").fetchall()
            if existing:
                row = existing[0]
                if row["campaign_id"] != cid or row["protocol_hash"] != phash:
                    raise RuntimeError("Protocol mutation detected: refusing to modify existing campaign registry.")
            else:
                con.execute(
                    "INSERT INTO campaigns(campaign_id, protocol_hash, manifest_sha256, created_utc) VALUES (?, ?, ?, ?)",
                    (cid, phash, manifest_hash, utc_now()),
                )

            for r in runs:
                con.execute(
                    """
                    INSERT OR IGNORE INTO runs
                    (run_id, campaign_id, protocol_hash, boundary, delta, c6, model, seed, lane, priority, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING')
                    """,
                    (r.run_id, r.campaign_id, r.protocol_hash, r.boundary, r.delta, r.c6, r.model, r.seed, r.lane, r.priority),
                )
            con.commit()

        return manifest

    def claim_next(self, lane: str, worker_id: str, lease_seconds: int = 300) -> dict[str, Any] | None:
        with self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            t = now_epoch()

            stale = con.execute(
                """
                SELECT * FROM runs
                WHERE status='RUNNING'
                  AND lane=?
                  AND lease_expires_epoch IS NOT NULL
                  AND lease_expires_epoch < ?
                ORDER BY priority
                LIMIT 1
                """,
                (lane, t),
            ).fetchone()

            if stale is not None:
                attempts = int(stale["attempt"])
                if attempts >= int(stale["max_attempts"]):
                    con.execute(
                        "UPDATE runs SET status='QUARANTINED', error=? WHERE run_id=?",
                        ("stale lease exceeded max attempts", stale["run_id"]),
                    )
                    con.execute(
                        "INSERT INTO events(run_id, event_utc, event_type, detail) VALUES (?, ?, 'QUARANTINED', ?)",
                        (stale["run_id"], utc_now(), "stale lease exceeded max attempts"),
                    )
                    con.commit()
                    return self.claim_next(lane, worker_id, lease_seconds)

                con.execute(
                    """
                    UPDATE runs
                    SET status='FAILED_RETRYABLE',
                        worker_id=NULL,
                        lease_expires_epoch=NULL,
                        next_retry_epoch=?,
                        error=?
                    WHERE run_id=?
                    """,
                    (t, "stale lease reclaimed", stale["run_id"]),
                )

            row = con.execute(
                """
                SELECT * FROM runs
                WHERE lane=?
                  AND (
                    status='PENDING'
                    OR (status='FAILED_RETRYABLE' AND next_retry_epoch <= ?)
                  )
                ORDER BY
                  CASE WHEN status='PENDING' THEN 0 ELSE 1 END,
                  priority
                LIMIT 1
                """,
                (lane, t),
            ).fetchone()

            if row is None:
                con.commit()
                return None

            attempt = int(row["attempt"]) + 1
            if attempt > int(row["max_attempts"]):
                con.execute(
                    "UPDATE runs SET status='QUARANTINED', error=? WHERE run_id=?",
                    ("max attempts exceeded", row["run_id"]),
                )
                con.commit()
                return self.claim_next(lane, worker_id, lease_seconds)

            lease = t + lease_seconds
            con.execute(
                """
                UPDATE runs
                SET status='RUNNING',
                    attempt=?,
                    worker_id=?,
                    lease_expires_epoch=?,
                    heartbeat_utc=?,
                    claimed_utc=?,
                    error=NULL
                WHERE run_id=?
                """,
                (attempt, worker_id, lease, utc_now(), utc_now(), row["run_id"]),
            )
            con.execute(
                "INSERT INTO events(run_id, event_utc, event_type, detail) VALUES (?, ?, 'CLAIMED', ?)",
                (row["run_id"], utc_now(), f"worker={worker_id};attempt={attempt}"),
            )
            con.commit()
            return dict(row)

    def mark_heartbeat(self, run_id: str, worker_id: str, lease_seconds: int = 300) -> None:
        with self.connect() as con:
            con.execute(
                """
                UPDATE runs
                SET heartbeat_utc=?, lease_expires_epoch=?
                WHERE run_id=? AND worker_id=? AND status='RUNNING'
                """,
                (utc_now(), now_epoch() + lease_seconds, run_id, worker_id),
            )

    def mark_complete(self, run_id: str, worker_id: str, bundle_path: Path, complete_hash: str) -> None:
        with self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if row is None:
                raise RuntimeError(f"unknown run_id {run_id}")
            if row["status"] != "RUNNING":
                raise RuntimeError(f"run {run_id} is not RUNNING")
            if row["worker_id"] != worker_id:
                raise RuntimeError(f"worker mismatch for {run_id}")

            con.execute(
                """
                UPDATE runs
                SET status='COMPLETE',
                    completed_utc=?,
                    bundle_path=?,
                    complete_hash=?,
                    lease_expires_epoch=NULL
                WHERE run_id=?
                """,
                (utc_now(), str(bundle_path), complete_hash, run_id),
            )
            con.execute(
                "INSERT INTO events(run_id, event_utc, event_type, detail) VALUES (?, ?, 'COMPLETE', ?)",
                (run_id, utc_now(), str(bundle_path)),
            )
            con.commit()

    def mark_failed(self, run_id: str, worker_id: str, error: str, retryable: bool = True, backoff_seconds: int = 60) -> None:
        status = "FAILED_RETRYABLE" if retryable else "FAILED_FINAL"
        with self.connect() as con:
            row = con.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if row is None:
                return
            attempts = int(row["attempt"])
            if attempts >= int(row["max_attempts"]):
                status = "QUARANTINED"
            con.execute(
                """
                UPDATE runs
                SET status=?,
                    error=?,
                    worker_id=NULL,
                    lease_expires_epoch=NULL,
                    next_retry_epoch=?
                WHERE run_id=? AND worker_id=?
                """,
                (status, error[:2000], now_epoch() + backoff_seconds, run_id, worker_id),
            )
            con.execute(
                "INSERT INTO events(run_id, event_utc, event_type, detail) VALUES (?, ?, 'FAILED', ?)",
                (run_id, utc_now(), error[:2000]),
            )

    def summary(self) -> dict[str, Any]:
        with self.connect() as con:
            rows = con.execute("SELECT status, COUNT(*) AS n FROM runs GROUP BY status ORDER BY status").fetchall()
            total = con.execute("SELECT COUNT(*) AS n FROM runs").fetchone()["n"]
            dups = con.execute(
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
            "duplicate_registry_keys": dups,
        }


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def validate_finite_csv(path: Path, required_columns: list[str], run_id: str) -> list[str]:
    errors: list[str] = []
    rows = read_csv_dicts(path)
    if not rows:
        errors.append(f"{path.name}: no rows")
        return errors
    for col in required_columns:
        if col not in rows[0]:
            errors.append(f"{path.name}: missing column {col}")
    for i, row in enumerate(rows):
        if "run_id" in row and row["run_id"] != run_id:
            errors.append(f"{path.name}: row {i} run_id mismatch")
        for k, v in row.items():
            if k in {"run_id", "status", "model", "boundary"}:
                continue
            if v == "":
                continue
            try:
                x = float(v)
                if not (x == x and abs(x) != float("inf")):
                    errors.append(f"{path.name}: non-finite value in {k}")
            except ValueError:
                pass
    return errors


def validate_artifact_manifest(bundle: Path) -> list[str]:
    errors: list[str] = []
    mpath = bundle / "artifact_manifest.json"
    if not mpath.exists():
        return ["missing artifact_manifest.json"]
    try:
        manifest = read_json(mpath)
    except Exception as exc:
        return [f"artifact_manifest.json parse error: {exc}"]

    files = manifest.get("files")
    if not isinstance(files, dict):
        return ["artifact_manifest.json missing files dict"]

    for rel, expected_hash in files.items():
        if rel == "artifact_manifest.json":
            continue
        p = bundle / rel
        if not p.exists():
            errors.append(f"manifest file missing: {rel}")
            continue
        actual = sha256_file(p)
        if actual != expected_hash:
            errors.append(f"hash mismatch: {rel}")
    return errors


def validate_checkpoint(bundle: Path, run_id: str) -> list[str]:
    errors: list[str] = []
    cpath = bundle / "checkpoint_manifest.json"
    if not cpath.exists():
        return ["missing checkpoint_manifest.json"]
    try:
        data = read_json(cpath)
    except Exception as exc:
        return [f"checkpoint_manifest.json parse error: {exc}"]

    if data.get("readable") is not True:
        errors.append("checkpoint_manifest readable is not true")

    checkpoint = data.get("checkpoint")
    if not checkpoint:
        errors.append("checkpoint_manifest missing checkpoint")
        return errors

    payload = bundle / str(checkpoint)
    if not payload.exists():
        errors.append(f"checkpoint payload missing: {checkpoint}")
        return errors

    expected_hash = data.get("sha256")
    if expected_hash and sha256_file(payload) != expected_hash:
        errors.append("checkpoint payload hash mismatch")

    if data.get("run_id") and data["run_id"] != run_id:
        errors.append("checkpoint_manifest run_id mismatch")

    return errors


def validate_complete_written_last(bundle: Path) -> list[str]:
    complete = bundle / "COMPLETE.json"
    if not complete.exists():
        return ["missing COMPLETE.json"]
    ctime = complete.stat().st_mtime
    errors = []
    for p in bundle.iterdir():
        if p.is_file() and p.name != "COMPLETE.json":
            if p.stat().st_mtime > ctime + 1e-6:
                errors.append(f"COMPLETE.json is older than {p.name}")
    return errors


def validate_run_bundle_semantic(bundle: Path, expected_run_id: str | None = None) -> tuple[bool, list[str]]:
    errors: list[str] = []

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
        "COMPLETE.json",
    ]

    for name in required:
        if not (bundle / name).exists():
            errors.append(f"missing {name}")

    if errors:
        return False, errors

    try:
        run_manifest = read_json(bundle / "run_manifest.json")
        run_id = run_manifest["run_id"]
    except Exception as exc:
        return False, [f"bad run_manifest.json: {exc}"]

    if expected_run_id and run_id != expected_run_id:
        errors.append("run_manifest run_id mismatch expected")

    try:
        complete = read_json(bundle / "COMPLETE.json")
        if complete.get("run_id") != run_id:
            errors.append("COMPLETE.json run_id mismatch")
        if complete.get("artifact_validation") != "PASS":
            errors.append("COMPLETE.json artifact_validation not PASS")
    except Exception as exc:
        errors.append(f"bad COMPLETE.json: {exc}")

    try:
        exit_status = read_json(bundle / "exit_status.json")
        if exit_status.get("exit_code") != 0:
            errors.append("exit_status exit_code is not zero")
        if exit_status.get("run_id") and exit_status["run_id"] != run_id:
            errors.append("exit_status run_id mismatch")
    except Exception as exc:
        errors.append(f"bad exit_status.json: {exc}")

    errors.extend(validate_finite_csv(bundle / "metrics.csv", ["run_id"], run_id))
    errors.extend(validate_finite_csv(bundle / "site_observables.csv", ["run_id"], run_id))
    errors.extend(validate_finite_csv(bundle / "corr_observables.csv", ["run_id"], run_id))
    errors.extend(validate_finite_csv(bundle / "history.csv", ["run_id"], run_id))
    errors.extend(validate_checkpoint(bundle, run_id))
    errors.extend(validate_artifact_manifest(bundle))
    errors.extend(validate_complete_written_last(bundle))

    return len(errors) == 0, errors


def hash_artifacts_before_complete(bundle: Path) -> dict[str, str]:
    files = {}
    for p in sorted(bundle.iterdir()):
        if p.is_file() and p.name not in {"artifact_manifest.json", "COMPLETE.json"}:
            files[p.name] = sha256_file(p)
    return files


def write_valid_synthetic_bundle(root: Path, run: RunSpec) -> Path:
    tmp = Path(tempfile.mkdtemp(prefix=f".tmp_{run.run_id}_", dir=root))
    write_json(tmp / "run_manifest.json", run.to_dict())
    (tmp / "resolved_config.yaml").write_text("mock: true\n")
    write_json(tmp / "exit_status.json", {"run_id": run.run_id, "exit_code": 0, "ended_utc": utc_now()})
    (tmp / "metrics.csv").write_text(f"run_id,status,energy_error\n{run.run_id},VALID,0.0\n")
    (tmp / "site_observables.csv").write_text(f"run_id,site,value\n{run.run_id},0,0.0\n")
    (tmp / "corr_observables.csv").write_text(f"run_id,i,j,value\n{run.run_id},0,1,0.0\n")
    with (tmp / "history.csv").open("w") as f:
        f.write("run_id,step,energy\n")
        for i in range(14):
            f.write(f"{run.run_id},{i},0.0\n")
    (tmp / "checkpoint.npz").write_bytes(b"synthetic-checkpoint-payload")
    write_json(
        tmp / "checkpoint_manifest.json",
        {
            "run_id": run.run_id,
            "checkpoint": "checkpoint.npz",
            "sha256": sha256_file(tmp / "checkpoint.npz"),
            "readable": True,
        },
    )
    write_json(tmp / "artifact_manifest.json", {"files": hash_artifacts_before_complete(tmp)})
    time.sleep(0.02)
    write_json(
        tmp / "COMPLETE.json",
        {
            "run_id": run.run_id,
            "completed_utc": utc_now(),
            "artifact_validation": "PASS",
            "artifact_manifest_sha256": sha256_file(tmp / "artifact_manifest.json"),
        },
    )
    ok, errors = validate_run_bundle_semantic(tmp, run.run_id)
    if not ok:
        raise RuntimeError(f"synthetic valid bundle failed validation: {errors}")
    return tmp


def atomic_promote_no_overwrite(tmp: Path, final: Path) -> None:
    final.parent.mkdir(parents=True, exist_ok=True)
    if final.exists():
        raise FileExistsError(f"refusing to overwrite existing bundle: {final}")
    os.replace(tmp, final)


def write_release_manifest(root: Path) -> Path:
    manifest = root / "MANIFEST.sha256"
    rows = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name != "MANIFEST.sha256":
            rows.append(f"{sha256_file(p)}  {p.relative_to(root)}")
    manifest.write_text("\n".join(rows) + "\n")
    return manifest
