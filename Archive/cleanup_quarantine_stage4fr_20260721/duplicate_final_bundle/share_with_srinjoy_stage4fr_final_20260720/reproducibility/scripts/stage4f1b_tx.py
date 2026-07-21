from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import sqlite3
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml


SCHEMA_VERSION = "stage4f-native-v1c"
RUN_STATUSES = {
    "PENDING", "RUNNING", "FAILED_RETRYABLE", "FAILED_FINAL", "QUARANTINED", "COMPLETE"
}
MANDATORY_PAYLOADS = {
    "run_manifest.json",
    "resolved_config.yaml",
    "command.json",
    "stdout.log",
    "stderr.log",
    "exit_status.json",
    "metrics.csv",
    "checkpoint.npz",
    "checkpoint_manifest.json",
    "site_observables.csv",
    "corr_observables.csv",
    "history.csv",
}
MANDATORY_BUNDLE_FILES = MANDATORY_PAYLOADS | {"artifact_manifest.json", "COMPLETE.json"}
IDENTITY_PATHS = (
    "pyproject.toml",
    "reproducibility/environment_stage4f_portable.yml",
    "scripts/run_stage4e.py",
    "scripts/stage4f1b_tx.py",
    "scripts/stage4f1b_worker.py",
    "scripts/stage4f1b_campaign.py",
    "scripts/stage4f1c_tx.py",
    "scripts/stage4f1c_worker.py",
    "scripts/stage4f1c_campaign.py",
    "scripts/stage4f1c_crash_probe.py",
    "src/graphkan_amo_nqs",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def now_epoch() -> float:
    return time.time()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json_bytes(data: Any) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def _identity_ledger(project_root: Path) -> dict[str, str | None]:
    ledger: dict[str, str | None] = {}
    for rel in IDENTITY_PATHS:
        path = project_root / rel
        if path.is_file():
            ledger[rel] = sha256_file(path)
        elif path.is_dir():
            for child in sorted(path.rglob("*")):
                if not child.is_file() or "__pycache__" in child.parts or child.suffix in {".pyc", ".pyo"}:
                    continue
                ledger[child.relative_to(project_root).as_posix()] = sha256_file(child)
        else:
            ledger[rel] = None
    return ledger


def protocol_identity(config: dict[str, Any], project_root: Path = Path(".")) -> dict[str, Any]:
    project_root = project_root.resolve()
    base_configs: dict[str, Any] = {}
    for key, rel in sorted(config.get("base_configs", {}).items()):
        path = project_root / str(rel)
        base_configs[str(key)] = {
            "relative_path": Path(str(rel)).as_posix(),
            "sha256": sha256_file(path) if path.is_file() else None,
        }
    return {
        "schema_version": config.get("campaign", {}).get("schema_version", SCHEMA_VERSION),
        "physics": config.get("physics", {}),
        "models": config.get("models", {}),
        "scheduler": config.get("scheduler", {}),
        "runner_overrides": config.get("runner_overrides", {}),
        "completion_requirements": config.get("completion_requirements", {}),
        "base_configs": base_configs,
        "source_ledger": _identity_ledger(project_root),
    }


def protocol_hash_from_config(config: dict[str, Any], project_root: Path = Path(".")) -> str:
    return sha256_bytes(canonical_json_bytes(protocol_identity(config, project_root)))


def safe_id(text: Any) -> str:
    return str(text).replace(".", "p").replace("/", "_").replace(":", "_").replace(" ", "_")


def token_hash(token: str) -> str:
    return sha256_bytes(token.encode("utf-8"))


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
    n_sites: int = 10
    expected_history_rows: int = 14

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
            "n_sites": self.n_sites,
            "expected_history_rows": self.expected_history_rows,
        }

    @property
    def spec_hash(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_dict()))


def make_campaign_id(config: dict[str, Any], project_root: Path = Path(".")) -> str:
    name = config["campaign"]["name"]
    schema = config["campaign"].get("schema_version", SCHEMA_VERSION)
    return f"{name}__{schema}__P{protocol_hash_from_config(config, project_root)[:16]}"


def _history_rows_for_boundary(config: dict[str, Any], boundary: str, project_root: Path) -> int:
    rel = config.get("base_configs", {}).get(boundary)
    if rel and (project_root / rel).is_file():
        base = load_config(project_root / rel)
        training = base.get("training", {})
        training = {**training, **config.get("runner_overrides", {}).get("training", {})}
        n_iter = int(training.get("n_iter", 1))
        chunk = max(1, int(training.get("chunk_size", n_iter)))
        return int(math.ceil(n_iter / chunk))
    return int(config.get("completion_requirements", {}).get("history_rows", 1))


def expected_runs(config: dict[str, Any], project_root: Path = Path(".")) -> list[RunSpec]:
    phash = protocol_hash_from_config(config, project_root)
    campaign_id = make_campaign_id(config, project_root)
    physics = config["physics"]
    runs: list[RunSpec] = []
    priority = 0
    for seed in physics["seeds"]:
        for boundary in physics["boundaries"]:
            for delta in physics["deltas"]:
                for c6 in physics["c6_values"]:
                    for model, model_cfg in config["models"].items():
                        runs.append(
                            RunSpec(
                                campaign_id=campaign_id,
                                protocol_hash=phash,
                                boundary=str(boundary),
                                delta=f"{float(delta):.2f}",
                                c6=str(int(float(c6))),
                                model=str(model),
                                seed=int(seed),
                                lane=str(model_cfg.get("lane", "core")),
                                priority=priority,
                                n_sites=int(physics["n_sites"]),
                                expected_history_rows=_history_rows_for_boundary(config, str(boundary), project_root),
                            )
                        )
                        priority += 1
    return runs


class Stage4F1BRegistry:
    """SQLite state machine with claim/attempt fencing.

    SQLite WAL is supported only on a local filesystem. Multi-host execution must
    use one registry per host and merge immutable bundles offline.
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path, timeout=30)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys=ON")
        return con

    def init(self) -> None:
        with self.connect() as con:
            con.execute("PRAGMA journal_mode=WAL")
            con.execute(
                """CREATE TABLE IF NOT EXISTS campaigns (
                    campaign_id TEXT PRIMARY KEY,
                    protocol_hash TEXT NOT NULL,
                    manifest_sha256 TEXT NOT NULL,
                    created_utc TEXT NOT NULL
                )"""
            )
            con.execute(
                """CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    campaign_id TEXT NOT NULL,
                    protocol_hash TEXT NOT NULL,
                    expected_spec_hash TEXT NOT NULL,
                    expected_spec_json TEXT NOT NULL,
                    boundary TEXT NOT NULL,
                    delta TEXT NOT NULL,
                    c6 TEXT NOT NULL,
                    model TEXT NOT NULL,
                    seed INTEGER NOT NULL,
                    lane TEXT NOT NULL,
                    priority INTEGER NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('PENDING','RUNNING','FAILED_RETRYABLE','FAILED_FINAL','QUARANTINED','COMPLETE')),
                    attempt INTEGER NOT NULL DEFAULT 0,
                    attempt_id TEXT,
                    claim_token_hash TEXT,
                    worker_id TEXT,
                    lease_expires_epoch REAL,
                    heartbeat_utc TEXT,
                    claimed_utc TEXT,
                    completed_utc TEXT,
                    next_retry_epoch REAL DEFAULT 0,
                    max_attempts INTEGER NOT NULL DEFAULT 3,
                    bundle_path TEXT,
                    complete_hash TEXT,
                    error TEXT,
                    UNIQUE(campaign_id, boundary, delta, c6, model, seed)
                )"""
            )
            con.execute(
                """CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,
                    event_utc TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    attempt_id TEXT,
                    detail TEXT
                )"""
            )

    def register_campaign(self, config_path: Path, config: dict[str, Any], project_root: Path = Path(".")) -> dict[str, Any]:
        self.init()
        campaign_id = make_campaign_id(config, project_root)
        protocol_hash = protocol_hash_from_config(config, project_root)
        runs = expected_runs(config, project_root)
        manifest = {
            "schema_version": config["campaign"].get("schema_version", SCHEMA_VERSION),
            "campaign_id": campaign_id,
            "protocol_hash": protocol_hash,
            "protocol_identity": protocol_identity(config, project_root),
            "created_utc": utc_now(),
            "config_name": config_path.name,
            "config_sha256": sha256_file(config_path),
            "expected_runs": len(runs),
            "claim_boundary": config["campaign"].get("claim_boundary", "engineering gate only"),
        }
        manifest_hash = sha256_bytes(canonical_json_bytes(manifest))
        max_attempts = int(config.get("scheduler", {}).get("max_attempts", 3))
        if max_attempts < 1:
            raise ValueError("scheduler.max_attempts must be >= 1")
        with self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            existing = con.execute("SELECT * FROM campaigns").fetchall()
            if existing and (
                existing[0]["campaign_id"] != campaign_id or existing[0]["protocol_hash"] != protocol_hash
            ):
                raise RuntimeError("Protocol mutation detected: refusing to modify existing campaign registry.")
            if not existing:
                con.execute(
                    "INSERT INTO campaigns(campaign_id, protocol_hash, manifest_sha256, created_utc) VALUES (?, ?, ?, ?)",
                    (campaign_id, protocol_hash, manifest_hash, utc_now()),
                )
            for run in runs:
                spec_json = canonical_json_bytes(run.to_dict()).decode("utf-8")
                con.execute(
                    """INSERT OR IGNORE INTO runs
                    (run_id,campaign_id,protocol_hash,expected_spec_hash,expected_spec_json,boundary,delta,c6,model,seed,lane,priority,status,max_attempts)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?, 'PENDING', ?)""",
                    (
                        run.run_id, run.campaign_id, run.protocol_hash, run.spec_hash, spec_json,
                        run.boundary, run.delta, run.c6, run.model, run.seed, run.lane, run.priority,
                        max_attempts,
                    ),
                )
            con.commit()
        return manifest

    def _event(self, con: sqlite3.Connection, run_id: str, event: str, attempt_id: str | None, detail: str) -> None:
        con.execute(
            "INSERT INTO events(run_id,event_utc,event_type,attempt_id,detail) VALUES (?,?,?,?,?)",
            (run_id, utc_now(), event, attempt_id, detail[:2000]),
        )

    def claim_next(self, lane: str, worker_id: str, lease_seconds: int = 300) -> dict[str, Any] | None:
        if lease_seconds < 1:
            raise ValueError("lease_seconds must be >= 1")
        with self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            now = now_epoch()
            stale_rows = con.execute(
                "SELECT * FROM runs WHERE status='RUNNING' AND lane=? AND lease_expires_epoch < ? ORDER BY priority",
                (lane, now),
            ).fetchall()
            for stale in stale_rows:
                terminal = int(stale["attempt"]) >= int(stale["max_attempts"])
                status = "QUARANTINED" if terminal else "FAILED_RETRYABLE"
                con.execute(
                    """UPDATE runs SET status=?,worker_id=NULL,attempt_id=NULL,claim_token_hash=NULL,
                    lease_expires_epoch=NULL,next_retry_epoch=?,error=? WHERE run_id=? AND status='RUNNING'""",
                    (status, now, "stale lease reclaimed", stale["run_id"]),
                )
                self._event(con, stale["run_id"], status, stale["attempt_id"], "stale lease reclaimed")
            while True:
                row = con.execute(
                    """SELECT * FROM runs WHERE lane=? AND
                    (status='PENDING' OR (status='FAILED_RETRYABLE' AND next_retry_epoch <= ?))
                    ORDER BY CASE WHEN status='PENDING' THEN 0 ELSE 1 END, priority LIMIT 1""",
                    (lane, now),
                ).fetchone()
                if row is None:
                    con.commit()
                    return None
                attempt = int(row["attempt"]) + 1
                if attempt > int(row["max_attempts"]):
                    con.execute("UPDATE runs SET status='QUARANTINED',error=? WHERE run_id=?", ("max attempts exceeded", row["run_id"]))
                    self._event(con, row["run_id"], "QUARANTINED", None, "max attempts exceeded")
                    continue
                attempt_id = uuid.uuid4().hex
                claim_token = uuid.uuid4().hex + uuid.uuid4().hex
                result = con.execute(
                    """UPDATE runs SET status='RUNNING',attempt=?,attempt_id=?,claim_token_hash=?,worker_id=?,
                    lease_expires_epoch=?,heartbeat_utc=?,claimed_utc=?,error=NULL WHERE run_id=? AND status IN ('PENDING','FAILED_RETRYABLE')""",
                    (
                        attempt, attempt_id, token_hash(claim_token), worker_id, now + lease_seconds,
                        utc_now(), utc_now(), row["run_id"],
                    ),
                )
                if result.rowcount != 1:
                    continue
                self._event(con, row["run_id"], "CLAIMED", attempt_id, f"worker={worker_id};attempt={attempt}")
                claimed = dict(con.execute("SELECT * FROM runs WHERE run_id=?", (row["run_id"],)).fetchone())
                claimed["claim_token"] = claim_token
                claimed["expected_spec"] = json.loads(claimed["expected_spec_json"])
                con.commit()
                return claimed

    def mark_heartbeat(self, run_id: str, worker_id: str, attempt_id: str, claim_token: str, lease_seconds: int = 300) -> bool:
        with self.connect() as con:
            result = con.execute(
                """UPDATE runs SET heartbeat_utc=?,lease_expires_epoch=?
                WHERE run_id=? AND worker_id=? AND attempt_id=? AND claim_token_hash=? AND status='RUNNING'""",
                (utc_now(), now_epoch() + lease_seconds, run_id, worker_id, attempt_id, token_hash(claim_token)),
            )
            return result.rowcount == 1

    def owns_claim(self, run_id: str, worker_id: str, attempt_id: str, claim_token: str) -> bool:
        with self.connect() as con:
            row = con.execute(
                """SELECT 1 FROM runs WHERE run_id=? AND worker_id=? AND attempt_id=?
                AND claim_token_hash=? AND status='RUNNING' AND lease_expires_epoch >= ?""",
                (run_id, worker_id, attempt_id, token_hash(claim_token), now_epoch()),
            ).fetchone()
        return row is not None

    def mark_complete(self, run_id: str, worker_id: str, attempt_id: str, claim_token: str, bundle_path: Path, complete_hash: str) -> None:
        with self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            result = con.execute(
                """UPDATE runs SET status='COMPLETE',completed_utc=?,bundle_path=?,complete_hash=?,
                lease_expires_epoch=NULL WHERE run_id=? AND worker_id=? AND attempt_id=?
                AND claim_token_hash=? AND status='RUNNING' AND lease_expires_epoch >= ?""",
                (
                    utc_now(), str(bundle_path), complete_hash, run_id, worker_id, attempt_id,
                    token_hash(claim_token), now_epoch(),
                ),
            )
            if result.rowcount != 1:
                raise RuntimeError(f"lost or expired claim for {run_id}; completion fenced")
            self._event(con, run_id, "COMPLETE", attempt_id, str(bundle_path))
            con.commit()

    def mark_failed(self, run_id: str, worker_id: str, attempt_id: str, claim_token: str, error: str, retryable: bool = True, backoff_seconds: int = 60) -> bool:
        with self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if row is None:
                return False
            status = "FAILED_RETRYABLE" if retryable else "FAILED_FINAL"
            if int(row["attempt"]) >= int(row["max_attempts"]):
                status = "QUARANTINED"
            result = con.execute(
                """UPDATE runs SET status=?,error=?,worker_id=NULL,attempt_id=NULL,claim_token_hash=NULL,
                lease_expires_epoch=NULL,next_retry_epoch=? WHERE run_id=? AND worker_id=? AND attempt_id=?
                AND claim_token_hash=? AND status='RUNNING'""",
                (
                    status, error[:2000], now_epoch() + max(0, backoff_seconds), run_id, worker_id,
                    attempt_id, token_hash(claim_token),
                ),
            )
            if result.rowcount:
                self._event(con, run_id, status, attempt_id, error)
            con.commit()
            return result.rowcount == 1

    def run_rows(self) -> list[dict[str, Any]]:
        with self.connect() as con:
            return [dict(row) for row in con.execute("SELECT * FROM runs ORDER BY priority")]

    def summary(self) -> dict[str, Any]:
        with self.connect() as con:
            rows = con.execute("SELECT status,COUNT(*) AS n FROM runs GROUP BY status ORDER BY status").fetchall()
            total = con.execute("SELECT COUNT(*) AS n FROM runs").fetchone()["n"]
            duplicates = con.execute(
                """SELECT COUNT(*) AS n FROM (
                SELECT campaign_id,boundary,delta,c6,model,seed,COUNT(*) AS c FROM runs
                GROUP BY campaign_id,boundary,delta,c6,model,seed HAVING c > 1)"""
            ).fetchone()["n"]
        return {"total_expected": total, "status_counts": {r["status"]: r["n"] for r in rows}, "duplicate_registry_keys": duplicates}


def _safe_relative_path(rel: str) -> bool:
    path = Path(rel)
    return bool(rel) and not path.is_absolute() and ".." not in path.parts and path.as_posix() == rel


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _finite_number(value: str) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _check_columns(path: Path, rows: list[dict[str, str]], required: set[str]) -> list[str]:
    if not rows:
        return [f"{path.name}: no rows"]
    missing = sorted(required - set(rows[0]))
    return [f"{path.name}: missing column {name}" for name in missing]


def validate_artifact_manifest(bundle: Path) -> list[str]:
    try:
        manifest = read_json(bundle / "artifact_manifest.json")
    except Exception as exc:
        return [f"artifact_manifest.json parse error: {exc}"]
    files = manifest.get("files")
    if not isinstance(files, dict):
        return ["artifact_manifest.json missing files object"]
    errors: list[str] = []
    if set(files) != MANDATORY_PAYLOADS:
        errors.append(f"artifact manifest coverage mismatch: expected {sorted(MANDATORY_PAYLOADS)}, got {sorted(files)}")
    actual_files = {p.name for p in bundle.iterdir() if p.is_file()}
    if actual_files != MANDATORY_BUNDLE_FILES:
        errors.append(f"bundle file set mismatch: unexpected/missing files {sorted(actual_files ^ MANDATORY_BUNDLE_FILES)}")
    for rel, expected_hash in files.items():
        if not isinstance(rel, str) or not _safe_relative_path(rel):
            errors.append(f"unsafe manifest path: {rel!r}")
            continue
        path = bundle / rel
        if not path.is_file():
            errors.append(f"manifest file missing: {rel}")
        elif not isinstance(expected_hash, str) or len(expected_hash) != 64 or sha256_file(path) != expected_hash:
            errors.append(f"hash mismatch: {rel}")
    return errors


def validate_checkpoint(bundle: Path, run_id: str, expected_model: str) -> list[str]:
    errors: list[str] = []
    try:
        meta = read_json(bundle / "checkpoint_manifest.json")
    except Exception as exc:
        return [f"checkpoint_manifest.json parse error: {exc}"]
    if meta.get("schema_version") != "jax-tree-leaves-npz-v1":
        errors.append("checkpoint schema_version mismatch")
    if meta.get("run_id") != run_id or meta.get("model") != expected_model:
        errors.append("checkpoint identity mismatch")
    if meta.get("checkpoint") != "checkpoint.npz":
        errors.append("checkpoint path must be checkpoint.npz")
    payload = bundle / "checkpoint.npz"
    if not payload.is_file():
        return errors + ["checkpoint payload missing"]
    if meta.get("sha256") != sha256_file(payload):
        errors.append("checkpoint payload hash mismatch")
    try:
        with np.load(payload, allow_pickle=False) as archive:
            names = archive.files
            if not names:
                errors.append("checkpoint has no arrays")
            for name in names:
                array = np.asarray(archive[name])
                if array.dtype.kind not in "biufc" or not np.all(np.isfinite(array)):
                    errors.append(f"checkpoint array {name} is not finite numeric data")
            if meta.get("array_count") != len(names):
                errors.append("checkpoint array_count mismatch")
            if meta.get("arrays") != {name: list(np.asarray(archive[name]).shape) for name in names}:
                errors.append("checkpoint array shape summary mismatch")
    except Exception as exc:
        errors.append(f"checkpoint restore failed: {type(exc).__name__}: {exc}")
    return errors


def validate_complete_written_last(bundle: Path) -> list[str]:
    complete = bundle / "COMPLETE.json"
    if not complete.is_file():
        return ["missing COMPLETE.json"]
    complete_time = complete.stat().st_mtime_ns
    return [f"COMPLETE.json is older than {p.name}" for p in bundle.iterdir() if p.is_file() and p.name != "COMPLETE.json" and p.stat().st_mtime_ns > complete_time]


def _compare_identity(actual: dict[str, Any], expected: dict[str, Any], prefix: str) -> list[str]:
    errors = []
    for key in ("campaign_id", "protocol_hash", "run_id", "boundary", "delta", "c6", "model", "seed", "lane", "n_sites", "expected_history_rows"):
        if actual.get(key) != expected.get(key):
            errors.append(f"{prefix} {key} mismatch")
    return errors


def validate_run_bundle_semantic(
    bundle: Path,
    expected_run_id: str | None = None,
    *,
    expected_spec: dict[str, Any] | None = None,
    expected_attempt_id: str | None = None,
    expected_claim_token: str | None = None,
) -> tuple[bool, list[str]]:
    errors: list[str] = []
    missing = [name for name in sorted(MANDATORY_BUNDLE_FILES) if not (bundle / name).is_file()]
    if missing:
        return False, [f"missing {name}" for name in missing]
    try:
        run_manifest = read_json(bundle / "run_manifest.json")
    except Exception as exc:
        return False, [f"bad run_manifest.json: {exc}"]
    run_id = run_manifest.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        return False, ["run_manifest run_id missing"]
    if run_manifest.get("schema_version") != SCHEMA_VERSION:
        errors.append("run_manifest schema_version mismatch")
    if expected_run_id and run_id != expected_run_id:
        errors.append("run_manifest run_id mismatch expected")
    if expected_spec:
        errors.extend(_compare_identity(run_manifest, expected_spec, "run_manifest"))
        if run_manifest.get("expected_spec_hash") != sha256_bytes(canonical_json_bytes(expected_spec)):
            errors.append("run_manifest expected_spec_hash mismatch")
    if expected_attempt_id and run_manifest.get("attempt_id") != expected_attempt_id:
        errors.append("run_manifest attempt_id mismatch")
    if expected_claim_token and run_manifest.get("claim_token_hash") != token_hash(expected_claim_token):
        errors.append("run_manifest claim token mismatch")
    try:
        resolved = load_config(bundle / "resolved_config.yaml")
        identity = resolved.get("transaction_identity", {})
        if expected_spec:
            errors.extend(_compare_identity(identity, expected_spec, "resolved_config"))
        if identity.get("attempt_id") != run_manifest.get("attempt_id"):
            errors.append("resolved_config attempt_id mismatch")
    except Exception as exc:
        errors.append(f"bad resolved_config.yaml: {exc}")
    try:
        complete = read_json(bundle / "COMPLETE.json")
        if complete.get("schema_version") != SCHEMA_VERSION or complete.get("run_id") != run_id:
            errors.append("COMPLETE identity mismatch")
        if complete.get("attempt_id") != run_manifest.get("attempt_id"):
            errors.append("COMPLETE attempt_id mismatch")
        if complete.get("artifact_manifest_sha256") != sha256_file(bundle / "artifact_manifest.json"):
            errors.append("COMPLETE artifact manifest hash mismatch")
        if complete.get("artifact_validation") != "PASS":
            errors.append("COMPLETE artifact_validation not PASS")
    except Exception as exc:
        errors.append(f"bad COMPLETE.json: {exc}")
    try:
        exit_status = read_json(bundle / "exit_status.json")
        if exit_status.get("run_id") != run_id or exit_status.get("attempt_id") != run_manifest.get("attempt_id"):
            errors.append("exit_status identity mismatch")
        if exit_status.get("exit_code") != 0 or exit_status.get("status") != "COMPLETED":
            errors.append("exit_status is not a successful completion")
    except Exception as exc:
        errors.append(f"bad exit_status.json: {exc}")

    try:
        metrics = read_csv_dicts(bundle / "metrics.csv")
        errors.extend(_check_columns(bundle / "metrics.csv", metrics, {"run_id", "status", "scientific_validity", "model", "boundary", "delta", "c6", "seed", "energy_mean", "energy_error"}))
        if len(metrics) != 1:
            errors.append("metrics.csv must contain exactly one row")
        for row in metrics:
            if row.get("run_id") != run_id or row.get("status") != "COMPLETED":
                errors.append("metrics.csv identity/status mismatch")
            if row.get("scientific_validity") not in {"VALID", "FLAGGED"}:
                errors.append("metrics.csv invalid scientific_validity enum")
            if expected_spec:
                for key in ("model", "boundary", "delta", "c6", "seed"):
                    if str(row.get(key)) != str(expected_spec[key]):
                        errors.append(f"metrics.csv {key} mismatch")
            for key in ("energy_mean", "energy_error"):
                if not _finite_number(row.get(key, "")):
                    errors.append(f"metrics.csv non-finite/non-numeric {key}")
    except Exception as exc:
        errors.append(f"metrics.csv parse error: {exc}")

    n_sites = int((expected_spec or run_manifest).get("n_sites", 0))
    history_count = int((expected_spec or run_manifest).get("expected_history_rows", 0))
    for name, required, count in (
        ("history.csv", {"run_id", "step", "energy_mean", "energy_error"}, history_count),
        ("site_observables.csv", {"run_id", "site", "ed_density", "exact_model_density"}, n_sites),
        ("corr_observables.csv", {"run_id", "i", "j", "ed_connected", "exact_model_connected"}, n_sites * n_sites),
    ):
        try:
            rows = read_csv_dicts(bundle / name)
            errors.extend(_check_columns(bundle / name, rows, required))
            if len(rows) != count:
                errors.append(f"{name}: expected exactly {count} rows, got {len(rows)}")
            for index, row in enumerate(rows):
                if row.get("run_id") != run_id:
                    errors.append(f"{name}: row {index} run_id mismatch")
                for key, value in row.items():
                    if key not in {"run_id", "runner_run_id", "status", "model", "boundary", "profile", "stage", "schema_version", "model_label", "model_kind", "basis_family", "instance_id", "scientific_validity_exact", "scientific_validity_sample"} and value != "" and not _finite_number(value):
                        errors.append(f"{name}: row {index} non-numeric {key}")
        except Exception as exc:
            errors.append(f"{name} parse error: {exc}")
    model = str((expected_spec or run_manifest).get("model", ""))
    errors.extend(validate_checkpoint(bundle, run_id, model))
    errors.extend(validate_artifact_manifest(bundle))
    errors.extend(validate_complete_written_last(bundle))
    return not errors, errors


def hash_artifacts_before_complete(bundle: Path) -> dict[str, str]:
    return {name: sha256_file(bundle / name) for name in sorted(MANDATORY_PAYLOADS)}


def _checkpoint_metadata(path: Path, run: RunSpec) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: list(np.asarray(archive[name]).shape) for name in archive.files}
    return {
        "schema_version": "jax-tree-leaves-npz-v1",
        "run_id": run.run_id,
        "model": run.model,
        "checkpoint": "checkpoint.npz",
        "sha256": sha256_file(path),
        "array_count": len(arrays),
        "arrays": arrays,
    }


def write_valid_synthetic_bundle(
    root: Path,
    run: RunSpec,
    *,
    attempt_id: str = "synthetic-attempt",
    claim_token: str = "synthetic-claim-token",
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".tmp_{safe_id(run.run_id)}_", dir=root))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        **run.to_dict(),
        "expected_spec_hash": run.spec_hash,
        "attempt_id": attempt_id,
        "claim_token_hash": token_hash(claim_token),
        "created_utc": utc_now(),
    }
    write_json(tmp / "run_manifest.json", manifest)
    resolved = {
        "synthetic": True,
        "transaction_identity": {**run.to_dict(), "attempt_id": attempt_id},
    }
    (tmp / "resolved_config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=True))
    write_json(tmp / "command.json", {"argv": ["synthetic"], "run_id": run.run_id, "attempt_id": attempt_id})
    (tmp / "stdout.log").write_text("synthetic fixture\n")
    (tmp / "stderr.log").write_text("")
    write_json(tmp / "exit_status.json", {"run_id": run.run_id, "attempt_id": attempt_id, "exit_code": 0, "status": "COMPLETED", "ended_utc": utc_now()})
    (tmp / "metrics.csv").write_text(
        f"run_id,status,scientific_validity,model,boundary,delta,c6,seed,energy_mean,energy_error\n"
        f"{run.run_id},COMPLETED,VALID,{run.model},{run.boundary},{run.delta},{run.c6},{run.seed},-1.0,0.01\n"
    )
    with (tmp / "history.csv").open("w") as handle:
        handle.write("run_id,step,energy_mean,energy_error\n")
        for step in range(run.expected_history_rows):
            handle.write(f"{run.run_id},{step + 1},-1.0,0.01\n")
    with (tmp / "site_observables.csv").open("w") as handle:
        handle.write("run_id,site,ed_density,exact_model_density\n")
        for site in range(run.n_sites):
            handle.write(f"{run.run_id},{site},0.5,0.5\n")
    with (tmp / "corr_observables.csv").open("w") as handle:
        handle.write("run_id,i,j,ed_connected,exact_model_connected\n")
        for i in range(run.n_sites):
            for j in range(run.n_sites):
                handle.write(f"{run.run_id},{i},{j},0.0,0.0\n")
    np.savez_compressed(tmp / "checkpoint.npz", leaf_0000=np.zeros((2, 2), dtype=np.float64))
    write_json(tmp / "checkpoint_manifest.json", _checkpoint_metadata(tmp / "checkpoint.npz", run))
    write_json(tmp / "artifact_manifest.json", {"schema_version": SCHEMA_VERSION, "files": hash_artifacts_before_complete(tmp)})
    time.sleep(0.002)
    write_json(tmp / "COMPLETE.json", {
        "schema_version": SCHEMA_VERSION,
        "run_id": run.run_id,
        "attempt_id": attempt_id,
        "completed_utc": utc_now(),
        "artifact_validation": "PASS",
        "artifact_manifest_sha256": sha256_file(tmp / "artifact_manifest.json"),
    })
    ok, errors = validate_run_bundle_semantic(
        tmp,
        expected_spec=run.to_dict(),
        expected_attempt_id=attempt_id,
        expected_claim_token=claim_token,
    )
    if not ok:
        raise RuntimeError(f"synthetic valid bundle failed validation: {errors}")
    return tmp


def _lock_owner_is_alive(lock: Path) -> bool:
    try:
        text = lock.read_text()
        pid = int(text.split("pid=", 1)[1].split()[0])
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except (FileNotFoundError, PermissionError, ValueError, IndexError):
        return True


def atomic_promote_no_overwrite(
    tmp: Path,
    final: Path,
    quarantine_root: Path | None = None,
    *,
    fault_after_lock: bool = False,
) -> None:
    final.parent.mkdir(parents=True, exist_ok=True)
    lock = final.parent / f".{final.name}.publish.lock"
    fd: int | None = None
    try:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            if _lock_owner_is_alive(lock):
                raise
            lock.unlink()
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.write(fd, f"pid={os.getpid()} utc={utc_now()}\n".encode())
        os.close(fd)
        fd = None
        if fault_after_lock:
            import signal
            os.kill(os.getpid(), signal.SIGKILL)
        if final.exists():
            if quarantine_root is None:
                raise FileExistsError(f"refusing to overwrite existing bundle: {final}")
            quarantine_root.mkdir(parents=True, exist_ok=True)
            preserved = quarantine_root / f"{final.name}__collision__{uuid.uuid4().hex}"
            os.rename(final, preserved)
        os.rename(tmp, final)
    finally:
        if fd is not None:
            os.close(fd)
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


def reconcile_campaign(root: Path, reg: Stage4F1BRegistry) -> dict[str, Any]:
    rows = reg.run_rows()
    by_id = {row["run_id"]: row for row in rows}
    errors: list[dict[str, Any]] = []
    checked = 0
    runs_root = root / "runs"
    filesystem_ids = {p.name for p in runs_root.iterdir() if p.is_dir()} if runs_root.exists() else set()
    for run_id, row in by_id.items():
        bundle = runs_root / run_id
        if row["status"] == "COMPLETE":
            if not bundle.is_dir():
                errors.append({"run_id": run_id, "error": "registry COMPLETE but bundle missing"})
                continue
            checked += 1
            expected = json.loads(row["expected_spec_json"])
            ok, bundle_errors = validate_run_bundle_semantic(bundle, expected_spec=expected, expected_attempt_id=row["attempt_id"])
            if not ok:
                errors.append({"run_id": run_id, "error": "invalid COMPLETE bundle", "details": bundle_errors})
            complete_path = bundle / "COMPLETE.json"
            if complete_path.is_file():
                actual_hash = sha256_file(complete_path)
                if row["complete_hash"] != actual_hash:
                    errors.append({"run_id": run_id, "error": "registry complete_hash mismatch"})
            else:
                errors.append({"run_id": run_id, "error": "registry COMPLETE but COMPLETE.json is missing"})
            if Path(str(row["bundle_path"])).resolve() != bundle.resolve():
                errors.append({"run_id": run_id, "error": "registry bundle_path mismatch"})
        elif bundle.exists():
            errors.append({"run_id": run_id, "error": f"canonical bundle exists while registry status is {row['status']}"})
    for orphan in sorted(filesystem_ids - set(by_id)):
        errors.append({"run_id": orphan, "error": "filesystem bundle has no registry work item"})
    locks = sorted(p.name for p in runs_root.glob(".*.publish.lock")) if runs_root.exists() else []
    if locks:
        errors.append({"run_id": None, "error": "stale publication locks", "details": locks})
    return {"complete_bundles_checked": checked, "registry_filesystem_errors": errors}


def write_release_manifest(root: Path) -> Path:
    manifest = root / "MANIFEST.sha256"
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != manifest.name and "__pycache__" not in path.parts and path.suffix != ".pyc":
            rows.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}")
    manifest.write_text("\n".join(rows) + "\n")
    return manifest
