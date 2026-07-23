from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import hashlib
from pathlib import Path

import pytest
import yaml

from stage4f1b_tx import (
    Stage4F1BRegistry,
    atomic_promote_no_overwrite,
    expected_runs,
    load_config,
    protocol_hash_from_config,
    reconcile_campaign,
    sha256_file,
    validate_run_bundle_semantic,
    write_valid_synthetic_bundle,
)


def one_run_config(tmp_path: Path) -> tuple[dict, Path]:
    cfg = load_config(Path("configs/stage4f1c_campaign.yaml"))
    cfg["campaign"]["output_root"] = str(tmp_path / "campaign")
    cfg["physics"]["boundaries"] = ["open"]
    cfg["physics"]["deltas"] = ["1.50"]
    cfg["physics"]["c6_values"] = ["4"]
    cfg["physics"]["seeds"] = [0]
    cfg["models"] = {"rbf_radial_b32": {"lane": "core"}}
    path = tmp_path / "campaign.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return cfg, path


def test_protocol_hash_is_path_context_independent(tmp_path):
    cfg, _ = one_run_config(tmp_path)
    assert protocol_hash_from_config(cfg, Path(".")) == protocol_hash_from_config(cfg, Path(".").resolve())


def test_configured_max_attempts_is_stored(tmp_path):
    cfg, path = one_run_config(tmp_path)
    cfg["scheduler"]["max_attempts"] = 7
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    registry = Stage4F1BRegistry(Path(cfg["campaign"]["output_root"]) / "registry.sqlite")
    registry.register_campaign(path, cfg)
    assert registry.run_rows()[0]["max_attempts"] == 7


def test_explicit_physics_cells_do_not_expand_to_cartesian_product(tmp_path):
    cfg, _ = one_run_config(tmp_path)
    cfg["physics"]["boundaries"] = ["open", "periodic"]
    cfg["physics"]["cells"] = [
        {"delta": "1.25", "c6": "3"},
        {"delta": "2.00", "c6": "5"},
    ]
    cfg["models"] = {
        "rbf_radial_b32": {"lane": "core"},
        "endpoint_full_b32": {"lane": "core"},
        "mlp_endpoint_large": {"lane": "core"},
        "rbm_h32": {"lane": "control"},
    }
    runs = expected_runs(cfg)
    assert len(runs) == 16
    assert {(run.delta, run.c6) for run in runs} == {("1.25", "3"), ("2.00", "5")}
    assert sum(run.lane == "core" for run in runs) == 12
    assert sum(run.lane == "control" for run in runs) == 4


def test_release_campaign_ledger_exactly_equals_frozen_campaign_manifest():
    campaign = json.loads(Path("results_stage4f_r/frozen/campaign_manifest.json").read_text())
    release = json.loads(Path("release_integrity/frozen_source_ledger.json").read_text())
    assert release["source_ledger"] == campaign["protocol_identity"]["source_ledger"]
    assert len(release["source_ledger"]) == 105


def test_late_stale_worker_is_fenced(tmp_path):
    cfg, path = one_run_config(tmp_path)
    registry = Stage4F1BRegistry(Path(cfg["campaign"]["output_root"]) / "registry.sqlite")
    registry.register_campaign(path, cfg)
    first = registry.claim_next("core", "old", lease_seconds=1)
    assert first
    time.sleep(1.05)
    second = registry.claim_next("core", "new", lease_seconds=5)
    assert second and second["attempt_id"] != first["attempt_id"]
    assert not registry.mark_heartbeat(first["run_id"], "old", first["attempt_id"], first["claim_token"], 5)
    with pytest.raises(RuntimeError, match="fenced"):
        registry.mark_complete(first["run_id"], "old", first["attempt_id"], first["claim_token"], tmp_path / "late", "0" * 64)


def test_reconciliation_detects_missing_bundle_and_hash_mismatch(tmp_path):
    cfg, path = one_run_config(tmp_path)
    root = Path(cfg["campaign"]["output_root"])
    registry = Stage4F1BRegistry(root / "registry.sqlite")
    registry.register_campaign(path, cfg)
    row = registry.claim_next("core", "worker", lease_seconds=30)
    assert row
    run = expected_runs(cfg)[0]
    tmp = write_valid_synthetic_bundle(root / "attempts", run, attempt_id=row["attempt_id"], claim_token=row["claim_token"])
    final = root / "runs" / run.run_id
    atomic_promote_no_overwrite(tmp, final)
    registry.mark_complete(run.run_id, "worker", row["attempt_id"], row["claim_token"], final, sha256_file(final / "COMPLETE.json"))
    assert not reconcile_campaign(root, registry)["registry_filesystem_errors"]
    (final / "COMPLETE.json").unlink()
    errors = reconcile_campaign(root, registry)["registry_filesystem_errors"]
    assert any("invalid COMPLETE bundle" in item["error"] or "complete_hash" in item["error"] for item in errors)


def test_concurrent_publication_has_one_winner(tmp_path):
    cfg, _ = one_run_config(tmp_path)
    run = expected_runs(cfg)[0]
    first = write_valid_synthetic_bundle(tmp_path / "one", run)
    second = write_valid_synthetic_bundle(tmp_path / "two", run)
    final = tmp_path / "runs" / run.run_id
    outcomes: list[str] = []

    def publish(path: Path) -> None:
        try:
            atomic_promote_no_overwrite(path, final)
            outcomes.append("success")
        except FileExistsError:
            outcomes.append("blocked")

    threads = [threading.Thread(target=publish, args=(path,)) for path in (first, second)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(outcomes) == ["blocked", "success"]


def test_invalid_canonical_is_preserved_before_retry_publication(tmp_path):
    cfg, _ = one_run_config(tmp_path)
    run = expected_runs(cfg)[0]
    final = tmp_path / "runs" / run.run_id
    final.mkdir(parents=True)
    (final / "damage.txt").write_text("preserve me")
    replacement = write_valid_synthetic_bundle(tmp_path / "replacement", run)
    quarantine = tmp_path / "quarantine"
    atomic_promote_no_overwrite(replacement, final, quarantine_root=quarantine)
    assert (final / "COMPLETE.json").is_file()
    assert any((path / "damage.txt").is_file() for path in quarantine.iterdir())


def test_dry_run_worker_is_truthful_and_reconciled(tmp_path):
    cfg, path = one_run_config(tmp_path)
    root = Path(cfg["campaign"]["output_root"])
    env = dict(os.environ)
    env["PYTHONPATH"] = "scripts:src"
    init = subprocess.run(
        [sys.executable, "scripts/stage4f1c_campaign.py", "init", "--config", str(path)],
        text=True, capture_output=True, env=env,
    )
    assert init.returncode == 0, init.stderr
    worker = subprocess.run(
        [sys.executable, "scripts/stage4f1c_worker.py", "--config", str(path), "--lane", "core", "--worker-id", "test", "--dry-run", "--max-runs", "1"],
        text=True, capture_output=True, env=env,
    )
    assert worker.returncode == 0, worker.stderr
    validation = subprocess.run(
        [sys.executable, "scripts/stage4f1c_campaign.py", "validate", "--campaign", str(root), "--require-complete"],
        text=True, capture_output=True, env=env,
    )
    assert validation.returncode == 0, validation.stdout + validation.stderr
    summary = json.loads((root / "stage4f1c_validation_summary.json").read_text())
    assert summary["validation"] == "PASS"


def test_no_write_validation_preserves_campaign_bytes_and_mtimes(tmp_path):
    cfg, path = one_run_config(tmp_path)
    root = Path(cfg["campaign"]["output_root"])
    env = dict(os.environ)
    env["PYTHONPATH"] = "scripts:src"
    for command in (
        [sys.executable, "scripts/stage4f1c_campaign.py", "init", "--config", str(path)],
        [sys.executable, "scripts/stage4f1c_worker.py", "--config", str(path), "--lane", "core", "--worker-id", "test", "--dry-run", "--max-runs", "1"],
    ):
        result = subprocess.run(command, text=True, capture_output=True, env=env)
        assert result.returncode == 0, result.stdout + result.stderr

    def snapshot() -> dict[str, tuple[str, int]]:
        return {
            str(item.relative_to(root)): (hashlib.sha256(item.read_bytes()).hexdigest(), item.stat().st_mtime_ns)
            for item in root.rglob("*") if item.is_file()
        }

    before = snapshot()
    validation = subprocess.run(
        [sys.executable, "scripts/stage4f1c_campaign.py", "validate", "--campaign", str(root), "--require-complete", "--no-write"],
        text=True, capture_output=True, env=env,
    )
    assert validation.returncode == 0, validation.stdout + validation.stderr
    assert snapshot() == before
    assert not (root / "stage4f1c_validation_summary.json").exists()


def test_corrupt_checkpoint_is_rejected_even_when_manifest_matches(tmp_path):
    cfg, _ = one_run_config(tmp_path)
    run = expected_runs(cfg)[0]
    bundle = write_valid_synthetic_bundle(tmp_path, run)
    (bundle / "checkpoint.npz").write_bytes(b"internally rehashed garbage")
    checkpoint_meta = json.loads((bundle / "checkpoint_manifest.json").read_text())
    checkpoint_meta["sha256"] = sha256_file(bundle / "checkpoint.npz")
    (bundle / "checkpoint_manifest.json").write_text(json.dumps(checkpoint_meta) + "\n")
    ok, errors = validate_run_bundle_semantic(bundle, expected_spec=run.to_dict())
    assert not ok
    assert any("checkpoint restore failed" in error for error in errors)
