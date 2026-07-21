from __future__ import annotations

import shutil
import time
from pathlib import Path

import yaml

from stage4f1b_tx import (
    Stage4F1BRegistry,
    expected_runs,
    load_config,
    protocol_hash_from_config,
    sha256_file,
    validate_run_bundle_semantic,
    write_valid_synthetic_bundle,
)


def local_cfg(tmp_path):
    cfg = load_config(Path("configs/stage4f1b_campaign.yaml"))
    cfg["campaign"]["output_root"] = str(tmp_path / "campaign")
    p = tmp_path / "campaign.yaml"
    p.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return cfg, p


def test_semantic_validator_rejects_adversarial_corruptions(tmp_path):
    cfg, _ = local_cfg(tmp_path)
    run = expected_runs(cfg)[0]
    valid = write_valid_synthetic_bundle(tmp_path, run)
    ok, errors = validate_run_bundle_semantic(valid, run.run_id)
    assert ok, errors

    cases = []

    def copy_case(name):
        dst = tmp_path / name
        shutil.copytree(valid, dst)
        return dst

    p = copy_case("tamper_metrics")
    (p / "metrics.csv").write_text(f"run_id,status,energy_error\n{run.run_id},VALID,123.0\n")
    cases.append(p)

    p = copy_case("bad_hash")
    (p / "artifact_manifest.json").write_text('{"files":{"metrics.csv":"0000"}}\n')
    cases.append(p)

    p = copy_case("missing_checkpoint_payload")
    (p / "checkpoint.npz").unlink()
    cases.append(p)

    p = copy_case("bad_checkpoint_manifest")
    (p / "checkpoint_manifest.json").write_text("{bad json")
    cases.append(p)

    p = copy_case("unreadable_checkpoint")
    (p / "checkpoint_manifest.json").write_text('{"run_id":"' + run.run_id + '","checkpoint":"checkpoint.npz","readable":false}\n')
    cases.append(p)

    p = copy_case("wrong_run_id")
    (p / "metrics.csv").write_text("run_id,status,energy_error\nWRONG,VALID,0.0\n")
    cases.append(p)

    p = copy_case("wrong_complete")
    (p / "COMPLETE.json").write_text('{"run_id":"WRONG","artifact_validation":"PASS"}\n')
    cases.append(p)

    p = copy_case("complete_not_last")
    time.sleep(0.02)
    (p / "metrics.csv").write_text(f"run_id,status,energy_error\n{run.run_id},VALID,0.0\n")
    cases.append(p)

    for case in cases:
        ok, errors = validate_run_bundle_semantic(case, run.run_id)
        assert not ok, f"{case.name} incorrectly accepted"


def test_protocol_mutation_is_rejected(tmp_path):
    cfg, cfg_path = local_cfg(tmp_path)
    root = Path(cfg["campaign"]["output_root"])
    reg = Stage4F1BRegistry(root / "registry.sqlite")
    reg.register_campaign(cfg_path, cfg)

    mutated = yaml.safe_load(yaml.safe_dump(cfg))
    mutated["physics"]["deltas"] = ["1.25", "1.50", "1.75", "2.00", "2.25"]
    mut_path = tmp_path / "mutated.yaml"
    mut_path.write_text(yaml.safe_dump(mutated, sort_keys=False))

    try:
        reg.register_campaign(mut_path, mutated)
    except RuntimeError as exc:
        assert "Protocol mutation" in str(exc)
    else:
        raise AssertionError("protocol mutation was accepted")


def test_stale_lease_is_reclaimed_and_pending_progresses(tmp_path):
    cfg, cfg_path = local_cfg(tmp_path)
    root = Path(cfg["campaign"]["output_root"])
    reg = Stage4F1BRegistry(root / "registry.sqlite")
    reg.register_campaign(cfg_path, cfg)

    first = reg.claim_next("core", "worker-A", lease_seconds=1)
    assert first is not None
    time.sleep(1.2)

    second = reg.claim_next("core", "worker-B", lease_seconds=1)
    assert second is not None

    third = reg.claim_next("core", "worker-C", lease_seconds=1)
    assert third is not None
    assert third["run_id"] != second["run_id"]


def test_atomic_promotion_refuses_overwrite(tmp_path):
    from stage4f1b_tx import atomic_promote_no_overwrite

    cfg, _ = local_cfg(tmp_path)
    run = expected_runs(cfg)[0]
    first = write_valid_synthetic_bundle(tmp_path, run)
    final = tmp_path / "final"
    atomic_promote_no_overwrite(first, final)

    second = write_valid_synthetic_bundle(tmp_path, run)
    try:
        atomic_promote_no_overwrite(second, final)
    except FileExistsError:
        pass
    else:
        raise AssertionError("overwrite was allowed")
