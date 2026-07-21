from __future__ import annotations

from pathlib import Path

from stage4f_tx import (
    RunSpec,
    Stage4FRegistry,
    atomic_promote,
    init_campaign,
    load_campaign_config,
    run_fault_injection,
    sha256_file,
    validate_run_bundle,
    write_mock_run_bundle,
)


def test_stage4f_campaign_registers_960_expected_runs(tmp_path):
    import yaml

    cfg_path = Path("configs/stage4f_campaign.yaml")
    cfg = load_campaign_config(cfg_path)
    cfg["campaign"]["output_root"] = str(tmp_path / "campaign")

    local_cfg = tmp_path / "campaign.yaml"
    local_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False))

    result = init_campaign(local_cfg)
    assert result["manifest"]["expected_runs"] == 960
    assert result["registry_summary"]["total_expected"] == 960
    assert result["registry_summary"]["duplicate_registry_keys"] == 0


def test_duplicate_worker_claim_is_blocked(tmp_path):
    db = tmp_path / "registry.sqlite"
    reg = Stage4FRegistry(db)
    reg.init()

    run = RunSpec(
        campaign_id="test_campaign",
        boundary="open",
        delta="1.50",
        c6="4",
        model="endpoint_product_b32",
        seed=0,
        lane="core",
    )
    reg.register_expected([run])

    first = reg.claim_next("core", "worker-A")
    second = reg.claim_next("core", "worker-B")

    assert first is not None
    assert second is None


def test_incomplete_bundle_is_not_validated(tmp_path):
    cfg = load_campaign_config(Path("configs/stage4f_campaign.yaml"))
    run = RunSpec(
        campaign_id="test_campaign",
        boundary="open",
        delta="1.50",
        c6="4",
        model="endpoint_product_b32",
        seed=0,
        lane="core",
    )

    bundle = write_mock_run_bundle(tmp_path, run, cfg["completion_requirements"], fault="missing_history")
    ok, errors = validate_run_bundle(bundle, cfg["completion_requirements"])

    assert not ok
    assert any("history" in e for e in errors)


def test_complete_bundle_validates_and_can_be_marked_complete(tmp_path):
    cfg = load_campaign_config(Path("configs/stage4f_campaign.yaml"))
    db = tmp_path / "registry.sqlite"
    reg = Stage4FRegistry(db)
    reg.init()

    run = RunSpec(
        campaign_id="test_campaign",
        boundary="open",
        delta="1.50",
        c6="4",
        model="endpoint_product_b32",
        seed=0,
        lane="core",
    )
    reg.register_expected([run])
    row = reg.claim_next("core", "worker-A")
    assert row is not None

    tmp_bundle = write_mock_run_bundle(tmp_path, run, cfg["completion_requirements"], fault=None)
    final_bundle = tmp_path / "runs" / run.run_id
    atomic_promote(tmp_bundle, final_bundle)

    ok, errors = validate_run_bundle(final_bundle, cfg["completion_requirements"])
    assert ok, errors

    reg.mark_complete(
        run.run_id,
        "worker-A",
        final_bundle,
        sha256_file(final_bundle / "COMPLETE.json"),
        sha256_file(final_bundle / "resolved_config.yaml"),
    )

    summary = reg.summary()
    assert summary["status_counts"]["COMPLETE"] == 1
    assert summary["duplicate_registry_keys"] == 0


def test_fault_injection_gate_passes(tmp_path):
    import yaml

    cfg = load_campaign_config(Path("configs/stage4f_campaign.yaml"))
    cfg["campaign"]["output_root"] = str(tmp_path / "campaign")
    cfg_path = tmp_path / "campaign.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    fault_cfg = yaml.safe_load(Path("configs/stage4f_fault_injection.yaml").read_text())
    fault_cfg["fault_injection"]["output_root"] = str(tmp_path / "faults")
    fault_path = tmp_path / "fault.yaml"
    fault_path.write_text(yaml.safe_dump(fault_cfg, sort_keys=False))

    summary = run_fault_injection(cfg_path, fault_path)
    assert summary["fault_injection"] == "PASS"
    assert summary["duplicate_worker_claim_blocked"] is True
    assert summary["registry_summary"]["status_counts"]["COMPLETE"] == 1
