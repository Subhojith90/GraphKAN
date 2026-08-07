from __future__ import annotations

from pathlib import Path

from stage4h_protocol import (
    ARMS,
    freeze,
    read_config,
    roster,
    validate_config,
    verify_freeze,
)


def test_stage4h_protocol_is_exactly_sixty_runs_and_twelve_shards(tmp_path: Path):
    cfg = read_config(Path("configs/stage4h_warmstart.yaml"))
    assert validate_config(cfg)["validation"] == "PASS"
    rows = roster(cfg)
    assert len(rows) == 60
    assert len({row["shard"] for row in rows}) == 12
    assert {row["arm"] for row in rows} == set(ARMS)
    assert {row["seed"] for row in rows} == set(range(5))
    result = freeze(cfg, tmp_path)
    assert result["expected_runs"] == 60
    assert result["expected_shards"] == 12
    assert verify_freeze(cfg, tmp_path)["validation"] == "PASS"


def test_stage4h_local_launcher_is_resumable_and_preserves_failed_evidence():
    text = Path("scripts/run_stage4h_local.py").read_text()
    assert 'choices=("prepare", "run", "status", "finalize", "all")' in text
    assert "--max-workers" in text
    assert "--resume" in text
    assert "failure.json" in text
    assert "SIGTERM" in text
    assert "merge_stage4h.py" in text
    assert "analyse_stage4h.py" in text


def test_stage4h_packager_crc_checks_and_verifies_every_payload():
    text = Path("scripts/package_stage4h.py").read_text()
    assert "archive.testzip()" in text
    assert "payload_hashes" in text
    assert "archived payload hash mismatch" in text
    assert '.with_suffix(output.suffix + ".sha256")' in text


def test_stage4h_frozen_decision_thresholds_are_supervisory_values():
    cfg = read_config(Path("configs/stage4h_warmstart.yaml"))
    gate = cfg["acceptance"]
    assert gate["transplant_max_log_amplitude_difference"] == 1e-12
    assert gate["energy_safety_margin"] == 1e-4
    assert gate["energy_safety_minimum_pairs"] == 14
    assert gate["paired_seed_sign_minimum"] == 4
    assert gate["scientific_rescue_minimum_cells"] == 2
    assert gate["periodic_alias_required_passes"] == 30
