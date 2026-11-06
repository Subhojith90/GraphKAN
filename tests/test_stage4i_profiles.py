from pathlib import Path

import conftest as profile_config


def test_stage4i_profiles_are_explicit_and_disjoint():
    assert profile_config.NQS_MODULES
    assert profile_config.SLOW_MODULES
    assert profile_config.NQS_MODULES.isdisjoint(profile_config.SLOW_MODULES)
    assert "test_stage4f1_transactional_scheduler.py" in profile_config.SLOW_MODULES


def test_stage4i_dependency_locks_exist():
    root = Path(__file__).resolve().parents[1]
    base = (root / "requirements/stage4i-base-py312.txt").read_text()
    full = (root / "requirements/stage4i-full-nqs-py312.txt").read_text()
    assert "numpy==" in base and "pytest==" in base
    assert "jax==" in full and "netket==" in full
