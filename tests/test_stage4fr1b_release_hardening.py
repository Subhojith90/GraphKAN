from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from stage4f1b_tx import expected_runs, load_config


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_campaign_ledger_is_exact_authoritative_serialisation():
    campaign = json.loads(Path("results_stage4f_r/frozen/campaign_manifest.json").read_text())
    release = json.loads(Path("release_integrity/frozen_source_ledger.json").read_text())
    assert release["authority"].endswith("#protocol_identity.source_ledger")
    assert release["source_ledger"] == campaign["protocol_identity"]["source_ledger"]
    assert "reproducibility/environment_stage4f_portable.yml" in release["source_ledger"]
    assert "scripts/finalize_stage4fr_publication.py" not in release["source_ledger"]


def test_separate_release_tool_ledger_closes():
    ledger = json.loads(Path("release_integrity/release_tool_ledger.json").read_text())
    assert "scripts/finalize_stage4fr_publication.py" in ledger["source_ledger"]
    assert all(digest(Path(path)) == expected for path, expected in ledger["source_ledger"].items())


def test_replay_roster_is_exactly_twenty_rows():
    config = load_config(Path("configs/stage4fr1b_second_host_replay.yaml"))
    runs = expected_runs(config)
    assert len(runs) == 20
    assert sum(run.lane == "core" for run in runs) == 16
    assert sum(run.lane == "control" for run in runs) == 4
    assert {run.model for run in runs} == {
        "rbf_radial_b32", "endpoint_full_b32", "endpoint_product_b32", "mlp_endpoint_large", "rbm_h32",
    }


def test_tolerances_are_frozen_and_cover_exact_roster():
    config = yaml.safe_load(Path("configs/stage4fr1b_second_host_replay.yaml").read_text())
    tolerances = json.loads(Path(config["completion_requirements"]["tolerance_table"]).read_text())
    assert tolerances["status"] == "FROZEN_BEFORE_REPLAY"
    assert len(tolerances["thresholds"]) == 20
    assert tolerances["periodic_collapse_absolute"] <= 1.0e-12


def test_linux_lock_is_hash_pinned_and_contains_campaign_stack():
    lock = Path("requirements/stage4fr1b-linux-py312.lock.txt").read_text().lower()
    for requirement in ("jax==0.7.1", "jaxlib==0.7.1", "flax==0.12.0", "optax==0.2.8", "netket==3.22.3"):
        assert requirement in lock
    assert lock.count("--hash=sha256:") > 100
