from pathlib import Path
import json

def test_release_has_current_manifest_and_historical_stage4d_manifest():
    root = Path(__file__).resolve().parents[1]
    assert (root / "release_manifest.json").exists()
    manifest = json.loads((root / "release_manifest.json").read_text())
    assert manifest["authoritative_stage"] == "Stage 4F-R.1b"
    historical = json.loads(
        (root / "release_integrity/historical/stage4d_root_manifest.json").read_text()
    )
    assert historical["status"] == "HISTORICAL_SUPERSEDED"

def test_release_has_stage4c_artifact_minimums():
    root = Path(__file__).resolve().parents[1]
    for rel in ["README.md", "REPRODUCIBILITY.md", "environment-nqs-py312-macos-arm64-stage4c.yml", "configs/netket_stage4c_hard_patch.yaml", "scripts/run_stage4c_patch.py", "scripts/summarize_stage4c.py", "scripts/run_stage4c_batch.sh", "src/graphkan_amo_nqs/models/flax_nqs.py"]:
        assert (root / rel).exists(), rel
