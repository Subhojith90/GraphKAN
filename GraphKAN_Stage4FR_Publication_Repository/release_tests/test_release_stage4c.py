from pathlib import Path
import json

def test_release_has_manifest_and_native_schema():
    root = Path(__file__).resolve().parents[1]
    assert (root / "release_manifest.json").exists()
    assert (root / "results_stage4c").exists()
    manifest = json.loads((root / "release_manifest.json").read_text())
    assert "results_stage4c/evidence_bundle/batch_index.csv" in manifest

def test_release_has_stage4c_artifact_minimums():
    root = Path(__file__).resolve().parents[1]
    for rel in ["README.md", "REPRODUCIBILITY.md", "environment-nqs-py312-macos-arm64-stage4c.yml", "configs/netket_stage4c_hard_patch.yaml", "scripts/run_stage4c_patch.py", "scripts/summarize_stage4c.py", "scripts/run_stage4c_batch.sh", "src/graphkan_amo_nqs/models/flax_nqs.py"]:
        assert (root / rel).exists(), rel
