from pathlib import Path
import yaml

def test_stage27_configs_and_scripts_exist():
    for p in ["configs/netket_stage2_7_hard_regime.yaml", "configs/netket_stage2_7_matched_fairness.yaml", "scripts/run_stage2_7_hard_regime.py", "scripts/run_stage2_7_fairness.py", "scripts/summarize_stage2_7.py", "environment-nqs-py312-macos-arm64-minimal.yml"]:
        assert Path(p).exists()
    cfg=yaml.safe_load(Path("configs/netket_stage2_7_hard_regime.yaml").read_text())
    assert cfg["parameters"]["delta"] == [2.0] and cfg["parameters"]["c6"] == [4.0]
    assert set(cfg["model_specs"]) == {"rbf_matched","mlp_matched","kan_matched"}
