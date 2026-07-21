from pathlib import Path
import yaml

def test_stage3f_required_assets_exist():
    root=Path(__file__).resolve().parents[1]
    for rel in [
        'scripts/run_stage3f.py','scripts/summarize_stage3f.py','scripts/run_stage3f_clean_replay.sh',
        'configs/netket_stage3f_component_ablation_n6.yaml','configs/netket_stage3f_n8_batch_a.yaml',
        'configs/netket_stage3f_mlp_control_sweep_n8.yaml','environment-nqs-py312-macos-arm64-stage3f.yml',
        'requirements/stage3f-py312-macos-arm64.lock.txt',
        'docs/tracking/stage_history/STAGE_3F_README.md']:
        assert (root/rel).exists(), rel

def test_n8_joint_gate_config_has_correlation_guard():
    root=Path(__file__).resolve().parents[1]
    cfg=yaml.safe_load((root/'configs/netket_stage3f_n8_batch_a.yaml').read_text())
    gate=cfg['joint_gate']
    assert gate['max_endpoint_corr_absolute_penalty'] == 0.01
    assert gate['max_endpoint_corr_relative_penalty'] == 0.25
