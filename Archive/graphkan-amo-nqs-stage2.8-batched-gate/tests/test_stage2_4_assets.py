from pathlib import Path
import yaml

def test_stage2_4_profile_is_small_ed_calibrated_and_separate():
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root/'configs'/'netket_stage2_4_ed_calibrated.yaml').read_text())
    assert cfg['output_root'] == 'results_stage2_4'
    assert cfg['chains']['n_sites'] == [6, 8]
    assert cfg['training']['learning_rate'] < 0.01
    assert cfg['health_gate']['max_abs_energy_error_to_ed'] > 0
    assert (root/'scripts'/'run_netket_vmc_stabilized.py').exists()
