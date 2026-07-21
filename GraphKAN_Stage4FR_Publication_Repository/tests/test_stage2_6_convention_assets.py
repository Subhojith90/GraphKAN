from pathlib import Path
import yaml

def test_stage26_configs_lock_ordering_and_observables():
    root=Path(__file__).resolve().parents[1]
    cfg=yaml.safe_load((root/'configs'/'netket_stage2_6_observable_grid.yaml').read_text())
    assert cfg['netket']['inverted_ordering'] is False
    assert cfg['health_gate']['max_density_mae_to_ed'] > 0
    assert cfg['health_gate']['max_connected_correlation_mae_to_ed'] > 0

def test_stage26_assets_exist():
    root=Path(__file__).resolve().parents[1]
    for rel in ['scripts/check_netket_conventions.py','scripts/run_netket_vmc_stage2_6.py','scripts/summarize_stage2_6.py','requirements/nqs-executed-macos-arm64.txt','environment-nqs-py312-macos-arm64.yml']:
        assert (root/rel).exists()
