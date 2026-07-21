from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]
def test_stage4d_configs_are_budgeted_and_no_graphkan():
    for name, limit in [('netket_stage4d_mechanism_screen.yaml',100),('netket_stage4d_low_detuning_autopsy.yaml',60),('netket_stage4d_n10_pilot.yaml',60)]:
        c=yaml.safe_load((ROOT/'configs'/name).read_text())
        n=len(c['chains']['n_sites'])*len(c['parameters']['delta'])*len(c['parameters']['c6'])*len(c['training']['seeds'])*len(c['model_specs'])
        assert n<=limit
        assert all('graph' not in k.lower() for k in c['model_specs'])
def test_hybrid_is_implemented():
    s=(ROOT/'src/graphkan_amo_nqs/models/flax_nqs.py').read_text()
    assert 'kan_endpoint_hybrid_jastrow' in s
    assert 'kan_endpoint_hybrid' in s
