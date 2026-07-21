from pathlib import Path


def test_stage4b_sampler_uses_hilbert_not_operator():
    text = Path('scripts/run_stage4b_controls.py').read_text()
    assert 'hi, hamiltonian = build_netket_rydberg_operator' in text
    assert 'nk.sampler.MetropolisLocal(hi,' in text
    assert 'MetropolisLocal(hamiltonian,' not in text


def test_stage4b_smooth_endpoint_is_explicitly_supported():
    text = Path('src/graphkan_amo_nqs/models/flax_nqs.py').read_text()
    assert '"smooth_endpoint_sum_jastrow": "rbf"' in text
