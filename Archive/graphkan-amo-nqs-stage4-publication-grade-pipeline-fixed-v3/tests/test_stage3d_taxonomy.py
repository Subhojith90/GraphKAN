from graphkan_amo_nqs.models.flax_nqs import resolve_basis_family

def test_stage3d_explicit_taxonomy():
    assert resolve_basis_family('kan_jastrow', 'sine') == 'sine'
    assert resolve_basis_family('kan_endpoint_jastrow', 'sine') == 'sine'
    assert resolve_basis_family('kan_jastrow', 'rbf') == 'rbf'
    assert resolve_basis_family('kan_endpoint_jastrow', 'rbf') == 'rbf'
    assert resolve_basis_family('smooth_jastrow', 'rbf') == 'rbf'
    assert resolve_basis_family('mlp_endpoint_pair', 'rbf') == 'rbf'
