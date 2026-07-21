from graphkan_amo_nqs.models.flax_nqs import resolve_basis_family

def test_stage3c_explicit_families():
    assert resolve_basis_family("kan_endpoint_jastrow", "sine") == "sine"
    assert resolve_basis_family("kan_endpoint_sum_jastrow", "sine") == "sine"
    assert resolve_basis_family("kan_endpoint_diff_jastrow", "sine") == "sine"
    assert resolve_basis_family("kan_endpoint_product_jastrow", "sine") == "sine"
    assert resolve_basis_family("mlp_endpoint_pair", "rbf") == "rbf"
