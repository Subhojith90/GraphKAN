def test_imports():
    import graphkan_amo_nqs
    from graphkan_amo_nqs.geometry import make_chain_geometry
    from graphkan_amo_nqs.hamiltonians import RydbergParams
    from graphkan_amo_nqs.models import make_model
    assert graphkan_amo_nqs.__version__
    assert make_chain_geometry(4).n_sites == 4
    assert RydbergParams().omega == 1.0
    assert make_model("kan_jastrow")
