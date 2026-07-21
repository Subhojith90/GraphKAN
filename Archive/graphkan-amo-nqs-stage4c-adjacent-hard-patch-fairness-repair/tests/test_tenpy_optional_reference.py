from graphkan_amo_nqs.hamiltonians.tenpy_references import tenpy_dmrg_available

def test_tenpy_optional_bool():
    assert isinstance(tenpy_dmrg_available(), bool)
