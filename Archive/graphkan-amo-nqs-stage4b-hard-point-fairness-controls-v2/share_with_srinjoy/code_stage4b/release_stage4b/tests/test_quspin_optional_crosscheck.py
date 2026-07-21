from graphkan_amo_nqs.hamiltonians.quspin_builders import quspin_crosscheck_available

def test_quspin_optional_bool():
    assert isinstance(quspin_crosscheck_available(), bool)
