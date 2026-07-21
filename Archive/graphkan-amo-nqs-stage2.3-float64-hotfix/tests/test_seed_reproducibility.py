from graphkan_amo_nqs.seed import derive_seed

def test_seed_reproducibility():
    assert derive_seed(1, "a") == derive_seed(1, "a")
    assert derive_seed(1, "a") != derive_seed(2, "a")
