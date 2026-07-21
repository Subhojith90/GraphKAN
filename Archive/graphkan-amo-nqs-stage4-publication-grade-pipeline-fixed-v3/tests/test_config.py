from graphkan_amo_nqs.config import load_config

def test_load_config():
    cfg = load_config("configs/quick_ci.yaml")
    assert cfg["profile"] == "quick_ci"
