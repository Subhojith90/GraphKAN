from pathlib import Path
import yaml


def test_exact_reference_benchmark_is_registered_and_fair():
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "configs" / "exact_reference_benchmark.yaml").read_text())
    assert cfg["profile"] == "exact_reference_benchmark_v1"
    assert cfg["models"] == ["product", "smooth_jastrow", "mlp_pair", "kan_jastrow"]
    assert cfg["training"]["seeds"] == [0, 1, 2, 3, 4]
    assert cfg["training"]["maxiter"] >= 60
    assert sorted(cfg["chains"]["n_sites"]) == [8, 10]
