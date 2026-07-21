from pathlib import Path


def test_stage3b_assets_present():
    root = Path(__file__).resolve().parents[1]
    assert (root / "configs" / "netket_stage3b_endpoint_kan_gate.yaml").exists()
    assert (root / "scripts" / "run_stage3b_endpoint_kan.py").exists()
    assert (root / "scripts" / "summarize_stage3b.py").exists()
    source = (root / "src" / "graphkan_amo_nqs" / "models" / "flax_nqs.py").read_text()
    assert "kan_endpoint_jastrow" in source
    assert "endpoint_coeff" in source
