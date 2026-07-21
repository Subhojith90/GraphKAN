from pathlib import Path
import yaml
from graphkan_amo_nqs.models.flax_nqs import resolve_basis_family

ROOT = Path(__file__).resolve().parents[1]

def test_hybrid_resolves_to_rbf_when_explicitly_overridden():
    assert resolve_basis_family("kan_endpoint_hybrid_jastrow", "rbf") == "rbf"

def test_fullbudget_smoke_is_registered_and_small():
    cfg = yaml.safe_load((ROOT / "configs/netket_stage4d_fullbudget_smoke.yaml").read_text())
    assert cfg["training"]["n_samples"] == 4096
    assert cfg["training"]["n_chains"] == 64
    assert cfg["training"]["n_iter"] == 700
    n = len(cfg["chains"]["n_sites"])*len(cfg["parameters"]["delta"])*len(cfg["parameters"]["c6"])*len(cfg["training"]["seeds"])*len(cfg["model_specs"])
    assert n == 6

def test_mechanism_mini_requires_smoke_and_is_36_runs():
    cfg = yaml.safe_load((ROOT / "configs/netket_stage4d_mechanism_mini.yaml").read_text())
    assert cfg["requires_smoke_gate"] is True
    assert cfg["training"]["n_samples"] == 4096
    assert cfg["training"]["n_iter"] == 700
    n = len(cfg["chains"]["n_sites"])*len(cfg["parameters"]["delta"])*len(cfg["parameters"]["c6"])*len(cfg["training"]["seeds"])*len(cfg["model_specs"])
    assert n == 36
