import pytest

from graphkan_amo_nqs.models.flax_nqs import resolve_basis_family


def test_model_kinds_have_explicit_basis_defaults():
    assert resolve_basis_family("kan_jastrow") == "sine"
    assert resolve_basis_family("kan_boundary_jastrow") == "sine"
    assert resolve_basis_family("kan_endpoint_jastrow") == "sine"
    assert resolve_basis_family("smooth_jastrow") == "rbf"
    assert resolve_basis_family("mlp_pair") == "rbf"


def test_explicit_basis_override_is_recordable_and_validated():
    assert resolve_basis_family("kan_endpoint_jastrow", "sine") == "sine"
    assert resolve_basis_family("smooth_jastrow", "rbf") == "rbf"
    with pytest.raises(ValueError):
        resolve_basis_family("kan_endpoint_jastrow", "unknown")
