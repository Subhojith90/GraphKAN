import numpy as np
import pytest
jax = pytest.importorskip("jax")
pytest.importorskip("flax")
from graphkan_amo_nqs.models.flax_nqs import build_flax_model

def count_params(tree):
    return int(sum(np.asarray(x).size for x in jax.tree_util.tree_leaves(tree["params"])))

@pytest.mark.parametrize("kind,basis,n_basis", [
    ("kan_jastrow", "sine", 13),
    ("kan_endpoint_jastrow", "sine", 10),
    ("kan_jastrow", "rbf", 13),
    ("kan_endpoint_jastrow", "rbf", 10),
    ("mlp_pair", "rbf", 10),
    ("mlp_endpoint_pair", "rbf", 7),
])
def test_stage3e_models_are_19_parameter_matched(kind, basis, n_basis):
    positions = np.arange(6, dtype=float)[:, None]
    model = build_flax_model(kind, positions, n_basis=n_basis, hidden=1, omega=1.0, inverted_ordering=False, basis_family=basis)
    params = model.init(jax.random.PRNGKey(0), jax.numpy.ones((1, 6), dtype=jax.numpy.float64))
    assert count_params(params) == 19
