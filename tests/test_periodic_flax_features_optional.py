import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("flax")

from graphkan_amo_nqs.geometry import make_periodic_chain_geometry
from graphkan_amo_nqs.geometry.distances import geometry_distance_matrix
from graphkan_amo_nqs.models.flax_nqs import build_flax_model


def _count(params):
    return sum(np.asarray(x).size for x in jax.tree_util.tree_leaves(params))


def test_periodic_endpoint_model_reduces_to_radial_parameterization():
    import jax.numpy as jnp
    geometry = make_periodic_chain_geometry(6)
    d = geometry_distance_matrix(geometry)
    endpoint_disabled = build_flax_model(
        "kan_endpoint_sum_jastrow", geometry.positions, n_basis=12, hidden=1,
        basis_family="rbf", distance_matrix=d, endpoint_features_enabled=False,
    )
    radial = build_flax_model(
        "kan_jastrow", geometry.positions, n_basis=12, hidden=1,
        basis_family="rbf", distance_matrix=d, endpoint_features_enabled=False,
    )
    x = jnp.ones((1, 6), dtype=jnp.float64)
    p1 = endpoint_disabled.init(jax.random.PRNGKey(0), x)["params"]
    p2 = radial.init(jax.random.PRNGKey(1), x)["params"]
    assert _count(p1) == _count(p2)
    assert "endpoint_coeff" not in p1
