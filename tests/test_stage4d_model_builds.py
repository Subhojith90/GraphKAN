from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("jax", reason="NQS model-build tests require JAX.")
pytest.importorskip("flax", reason="NQS model-build tests require Flax.")
pytest.importorskip("netket", reason="NQS model-build tests require NetKet.")

from graphkan_amo_nqs.models.flax_nqs import build_flax_model, resolve_basis_family


@pytest.mark.parametrize(
    ("kind", "n_basis", "hidden"),
    [
        ("kan_jastrow", 13, 1),
        ("kan_endpoint_sum_jastrow", 12, 1),
        ("kan_endpoint_product_jastrow", 12, 1),
        ("kan_endpoint_hybrid_jastrow", 12, 1),
        ("mlp_pair", 15, 7),
        ("mlp_endpoint_pair", 12, 7),
        ("rbm", 1, 8),
    ],
)
def test_stage4d_n10_model_kind_builds(kind: str, n_basis: int, hidden: int) -> None:
    positions = np.arange(10, dtype=np.float64)[:, None]
    family = resolve_basis_family(kind, "rbf")
    if kind == "rbm":
        assert family == "none"
    else:
        assert family == "rbf"

    model = build_flax_model(
        kind,
        positions,
        n_basis=n_basis,
        hidden=hidden,
        r_scale=1.0,
        omega=1.0,
        inverted_ordering=False,
        basis_family=family,
    )
    params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 10), dtype=jnp.float64))
    output = model.apply(params, jnp.ones((1, 10), dtype=jnp.float64))

    assert "params" in params
    assert output.shape == (1,)


def test_stage4g_nongeometric_control_matches_endpoint_full_parameter_count() -> None:
    from graphkan_amo_nqs.models.stage4g_flax_nqs import build_flax_model as build_stage4g_model

    positions = np.arange(12, dtype=np.float64)[:, None]
    x = jnp.ones((1, 12), dtype=jnp.float64)
    counts = []
    for kind in ("kan_endpoint_jastrow", "kan_nongeometric_control_jastrow"):
        model = build_stage4g_model(kind, positions, n_basis=32, hidden=1, basis_family="rbf")
        params = model.init(jax.random.PRNGKey(0), x)["params"]
        counts.append(sum(np.asarray(leaf).size for leaf in jax.tree_util.tree_leaves(params)))
    assert counts == [47, 47]
