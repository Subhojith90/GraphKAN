import numpy as np
import pytest

from graphkan_amo_nqs.training.stage4i_orthogonal import (
    model_projection,
    orthogonalisation,
)


def test_stage4i_orthogonalisation_preserves_span_and_improves_conditioning():
    result = orthogonalisation(12, 5.0 ** (1 / 6))
    assert result["validation"] == "PASS"
    assert result["raw_rank"] == result["orthogonal_rank"]
    assert result["column_space_projector_error"] <= 1e-10
    assert result["orthogonality_error"] <= 1e-12
    assert result["orthogonal_condition_number"] < result["raw_condition_number"]
    projection = model_projection(result)
    assert projection["one_body_projection"].shape == (12, 3)
    assert projection["radial_projection"].shape == (32, 3)


@pytest.mark.nqs
def test_orthogonal_model_is_exactly_radial_at_zero_endpoint_coefficient():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    jnp = pytest.importorskip("jax.numpy")
    pytest.importorskip("flax")
    from graphkan_amo_nqs.models.stage4g_flax_nqs import build_flax_model

    positions = np.arange(12, dtype=float)[:, None]
    result = orthogonalisation(12, 5.0 ** (1 / 6))
    radial = build_flax_model(
        "kan_jastrow",
        positions,
        n_basis=32,
        basis_family="rbf",
        r_scale=5.0 ** (1 / 6),
    )
    orthogonal = build_flax_model(
        "kan_endpoint_orthogonal_jastrow",
        positions,
        n_basis=32,
        basis_family="rbf",
        r_scale=5.0 ** (1 / 6),
        endpoint_orthogonalisation=model_projection(result),
    )
    sigma = jnp.asarray(
        1 - 2 * np.array([[int(x) for x in f"{k:012b}"] for k in range(4096)]),
        dtype=jnp.float64,
    )
    rp = radial.init(jax.random.PRNGKey(0), sigma[:1])["params"]
    op = orthogonal.init(jax.random.PRNGKey(1), sigma[:1])["params"]
    op = {
        "site_bias": rp["site_bias"],
        "kan_coeff": rp["kan_coeff"],
        "endpoint_coeff": jnp.zeros_like(op["endpoint_coeff"]),
    }
    difference = np.max(
        np.abs(
            np.asarray(radial.apply({"params": rp}, sigma))
            - np.asarray(orthogonal.apply({"params": op}, sigma))
        )
    )
    assert difference <= 1e-12
