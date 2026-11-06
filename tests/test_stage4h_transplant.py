from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from graphkan_amo_nqs.training.stage4h_transplant import (
    find_radial_checkpoint,
    full_basis_attestation,
    load_stage4g_radial_checkpoint,
    transplant_parameters,
)


def make_radial_checkpoint(path: Path) -> None:
    np.savez_compressed(
        path, leaf_0000=np.arange(32, dtype=float), leaf_0001=np.arange(12, dtype=float)
    )
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "model_label": "rbf_radial_b32",
                "seed": 2,
                "run_id": "stage4g_confirmation_open:ryd_chain_N12_D1.25_C63_deadbeef:rbf_radial_b32:samples4096:shift0.1:seed2",
            }
        )
    )


def test_radial_checkpoint_schema_and_semantic_locator(tmp_path: Path):
    checkpoint = tmp_path / "source.npz"
    make_radial_checkpoint(checkpoint)
    loaded = load_stage4g_radial_checkpoint(checkpoint)
    assert set(loaded) == {"kan_coeff", "site_bias"}
    assert loaded["kan_coeff"].shape == (32,)
    assert find_radial_checkpoint(tmp_path, delta=1.25, c6=3.0, seed=2) == checkpoint


@pytest.mark.parametrize(
    "arm,extra,extra_size",
    [
        ("radial_continuation", None, 0),
        ("endpoint_full_warmstart", "endpoint_coeff", 3),
        ("endpoint_hybrid_warmstart", "endpoint_coeff", 2),
        ("nongeometric_warmstart", "control_coeff", 3),
    ],
)
def test_transplant_copies_radial_and_zeroes_every_extra(
    arm: str, extra: str | None, extra_size: int
):
    radial = {
        "kan_coeff": np.arange(32, dtype=float),
        "site_bias": np.arange(12, dtype=float),
    }
    template = {"kan_coeff": np.ones(32), "site_bias": np.ones(12)}
    if extra:
        template[extra] = np.full(extra_size, 99.0)
    transplanted = transplant_parameters(template, radial, arm)
    assert np.array_equal(np.asarray(transplanted["kan_coeff"]), radial["kan_coeff"])
    assert np.array_equal(np.asarray(transplanted["site_bias"]), radial["site_bias"])
    if extra:
        assert np.array_equal(np.asarray(transplanted[extra]), np.zeros(extra_size))


def test_full_basis_endpoint_transplant_is_exact():
    jax = pytest.importorskip("jax")
    pytest.importorskip("flax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from graphkan_amo_nqs.models.stage4g_flax_nqs import build_flax_model

    positions = np.arange(12, dtype=float)[:, None]
    radial_model = build_flax_model(
        "kan_jastrow", positions, n_basis=32, hidden=1, basis_family="rbf"
    )
    endpoint_model = build_flax_model(
        "kan_endpoint_jastrow", positions, n_basis=32, hidden=1, basis_family="rbf"
    )
    x = jnp.ones((1, 12), dtype=jnp.float64)
    radial = radial_model.init(jax.random.PRNGKey(7), x)["params"]
    target_template = endpoint_model.init(jax.random.PRNGKey(9), x)["params"]
    target = transplant_parameters(
        target_template,
        {k: np.asarray(v) for k, v in radial.items()},
        "endpoint_full_warmstart",
    )
    binary = (
        (np.arange(4096, dtype=np.uint64)[:, None] >> np.arange(12, dtype=np.uint64))
        & 1
    ).astype(float)
    sigma = 1.0 - 2.0 * binary
    result = full_basis_attestation(radial_model, radial, endpoint_model, target, sigma)
    assert result["validation"] == "PASS"
    assert result["states_compared"] == 4096
    assert result["maximum_log_amplitude_difference"] == 0.0
