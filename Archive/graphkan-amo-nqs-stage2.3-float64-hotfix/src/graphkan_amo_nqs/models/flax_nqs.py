"""Trainable Flax log-amplitude models for fair NetKet VMC comparisons.

All pair models consume the same occupation-spin configuration and geometry-derived pair
features. They are deliberately small and explicit so parameter budgets can be reported.
"""
from __future__ import annotations

import numpy as np


def require_flax():
    try:
        import jax.numpy as jnp
        import flax.linen as nn
        return jnp, nn
    except Exception as exc:
        from graphkan_amo_nqs.exceptions import DependencyUnavailable
        raise DependencyUnavailable("Flax/JAX are required for trainable NQS models. Install `.[nqs]`.") from exc


def pair_feature_arrays(positions: np.ndarray, n_basis: int = 8, family: str = "sine", r_scale: float = 1.0):
    positions = np.asarray(positions, dtype=float)
    i, j = np.triu_indices(len(positions), k=1)
    r = np.linalg.norm(positions[i] - positions[j], axis=1) / max(float(r_scale), 1e-12)
    if family == "sine":
        B = np.stack([np.sin((k + 1) * np.pi * r / max(r.max(), 1e-12)) for k in range(n_basis)], axis=1)
    elif family == "rbf":
        centers = np.linspace(float(r.min()), float(r.max()), n_basis)
        width = max(float(np.ptp(centers)) / max(n_basis - 1, 1), 1e-6)
        B = np.exp(-0.5 * ((r[:, None] - centers[None, :]) / width) ** 2)
    else:
        raise ValueError(f"Unknown pair feature family: {family}")
    return i.astype(np.int32), j.astype(np.int32), B.astype(np.float64)


def build_flax_model(kind: str, positions: np.ndarray, *, n_basis: int = 8, hidden: int = 16, r_scale: float = 1.0):
    jnp, nn = require_flax()
    pair_i, pair_j, basis = pair_feature_arrays(positions, n_basis=n_basis, family="sine" if kind == "kan_jastrow" else "rbf", r_scale=r_scale)
    pair_i = jnp.asarray(pair_i); pair_j = jnp.asarray(pair_j); basis = jnp.asarray(basis, dtype=jnp.float64)

    class Product(nn.Module):
        @nn.compact
        def __call__(self, sigma):
            occ = 0.5 * (sigma + 1.0)
            return self.param("bias", nn.initializers.zeros, (), jnp.float64) * jnp.sum(occ, axis=-1)

    class PairJastrow(nn.Module):
        mode: str
        @nn.compact
        def __call__(self, sigma):
            occ = 0.5 * (sigma + 1.0)
            pair_occ = occ[..., pair_i] * occ[..., pair_j]
            count = jnp.sum(occ, axis=-1)
            base = self.param("density_bias", nn.initializers.zeros, (), jnp.float64) * count
            if self.mode == "smooth":
                coeff = self.param("pair_coeff", nn.initializers.normal(0.01), (basis.shape[-1],), jnp.float64)
                edge = basis @ coeff
            elif self.mode == "kan":
                # One-dimensional learnable basis mixer: the interpretable edge curve is basis @ coeff.
                coeff = self.param("kan_coeff", nn.initializers.normal(0.01), (basis.shape[-1],), jnp.float64)
                edge = basis @ coeff
            elif self.mode == "mlp":
                raw_r_features = basis
                edge = nn.Dense(hidden, dtype=jnp.float64, param_dtype=jnp.float64, name="edge_dense_1")(raw_r_features)
                edge = nn.tanh(edge)
                edge = nn.Dense(1, dtype=jnp.float64, param_dtype=jnp.float64, name="edge_dense_2")(edge)[..., 0]
            else:
                raise ValueError(self.mode)
            return base + jnp.sum(pair_occ * edge, axis=-1)

    kind = kind.lower()
    if kind == "product":
        return Product()
    if kind == "smooth_jastrow":
        return PairJastrow(mode="smooth")
    if kind == "kan_jastrow":
        return PairJastrow(mode="kan")
    if kind == "mlp_pair":
        return PairJastrow(mode="mlp")
    if kind == "rbm":
        import netket as nk
        return nk.models.RBM(alpha=max(1, hidden // max(len(positions), 1)), param_dtype=float)
    raise KeyError(f"Unsupported Flax NQS model: {kind}")
