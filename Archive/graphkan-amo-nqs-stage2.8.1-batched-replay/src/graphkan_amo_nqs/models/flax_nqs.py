"""Phase-aware Flax neural quantum states for Rydberg VMC validation.

For the Hamiltonian convention used here,
H = +(omega/2) sum_i sigma_x^i + diagonal terms,
the stoquastic ground-state gauge has an occupation-parity sign:
psi(sigma) = (-1)^(sum_i n_i) |psi(sigma)| for omega > 0.
The fixed phase below implements that exact gauge convention, while every
trainable model learns only the real log-amplitude.  This removes an artificial
positive-amplitude restriction present in Stage 2.4.
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
        scale = max(float(r.max()), 1e-12)
        B = np.stack([np.sin((k + 1) * np.pi * r / scale) for k in range(n_basis)], axis=1)
    elif family == "rbf":
        centers = np.linspace(float(r.min()), float(r.max()), n_basis)
        width = max(float(np.ptp(centers)) / max(n_basis - 1, 1), 1e-6)
        B = np.exp(-0.5 * ((r[:, None] - centers[None, :]) / width) ** 2)
    else:
        raise ValueError(f"Unknown pair feature family: {family}")
    return i.astype(np.int32), j.astype(np.int32), B.astype(np.float64)


def build_flax_model(kind: str, positions: np.ndarray, *, n_basis: int = 8, hidden: int = 16,
                     r_scale: float = 1.0, omega: float = 1.0, inverted_ordering: bool = False):
    """Construct a phase-aware NQS.

    The parity phase is activated for positive omega.  For omega < 0, the
    positive-amplitude gauge is already the natural one.  omega=0 has no
    off-diagonal sign constraint.
    """
    jnp, nn = require_flax()
    pair_i, pair_j, basis = pair_feature_arrays(
        positions, n_basis=n_basis,
        family="sine" if kind == "kan_jastrow" else "rbf",
        r_scale=r_scale,
    )
    pair_i = jnp.asarray(pair_i)
    pair_j = jnp.asarray(pair_j)
    basis = jnp.asarray(basis, dtype=jnp.float64)
    phase_sign = 1.0 if float(omega) > 0.0 else 0.0
    n_sites = int(len(positions))

    def occupation_from_sigma(sigma):
        # NetKet >=3.16 with inverted_ordering=False uses +1=up and -1=down.
        # We define the physical Rydberg occupation as n=1 for down/excited.
        return 0.5 * (sigma + 1.0) if bool(inverted_ordering) else 0.5 * (1.0 - sigma)

    def parity_phase(occ):
        # exp(i*pi*Nocc) = (-1)^Nocc. Keeping it in log form gives NetKet a
        # valid complex log-amplitude and preserves differentiability.
        return 1j * jnp.pi * phase_sign * jnp.sum(occ, axis=-1)

    class Product(nn.Module):
        @nn.compact
        def __call__(self, sigma):
            occ = occupation_from_sigma(sigma)
            # Open chains should not be forced to have a translation-invariant
            # product amplitude. One field per site is the minimal baseline.
            site_bias = self.param("site_bias", nn.initializers.zeros, (n_sites,), jnp.float64)
            amp = jnp.sum(occ * site_bias, axis=-1)
            return amp.astype(jnp.complex128) + parity_phase(occ)

    class PairJastrow(nn.Module):
        mode: str
        @nn.compact
        def __call__(self, sigma):
            occ = occupation_from_sigma(sigma)
            pair_occ = occ[..., pair_i] * occ[..., pair_j]
            site_bias = self.param("site_bias", nn.initializers.zeros, (n_sites,), jnp.float64)
            amp = jnp.sum(occ * site_bias, axis=-1)
            if self.mode == "smooth":
                coeff = self.param("pair_coeff", nn.initializers.normal(0.005), (basis.shape[-1],), jnp.float64)
                edge = basis @ coeff
            elif self.mode == "kan":
                # Learnable univariate Fourier mixture. The exported curve is
                # interpretable as edge(r) = B(r) dot coeff.
                coeff = self.param("kan_coeff", nn.initializers.normal(0.005), (basis.shape[-1],), jnp.float64)
                edge = basis @ coeff
            elif self.mode == "mlp":
                edge = nn.Dense(hidden, dtype=jnp.float64, param_dtype=jnp.float64,
                                kernel_init=nn.initializers.lecun_normal(),
                                name="edge_dense_1")(basis)
                edge = nn.tanh(edge)
                edge = nn.Dense(1, dtype=jnp.float64, param_dtype=jnp.float64,
                                kernel_init=nn.initializers.zeros,
                                name="edge_dense_2")(edge)[..., 0]
            else:
                raise ValueError(self.mode)
            amp = amp + jnp.sum(pair_occ * edge, axis=-1)
            return amp.astype(jnp.complex128) + parity_phase(occ)

    class RBM(nn.Module):
        @nn.compact
        def __call__(self, sigma):
            # Standard real RBM amplitude with the same fixed parity phase.
            occ = occupation_from_sigma(sigma)
            visible = self.param("visible_bias", nn.initializers.zeros, (n_sites,), jnp.float64)
            W = self.param("weights", nn.initializers.normal(0.01), (n_sites, max(1, hidden)), jnp.float64)
            hb = self.param("hidden_bias", nn.initializers.zeros, (max(1, hidden),), jnp.float64)
            amp = jnp.sum(occ * visible, axis=-1) + jnp.sum(jnp.logaddexp(0.0, occ @ W + hb), axis=-1)
            return amp.astype(jnp.complex128) + parity_phase(occ)

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
        return RBM()
    raise KeyError(f"Unsupported Flax NQS model: {kind}")
