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


def resolve_basis_family(kind: str, basis_family: str | None = None) -> str:
    """Resolve the radial pair basis explicitly and fail on ambiguous overrides.

    KAN variants use the sine/Fourier radial family by default; smooth Jastrow
    uses RBF. The selected family is a first-class run attribute and must be
    recorded by callers.
    """
    kind = kind.lower()
    defaults = {
        "smooth_jastrow": "rbf",
        "smooth_endpoint_sum_jastrow": "rbf",
        "kan_jastrow": "sine",
        "kan_boundary_jastrow": "sine",
        "kan_endpoint_jastrow": "sine",
        "kan_endpoint_sum_jastrow": "sine",
        "kan_endpoint_diff_jastrow": "sine",
        "kan_endpoint_product_jastrow": "sine",
        "mlp_pair": "rbf",
        "mlp_endpoint_pair": "rbf",
    }
    if kind not in defaults:
        return "none"
    selected = defaults[kind] if basis_family is None else str(basis_family).lower()
    if selected not in {"sine", "rbf"}:
        raise ValueError(f"Unknown basis_family: {selected}")
    return selected


def build_flax_model(kind: str, positions: np.ndarray, *, n_basis: int = 8, hidden: int = 16,
                     r_scale: float = 1.0, omega: float = 1.0, inverted_ordering: bool = False, basis_family: str | None = None):
    """Construct a phase-aware NQS.

    The parity phase is activated for positive omega.  For omega < 0, the
    positive-amplitude gauge is already the natural one.  omega=0 has no
    off-diagonal sign constraint.
    """
    jnp, nn = require_flax()
    n_sites = int(len(positions))
    selected_basis_family = resolve_basis_family(kind, basis_family)
    pair_i_np, pair_j_np, basis_np = pair_feature_arrays(
        positions, n_basis=n_basis,
        family=selected_basis_family,
        r_scale=r_scale,
    )
    # For an open chain, equal pair distance does not imply equal boundary environment.
    # A reflection-symmetric pair-centre coordinate adds only minimal boundary awareness.
    denom = max(n_sites - 1, 1)
    centre = (pair_i_np + pair_j_np) / (2.0 * denom)
    u = np.abs(2.0 * centre - 1.0)
    boundary_order = max(1, int(hidden))
    boundary_np = np.stack([u ** (k + 1) for k in range(boundary_order)], axis=1)
    pair_i = jnp.asarray(pair_i_np)
    pair_j = jnp.asarray(pair_j_np)
    basis = jnp.asarray(basis_np, dtype=jnp.float64)
    boundary_basis = jnp.asarray(boundary_np, dtype=jnp.float64)
    # Endpoint-aware features distinguish edge-edge, edge-bulk and bulk-bulk
    # pairs at equal separation while remaining reflection symmetric. For each
    # site, d_i is distance to the nearest open-chain boundary, normalised to [0,1].
    site_d = np.minimum(np.arange(n_sites), (n_sites - 1) - np.arange(n_sites)).astype(np.float64)
    site_d = site_d / max(float(site_d.max()), 1.0)
    di = site_d[pair_i_np]
    dj = site_d[pair_j_np]
    endpoint_np = np.stack([di + dj, np.abs(di - dj), di * dj], axis=1)
    endpoint_basis = jnp.asarray(endpoint_np, dtype=jnp.float64)
    phase_sign = 1.0 if float(omega) > 0.0 else 0.0

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
            elif self.mode == "smooth_endpoint_sum":
                coeff = self.param("pair_coeff", nn.initializers.normal(0.005), (basis.shape[-1],), jnp.float64)
                endpoint_coeff = self.param("endpoint_coeff", nn.initializers.zeros, (1,), jnp.float64)
                edge = basis @ coeff + endpoint_basis[:, 0] * endpoint_coeff[0]
            elif self.mode == "kan":
                # Learnable univariate Fourier mixture. The exported curve is
                # interpretable as edge(r) = B(r) dot coeff.
                coeff = self.param("kan_coeff", nn.initializers.normal(0.005), (basis.shape[-1],), jnp.float64)
                edge = basis @ coeff
            elif self.mode == "kan_boundary":
                # Radial KAN plus a reflection-symmetric open-boundary correction.
                # This is not Graph-KAN: it is a controlled pair-centre ablation.
                coeff = self.param("kan_coeff", nn.initializers.normal(0.005), (basis.shape[-1],), jnp.float64)
                boundary_coeff = self.param("boundary_coeff", nn.initializers.zeros, (boundary_basis.shape[-1],), jnp.float64)
                edge = basis @ coeff + boundary_basis @ boundary_coeff
            elif self.mode.startswith("kan_endpoint"):
                # Explicit ablations of reflection-symmetric endpoint features.
                # "full" uses [d_i+d_j, |d_i-d_j|, d_i d_j]; the single-feature
                # variants retain exactly one term while preserving the selected radial basis family.
                coeff = self.param("kan_coeff", nn.initializers.normal(0.005), (basis.shape[-1],), jnp.float64)
                if self.mode == "kan_endpoint_sum":
                    feat = endpoint_basis[:, :1]
                elif self.mode == "kan_endpoint_diff":
                    feat = endpoint_basis[:, 1:2]
                elif self.mode == "kan_endpoint_product":
                    feat = endpoint_basis[:, 2:3]
                elif self.mode == "kan_endpoint_hybrid":
                    # Controlled two-feature hybrid: endpoint sum plus endpoint product.
                    feat = endpoint_basis[:, [0, 2]]
                else:
                    feat = endpoint_basis
                endpoint_coeff = self.param("endpoint_coeff", nn.initializers.zeros, (feat.shape[-1],), jnp.float64)
                edge = basis @ coeff + feat @ endpoint_coeff
            elif self.mode in {"mlp", "mlp_endpoint"}:
                features = basis if self.mode == "mlp" else jnp.concatenate([basis, endpoint_basis], axis=-1)
                edge = nn.Dense(hidden, dtype=jnp.float64, param_dtype=jnp.float64,
                                kernel_init=nn.initializers.lecun_normal(),
                                name="edge_dense_1")(features)
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
    if kind == "smooth_endpoint_sum_jastrow":
        return PairJastrow(mode="smooth_endpoint_sum")
    if kind == "kan_jastrow":
        return PairJastrow(mode="kan")
    if kind == "kan_boundary_jastrow":
        return PairJastrow(mode="kan_boundary")
    if kind == "kan_endpoint_jastrow":
        return PairJastrow(mode="kan_endpoint_full")
    if kind == "kan_endpoint_sum_jastrow":
        return PairJastrow(mode="kan_endpoint_sum")
    if kind == "kan_endpoint_diff_jastrow":
        return PairJastrow(mode="kan_endpoint_diff")
    if kind == "kan_endpoint_product_jastrow":
        return PairJastrow(mode="kan_endpoint_product")
    if kind == "kan_endpoint_hybrid_jastrow":
        return PairJastrow(mode="kan_endpoint_hybrid")
    if kind == "mlp_pair":
        return PairJastrow(mode="mlp")
    if kind == "mlp_endpoint_pair":
        return PairJastrow(mode="mlp_endpoint")
    if kind == "rbm":
        return RBM()
    raise KeyError(f"Unsupported Flax NQS model: {kind}")
