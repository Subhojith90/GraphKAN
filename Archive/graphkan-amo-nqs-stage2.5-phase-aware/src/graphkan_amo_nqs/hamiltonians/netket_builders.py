from __future__ import annotations

import numpy as np

from graphkan_amo_nqs.exceptions import DependencyUnavailable


def require_netket():
    try:
        import netket as nk  # type: ignore
        return nk
    except Exception as exc:
        raise DependencyUnavailable("NetKet is unavailable. Create the Python 3.11 environment and install with `pip install -e '.[nqs]'`.") from exc


def build_netket_rydberg_operator(positions: np.ndarray | None = None, params=None):
    """Build the same binary-occupation Rydberg Hamiltonian used by the exact engine.

    NetKet Spin local states are ordered as (-1, +1). The local number operator is
    n = diag(0, 1), and the transverse field is omega/2 * sigma_x. This matches
    `build_dense_rydberg` exactly under the binary occupation encoding.
    """
    nk = require_netket()
    if positions is None or params is None:
        raise ValueError("positions and params are required to build a NetKet Rydberg operator.")
    positions = np.asarray(positions, dtype=float)
    n_sites = len(positions)
    hi = nk.hilbert.Spin(s=0.5, N=n_sites)
    number = np.asarray([[0.0, 0.0], [0.0, 1.0]])
    sx = np.asarray([[0.0, 1.0], [1.0, 0.0]])
    hamiltonian = nk.operator.LocalOperator(hi, dtype=float)
    for i in range(n_sites):
        hamiltonian += nk.operator.LocalOperator(hi, -float(params.delta) * number, [i], dtype=float)
        hamiltonian += nk.operator.LocalOperator(hi, 0.5 * float(params.omega) * sx, [i], dtype=float)
    for i in range(n_sites):
        for j in range(i + 1, n_sites):
            r = float(np.linalg.norm(positions[i] - positions[j]))
            if r <= 0:
                raise ValueError("Rydberg geometry contains coincident sites.")
            vij = float(params.c6) / (r ** 6)
            pair_n = np.kron(number, number)
            hamiltonian += nk.operator.LocalOperator(hi, vij * pair_n, [i, j], dtype=float)
    return hi, hamiltonian
