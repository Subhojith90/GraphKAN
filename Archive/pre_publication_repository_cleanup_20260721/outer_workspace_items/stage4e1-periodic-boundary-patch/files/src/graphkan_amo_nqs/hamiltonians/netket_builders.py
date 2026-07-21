from __future__ import annotations

import numpy as np

from graphkan_amo_nqs.exceptions import DependencyUnavailable


def require_netket():
    try:
        import netket as nk  # type: ignore
        return nk
    except Exception as exc:
        raise DependencyUnavailable("NetKet is unavailable. Create the Python 3.11 environment and install with `pip install -e '.[nqs]'`.") from exc


def build_netket_rydberg_operator(positions: np.ndarray | None = None, params=None, *, inverted_ordering: bool = False, distance_matrix: np.ndarray | None = None):
    """Build the same binary-occupation Rydberg Hamiltonian used by the exact engine.

    The local-state ordering is explicitly pinned through ``inverted_ordering``.
    Stage 2.6 uses ``False`` (NetKet's current convention: +1=up, -1=down),
    therefore physical Rydberg occupation is n=(1-sigma)/2 in model and sampler
    diagnostics. The local number matrix is n=diag(0,1) in the operator basis.
    """
    nk = require_netket()
    if positions is None or params is None:
        raise ValueError("positions and params are required to build a NetKet Rydberg operator.")
    positions = np.asarray(positions, dtype=float)
    n_sites = len(positions)
    distances = None if distance_matrix is None else np.asarray(distance_matrix, dtype=float)
    if distances is not None and distances.shape != (n_sites, n_sites):
        raise ValueError("distance_matrix shape does not match positions")
    hi = nk.hilbert.Spin(s=0.5, N=n_sites, inverted_ordering=bool(inverted_ordering))
    number = np.asarray([[0.0, 0.0], [0.0, 1.0]])
    sx = np.asarray([[0.0, 1.0], [1.0, 0.0]])
    hamiltonian = nk.operator.LocalOperator(hi, dtype=float)
    for i in range(n_sites):
        hamiltonian += nk.operator.LocalOperator(hi, -float(params.delta) * number, [i], dtype=float)
        hamiltonian += nk.operator.LocalOperator(hi, 0.5 * float(params.omega) * sx, [i], dtype=float)
    for i in range(n_sites):
        for j in range(i + 1, n_sites):
            r = float(distances[i, j]) if distances is not None else float(np.linalg.norm(positions[i] - positions[j]))
            if r <= 0:
                raise ValueError("Rydberg geometry contains coincident sites.")
            vij = float(params.c6) / (r ** 6)
            pair_n = np.kron(number, number)
            hamiltonian += nk.operator.LocalOperator(hi, vij * pair_n, [i, j], dtype=float)
    return hi, hamiltonian
