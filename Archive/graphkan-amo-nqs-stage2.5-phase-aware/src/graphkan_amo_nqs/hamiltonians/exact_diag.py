from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.linalg
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .basis import enumerate_binary_states
from .rydberg import RydbergParams, build_dense_rydberg, build_sparse_rydberg
from graphkan_amo_nqs.observables.density import density_from_state
from graphkan_amo_nqs.observables.correlations import connected_correlations_from_state


@dataclass(frozen=True)
class ExactReference:
    energy: float
    vector: np.ndarray
    method: str
    n_sites: int
    density: np.ndarray
    connected: np.ndarray


def lowest_eigenpair(matrix, dense_threshold: int = 512, tol: float = 1e-10) -> tuple[float, np.ndarray, str]:
    dim = matrix.shape[0]
    if dim <= dense_threshold:
        arr = matrix.toarray() if sp.issparse(matrix) else np.asarray(matrix)
        vals, vecs = scipy.linalg.eigh(arr, subset_by_index=[0, 0])
        return float(vals[0]), np.asarray(vecs[:, 0], dtype=float), "dense_eigh"
    mat = matrix if sp.issparse(matrix) else sp.csr_matrix(matrix)
    try:
        vals, vecs = spla.eigsh(mat, k=1, which="SA", tol=tol, v0=np.ones(dim) / np.sqrt(dim))
        return float(vals[0]), np.asarray(vecs[:, 0], dtype=float), "sparse_eigsh"
    except spla.ArpackNoConvergence as err:
        if err.eigenvalues is not None and len(err.eigenvalues):
            return float(err.eigenvalues[0]), np.asarray(err.eigenvectors[:, 0], dtype=float), "sparse_eigsh_partial"
        raise


def compute_rydberg_reference(geometry, params: RydbergParams, dense_threshold: int = 512) -> ExactReference:
    H = build_sparse_rydberg(geometry.positions, params)
    energy, vec, method = lowest_eigenpair(H, dense_threshold=dense_threshold)
    states = enumerate_binary_states(geometry.n_sites)
    density = density_from_state(vec, states)
    connected = connected_correlations_from_state(vec, states)
    return ExactReference(energy=energy, vector=vec, method=method, n_sites=geometry.n_sites, density=density, connected=connected)
