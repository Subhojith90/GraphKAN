from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import scipy.sparse as sp

from graphkan_amo_nqs.geometry.distances import interaction_matrix as rydberg_interaction_matrix
from .basis import enumerate_binary_states


@dataclass(frozen=True)
class RydbergParams:
    omega: float = 1.0
    delta: float = 0.0
    c6: float = 1.0
    sign_rule: str = "parity"
    units: str = "Omega=1"

    @property
    def r_omega(self) -> float:
        if self.omega == 0:
            return float("inf")
        return float((abs(self.c6) / abs(self.omega)) ** (1.0 / 6.0))

    @property
    def r_delta(self) -> float:
        if self.delta == 0:
            return float("inf")
        return float((abs(self.c6) / abs(self.delta)) ** (1.0 / 6.0))


def interaction_matrix(positions: np.ndarray, c6: float, *, distance_matrix: np.ndarray | None = None) -> np.ndarray:
    return rydberg_interaction_matrix(positions, c6, distance_matrix=distance_matrix)


def diagonal_energies(states: np.ndarray, positions: np.ndarray, params: RydbergParams, *, distance_matrix: np.ndarray | None = None) -> np.ndarray:
    states = np.asarray(states, dtype=float)
    v = interaction_matrix(positions, params.c6, distance_matrix=distance_matrix)
    detuning = -float(params.delta) * np.sum(states, axis=1)
    pair = 0.5 * np.einsum("bi,ij,bj->b", states, v, states, optimize=True)
    return detuning + pair


def build_dense_rydberg(positions: np.ndarray, params: RydbergParams, *, distance_matrix: np.ndarray | None = None) -> np.ndarray:
    n = len(positions)
    states = enumerate_binary_states(n)
    diag = diagonal_energies(states, positions, params, distance_matrix=distance_matrix)
    dim = len(states)
    H = np.diag(diag.astype(float))
    off = float(params.omega) / 2.0
    if off != 0:
        for index in range(dim):
            for i in range(n):
                j = index ^ (1 << i)
                H[index, j] += off
    return H


def build_sparse_rydberg(positions: np.ndarray, params: RydbergParams, *, distance_matrix: np.ndarray | None = None) -> sp.csr_matrix:
    n = len(positions)
    states = enumerate_binary_states(n)
    diag = diagonal_energies(states, positions, params, distance_matrix=distance_matrix)
    dim = len(states)
    rows = list(range(dim))
    cols = list(range(dim))
    data = diag.astype(float).tolist()
    off = float(params.omega) / 2.0
    if off != 0:
        for index in range(dim):
            for i in range(n):
                rows.append(index)
                cols.append(index ^ (1 << i))
                data.append(off)
    return sp.coo_matrix((data, (rows, cols)), shape=(dim, dim), dtype=float).tocsr()


def analytic_diagonal_spectrum(positions: np.ndarray, params: RydbergParams, *, distance_matrix: np.ndarray | None = None) -> np.ndarray:
    if abs(params.omega) > 1e-14:
        raise ValueError("Analytic diagonal spectrum requires omega=0")
    states = enumerate_binary_states(len(positions))
    return np.sort(diagonal_energies(states, positions, params))


def instance_id(geometry, params: RydbergParams) -> str:
    from graphkan_amo_nqs.geometry.hashing import stable_hash
    payload = {**geometry.id_fields, "omega": params.omega, "delta": params.delta, "c6": params.c6}
    return f"ryd_{geometry.family}_N{geometry.n_sites}_D{params.delta:g}_C6{params.c6:g}_{stable_hash(payload)}"
