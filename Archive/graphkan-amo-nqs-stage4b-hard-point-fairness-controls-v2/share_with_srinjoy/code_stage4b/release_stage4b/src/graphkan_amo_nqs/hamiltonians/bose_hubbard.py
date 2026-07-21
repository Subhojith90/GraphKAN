from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True)
class BoseHubbardParams:
    J: float = 1.0
    U: float = 1.0
    V: float = 0.0
    mu: float = 0.0
    n_max: int = 2


def enumerate_boson_states(n_sites: int, n_max: int) -> np.ndarray:
    grids = np.meshgrid(*[np.arange(n_max + 1, dtype=np.int16) for _ in range(n_sites)], indexing="ij")
    return np.stack([g.ravel() for g in grids], axis=1)


def boson_bits_to_index(state: np.ndarray, n_max: int) -> int:
    base = n_max + 1
    return int(np.sum(np.asarray(state, dtype=int) * (base ** np.arange(len(state)))))


def diagonal_energies(states: np.ndarray, edges: list[tuple[int, int]], params: BoseHubbardParams) -> np.ndarray:
    n = states.astype(float)
    onsite = 0.5 * params.U * np.sum(n * (n - 1), axis=1) - params.mu * np.sum(n, axis=1)
    nn = np.zeros(len(states), dtype=float)
    for i, j in edges:
        nn += params.V * n[:, i] * n[:, j]
    return onsite + nn


def build_sparse_bose_hubbard(n_sites: int, edges: list[tuple[int, int]], params: BoseHubbardParams) -> sp.csr_matrix:
    states = enumerate_boson_states(n_sites, params.n_max)
    dim = len(states)
    rows = list(range(dim)); cols = list(range(dim)); data = diagonal_energies(states, edges, params).tolist()
    if params.J != 0:
        for idx, s in enumerate(states):
            for i, j in edges:
                for src, dst in ((i, j), (j, i)):
                    if s[src] > 0 and s[dst] < params.n_max:
                        t = s.copy(); t[src] -= 1; t[dst] += 1
                        amp = -params.J * np.sqrt(s[src] * (s[dst] + 1))
                        rows.append(idx); cols.append(boson_bits_to_index(t, params.n_max)); data.append(float(amp))
    return sp.coo_matrix((data, (rows, cols)), shape=(dim, dim)).tocsr()
