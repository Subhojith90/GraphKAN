from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from graphkan_amo_nqs.geometry.distances import pairwise_distances, interaction_matrix
from graphkan_amo_nqs.hamiltonians.rydberg import RydbergParams


@dataclass
class FeatureModel:
    name: str
    n_features: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def feature_matrix(self, states: np.ndarray, geometry, params: RydbergParams) -> np.ndarray:
        raise NotImplementedError

    def initial_params(self, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.normal(scale=0.01, size=self.n_features)

    def param_count(self) -> int:
        return int(self.n_features)

    def edge_function(self, theta: np.ndarray, r_grid: np.ndarray, params: RydbergParams) -> np.ndarray | None:
        return None


def pair_sums(states: np.ndarray, positions: np.ndarray, edge_basis: np.ndarray | None = None) -> np.ndarray:
    n = states.shape[1]
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    occ = np.stack([states[:, i] * states[:, j] for i, j in pairs], axis=1).astype(float)
    if edge_basis is None:
        return occ
    return occ @ edge_basis


def pair_distances(positions: np.ndarray) -> np.ndarray:
    d = pairwise_distances(positions)
    n = len(positions)
    return np.array([d[i, j] for i in range(n) for j in range(i + 1, n)], dtype=float)


def blockade_features_for_distances(r: np.ndarray, params: RydbergParams) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    rho_o = r / params.r_omega if np.isfinite(params.r_omega) and params.r_omega != 0 else r
    rho_d = r / params.r_delta if np.isfinite(params.r_delta) and params.r_delta != 0 else r
    v_over_o = params.c6 / (abs(params.omega) * np.maximum(r, 1e-12) ** 6) if params.omega != 0 else np.zeros_like(r)
    delta_over_o = np.full_like(r, params.delta / params.omega if params.omega != 0 else 0.0)
    return np.stack([rho_o, rho_d, np.log1p(np.abs(v_over_o)), delta_over_o], axis=1)
