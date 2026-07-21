from __future__ import annotations

import numpy as np

from .base import FeatureModel, pair_distances, pair_sums
from .kan_basis import basis_matrix, normalise_columns


class KANJastrowModel(FeatureModel):
    def __init__(self, n_basis: int = 10, basis_family: str = "sine", feature_policy: str = "rho_omega"):
        self.n_basis = n_basis
        self.basis_family = basis_family
        self.feature_policy = feature_policy
        super().__init__(name="kan_jastrow", n_features=0, metadata={"basis_family": basis_family, "n_basis": n_basis, "feature_policy": feature_policy})

    def _x(self, r: np.ndarray, params):
        r = np.asarray(r, dtype=float)
        if self.feature_policy == "raw_distance":
            return r
        if self.feature_policy == "rho_delta":
            return r / params.r_delta if np.isfinite(params.r_delta) and params.r_delta != 0 else r
        if self.feature_policy == "v_over_omega":
            return np.log1p(np.abs(params.c6 / (abs(params.omega) * np.maximum(r, 1e-12) ** 6))) if params.omega != 0 else np.zeros_like(r)
        return r / params.r_omega if np.isfinite(params.r_omega) and params.r_omega != 0 else r

    def feature_matrix(self, states: np.ndarray, geometry, params):
        r = pair_distances(geometry.positions)
        B = basis_matrix(self._x(r, params), self.basis_family, self.n_basis)
        B = normalise_columns(B)
        pair_features = pair_sums(states, geometry.positions, B)
        extra = np.full((len(states), 1), params.delta / params.omega if params.omega != 0 else 0.0)
        X = np.concatenate([np.ones((len(states), 1)), np.sum(states, axis=1, keepdims=True), extra, pair_features], axis=1)
        self.n_features = X.shape[1]
        return X

    def edge_function(self, theta: np.ndarray, r_grid: np.ndarray, params):
        B = basis_matrix(self._x(np.asarray(r_grid, dtype=float), params), self.basis_family, self.n_basis)
        return B @ theta[3:3 + self.n_basis]
