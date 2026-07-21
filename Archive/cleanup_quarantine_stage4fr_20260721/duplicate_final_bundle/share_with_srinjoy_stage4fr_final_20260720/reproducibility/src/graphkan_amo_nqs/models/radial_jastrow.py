from __future__ import annotations

import numpy as np

from .base import FeatureModel, pair_distances, pair_sums
from .kan_basis import basis_matrix, normalise_columns


class RadialJastrowModel(FeatureModel):
    def __init__(self, n_basis: int = 8, basis_family: str = "rbf", name: str = "radial_jastrow"):
        self.n_basis = n_basis
        self.basis_family = basis_family
        super().__init__(name=name, n_features=0, metadata={"basis_family": basis_family, "n_basis": n_basis})

    def feature_matrix(self, states: np.ndarray, geometry, params):
        r = pair_distances(geometry.positions)
        x = r / params.r_omega if np.isfinite(params.r_omega) and params.r_omega != 0 else r
        B = basis_matrix(x, self.basis_family, self.n_basis)
        B = normalise_columns(B)
        pair_features = pair_sums(states, geometry.positions, B)
        X = np.concatenate([np.ones((len(states), 1)), np.sum(states, axis=1, keepdims=True), pair_features], axis=1)
        self.n_features = X.shape[1]
        self._last_B = B; self._last_r = r
        return X

    def edge_function(self, theta: np.ndarray, r_grid: np.ndarray, params):
        x = r_grid / params.r_omega if np.isfinite(params.r_omega) and params.r_omega != 0 else r_grid
        B = basis_matrix(x, self.basis_family, self.n_basis)
        return B @ theta[2:2 + self.n_basis]
