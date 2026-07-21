from __future__ import annotations

import numpy as np

from graphkan_amo_nqs.seed import derive_seed
from .base import FeatureModel, pair_distances, pair_sums, blockade_features_for_distances
from .kan_basis import normalise_columns


class MLPPairFeatureModel(FeatureModel):
    """CPU exact-basis MLP-pair random-feature baseline.

    The hidden projection is fixed by seed so the optimisation remains exact and fast.
    Production Flax/NetKet runs should replace this with a fully trainable MLP-pair.
    """
    def __init__(self, hidden: int = 10, seed: int = 0):
        self.hidden = hidden
        self.seed = seed
        super().__init__(name="mlp_pair", n_features=0, metadata={"hidden": hidden, "feature_policy": "same_blockade_features_fixed_random_hidden"})

    def _edge_basis(self, r: np.ndarray, params):
        Z = blockade_features_for_distances(r, params)
        rng = np.random.default_rng(derive_seed(self.seed, "mlp_pair_hidden", self.hidden))
        W = rng.normal(scale=0.8, size=(Z.shape[1], self.hidden))
        b = rng.normal(scale=0.2, size=self.hidden)
        return normalise_columns(np.tanh(Z @ W + b))

    def feature_matrix(self, states: np.ndarray, geometry, params):
        r = pair_distances(geometry.positions)
        B = self._edge_basis(r, params)
        pair_features = pair_sums(states, geometry.positions, B)
        X = np.concatenate([np.ones((len(states), 1)), np.sum(states, axis=1, keepdims=True), pair_features], axis=1)
        self.n_features = X.shape[1]
        return X

    def edge_function(self, theta: np.ndarray, r_grid: np.ndarray, params):
        B = self._edge_basis(np.asarray(r_grid, dtype=float), params)
        return B @ theta[2:2 + self.hidden]
