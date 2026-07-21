from __future__ import annotations

import numpy as np

from graphkan_amo_nqs.seed import derive_seed
from .base import FeatureModel
from .kan_basis import normalise_columns


class RBMFeatureModel(FeatureModel):
    """RBM-like random-hidden-feature baseline for exact CPU pilots.

    This is not NetKet's trainable RBM. It gives a non-linear global occupation feature baseline
    without optional NetKet; reports label it as `rbm_feature`.
    """
    def __init__(self, hidden: int = 12, seed: int = 0):
        self.hidden = hidden; self.seed = seed
        super().__init__(name="rbm_feature", n_features=0, metadata={"hidden": hidden, "not_full_rbm": True})

    def feature_matrix(self, states: np.ndarray, geometry, params):
        n = states.shape[1]
        rng = np.random.default_rng(derive_seed(self.seed, "rbm_feature", n, self.hidden))
        W = rng.normal(scale=0.7 / max(n, 1) ** 0.5, size=(n, self.hidden))
        b = rng.normal(scale=0.1, size=self.hidden)
        hidden = np.log(2 * np.cosh(states @ W + b))
        X = np.concatenate([np.ones((len(states), 1)), states, normalise_columns(hidden)], axis=1)
        self.n_features = X.shape[1]
        return X
