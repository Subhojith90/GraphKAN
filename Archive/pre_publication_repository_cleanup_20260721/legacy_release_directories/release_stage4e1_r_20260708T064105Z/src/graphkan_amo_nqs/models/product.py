from __future__ import annotations

import numpy as np

from .base import FeatureModel


class ProductStateModel(FeatureModel):
    def __init__(self, site_resolved: bool = False):
        self.site_resolved = site_resolved
        super().__init__(name="product", n_features=0, metadata={"site_resolved": site_resolved})

    def feature_matrix(self, states: np.ndarray, geometry, params):
        states = np.asarray(states, dtype=float)
        if self.site_resolved:
            X = np.concatenate([np.ones((len(states), 1)), states], axis=1)
        else:
            X = np.stack([np.ones(len(states)), np.sum(states, axis=1)], axis=1)
        self.n_features = X.shape[1]
        return X
