from __future__ import annotations

import numpy as np

from .mlp_pair import MLPPairFeatureModel


class GraphMPNNFeatureModel(MLPPairFeatureModel):
    def __init__(self, hidden: int = 12, seed: int = 0):
        super().__init__(hidden=hidden, seed=seed)
        self.name = "graph_mpnn"
        self.metadata["graph_baseline"] = "fixed_random_message_features_cpu"
