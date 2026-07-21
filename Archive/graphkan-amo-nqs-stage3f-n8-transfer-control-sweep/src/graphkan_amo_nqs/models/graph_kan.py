from __future__ import annotations

from .kan_jastrow import KANJastrowModel


class GraphKANModel(KANJastrowModel):
    def __init__(self, n_basis: int = 10, basis_family: str = "sine", feature_policy: str = "rho_omega"):
        super().__init__(n_basis=n_basis, basis_family=basis_family, feature_policy=feature_policy)
        self.name = "graph_kan"
        self.metadata["graph_scope"] = "pair-edge CPU exact feature model; full message passing requires NetKet/Flax profile"
