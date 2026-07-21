from .kan_jastrow import KANJastrowModel


class SineKANBaseline(KANJastrowModel):
    def __init__(self, n_basis: int = 10):
        super().__init__(n_basis=n_basis, basis_family="sine", feature_policy="raw_distance")
        self.name = "sinekan_distance_baseline"
