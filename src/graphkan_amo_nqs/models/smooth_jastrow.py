from .radial_jastrow import RadialJastrowModel


class SmoothJastrowModel(RadialJastrowModel):
    def __init__(self, n_basis: int = 6):
        super().__init__(n_basis=n_basis, basis_family="linear_spline", name="smooth_jastrow")
