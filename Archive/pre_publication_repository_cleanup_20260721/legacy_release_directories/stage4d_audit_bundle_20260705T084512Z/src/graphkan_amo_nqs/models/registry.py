from __future__ import annotations

from .product import ProductStateModel
from .radial_jastrow import RadialJastrowModel
from .smooth_jastrow import SmoothJastrowModel
from .mlp_pair import MLPPairFeatureModel
from .kan_jastrow import KANJastrowModel
from .graph_mpnn import GraphMPNNFeatureModel
from .graph_kan import GraphKANModel
from .rbm_netket import RBMFeatureModel
from .sinekan_baseline import SineKANBaseline


def make_model(name: str, seed: int = 0):
    key = name.lower()
    if key == "product":
        return ProductStateModel(site_resolved=False)
    if key in {"radial_jastrow", "distance_binned_jastrow"}:
        return RadialJastrowModel(n_basis=8, basis_family="rbf", name="radial_jastrow")
    if key == "smooth_jastrow":
        return SmoothJastrowModel(n_basis=6)
    if key == "mlp_pair":
        return MLPPairFeatureModel(hidden=10, seed=seed)
    if key == "kan_jastrow":
        return KANJastrowModel(n_basis=10, basis_family="sine", feature_policy="rho_omega")
    if key == "graph_mpnn":
        return GraphMPNNFeatureModel(hidden=12, seed=seed)
    if key == "graph_kan":
        return GraphKANModel(n_basis=10)
    if key in {"rbm", "rbm_feature"}:
        return RBMFeatureModel(hidden=12, seed=seed)
    if key == "sinekan_baseline":
        return SineKANBaseline(n_basis=10)
    raise KeyError(f"Unknown model: {name}")


def registered_models() -> list[str]:
    return ["product", "radial_jastrow", "smooth_jastrow", "mlp_pair", "kan_jastrow", "graph_mpnn", "graph_kan", "rbm_feature", "sinekan_baseline"]
