from graphkan_amo_nqs.models.product import ProductStateModel
from graphkan_amo_nqs.models.radial_jastrow import RadialJastrowModel
from graphkan_amo_nqs.geometry import make_chain_geometry
from graphkan_amo_nqs.hamiltonians import RydbergParams, enumerate_binary_states

def test_jastrow_has_more_features_than_product():
    g=make_chain_geometry(4); p=RydbergParams(); states=enumerate_binary_states(4)
    assert RadialJastrowModel().feature_matrix(states,g,p).shape[1] > ProductStateModel().feature_matrix(states,g,p).shape[1]
