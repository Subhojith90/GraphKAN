from graphkan_amo_nqs.geometry import make_chain_geometry
from graphkan_amo_nqs.hamiltonians import RydbergParams, enumerate_binary_states
from graphkan_amo_nqs.models.registry import registered_models, make_model

def test_model_shapes():
    g=make_chain_geometry(5); p=RydbergParams(); states=enumerate_binary_states(5)
    for name in ["product","radial_jastrow","smooth_jastrow","mlp_pair","kan_jastrow","rbm_feature"]:
        m=make_model(name, seed=0); X=m.feature_matrix(states,g,p)
        assert X.shape[0] == 32
        assert X.shape[1] == m.param_count()
