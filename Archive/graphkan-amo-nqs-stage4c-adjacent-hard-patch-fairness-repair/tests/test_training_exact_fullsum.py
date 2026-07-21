from graphkan_amo_nqs.geometry import make_chain_geometry
from graphkan_amo_nqs.hamiltonians import RydbergParams, enumerate_binary_states, build_sparse_rydberg
from graphkan_amo_nqs.models import make_model
from graphkan_amo_nqs.training import train_feature_model

def test_train_feature_model_runs():
    g=make_chain_geometry(4); p=RydbergParams(); states=enumerate_binary_states(4); H=build_sparse_rydberg(g.positions,p)
    res=train_feature_model(make_model("kan_jastrow"), states, g, p, H, maxiter=2)
    assert res.theta.size > 0
