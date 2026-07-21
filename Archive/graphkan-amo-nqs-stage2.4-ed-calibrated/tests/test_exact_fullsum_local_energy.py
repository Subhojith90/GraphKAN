import numpy as np
from graphkan_amo_nqs.geometry import make_chain_geometry
from graphkan_amo_nqs.hamiltonians import RydbergParams, enumerate_binary_states, build_sparse_rydberg
from graphkan_amo_nqs.models import make_model
from graphkan_amo_nqs.training.exact_fullsum import train_feature_model, variational_state
from graphkan_amo_nqs.observables.local_energy import energy_from_state

def test_training_energy_consistency():
    g=make_chain_geometry(4); p=RydbergParams(omega=1,delta=0.5,c6=2)
    states=enumerate_binary_states(4); H=build_sparse_rydberg(g.positions,p)
    m=make_model("product")
    res=train_feature_model(m,states,g,p,H,seed=0,maxiter=3)
    psi=variational_state(res.theta,m,states,g,p)
    assert abs(energy_from_state(H,psi)-res.energy) < 1e-8
