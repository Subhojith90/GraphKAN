import numpy as np
from graphkan_amo_nqs.geometry import make_chain_geometry
from graphkan_amo_nqs.hamiltonians import RydbergParams, build_dense_rydberg, build_sparse_rydberg

def test_dense_sparse_agree():
    g=make_chain_geometry(4)
    p=RydbergParams(omega=1.0, delta=0.5, c6=2.0)
    assert np.allclose(build_dense_rydberg(g.positions,p), build_sparse_rydberg(g.positions,p).toarray())
