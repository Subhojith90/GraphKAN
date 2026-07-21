import numpy as np
from graphkan_amo_nqs.hamiltonians.bose_hubbard import BoseHubbardParams, enumerate_boson_states, diagonal_energies, build_sparse_bose_hubbard

def test_bose_j_zero_diagonal():
    p=BoseHubbardParams(J=0.0,U=2.0,V=0.5,mu=0.1,n_max=2)
    edges=[(0,1)]
    H=build_sparse_bose_hubbard(2,edges,p).toarray()
    states=enumerate_boson_states(2,2)
    assert np.allclose(np.diag(H), diagonal_energies(states, edges, p))
    assert np.allclose(H, np.diag(np.diag(H)))
