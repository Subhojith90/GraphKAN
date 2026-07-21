import numpy as np
import pytest
from graphkan_amo_nqs.exceptions import DependencyUnavailable
from graphkan_amo_nqs.hamiltonians.netket_builders import build_netket_rydberg_operator
from graphkan_amo_nqs.hamiltonians.rydberg import RydbergParams, build_dense_rydberg

def test_explicit_new_ordering_operator_matches_dense():
    pos=np.arange(2,dtype=float).reshape(-1,1); params=RydbergParams(omega=1,delta=1,c6=1)
    try: hi,h=build_netket_rydberg_operator(pos,params,inverted_ordering=False)
    except DependencyUnavailable: pytest.skip('NetKet optional')
    states=np.asarray(hi.all_states(),dtype=float)
    occ=0.5*(1.0-states)
    assert np.all(np.isin(occ,[0.0,1.0]))
    assert np.max(np.abs(np.asarray(h.to_dense())-build_dense_rydberg(pos,params))) < 1e-10
