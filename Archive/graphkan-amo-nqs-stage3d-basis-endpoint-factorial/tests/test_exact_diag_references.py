from graphkan_amo_nqs.geometry import make_chain_geometry
from graphkan_amo_nqs.hamiltonians import RydbergParams
from graphkan_amo_nqs.hamiltonians.exact_diag import compute_rydberg_reference

def test_exact_reference():
    ref=compute_rydberg_reference(make_chain_geometry(4), RydbergParams(omega=1,delta=0.2,c6=1))
    assert ref.vector.shape[0] == 16
    assert ref.density.shape[0] == 4
