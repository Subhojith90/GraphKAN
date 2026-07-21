import numpy as np
from graphkan_amo_nqs.geometry import make_chain_geometry
from graphkan_amo_nqs.hamiltonians import RydbergParams, build_dense_rydberg, analytic_diagonal_spectrum

def test_omega_zero_spectrum():
    g=make_chain_geometry(5)
    p=RydbergParams(omega=0.0, delta=1.0, c6=1.5)
    assert np.allclose(np.sort(np.linalg.eigvalsh(build_dense_rydberg(g.positions,p))), analytic_diagonal_spectrum(g.positions,p))
