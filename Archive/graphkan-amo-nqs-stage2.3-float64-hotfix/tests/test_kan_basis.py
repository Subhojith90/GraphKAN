import numpy as np
from graphkan_amo_nqs.models.kan_basis import basis_matrix

def test_kan_basis_families():
    x=np.linspace(0,1,5)
    for fam in ["sine","chebyshev","rbf","linear_spline"]:
        assert basis_matrix(x,fam,4).shape == (5,4)
