import numpy as np
import pytest

pytest.importorskip("netket")

from graphkan_amo_nqs.geometry import make_periodic_chain_geometry
from graphkan_amo_nqs.geometry.distances import geometry_distance_matrix
from graphkan_amo_nqs.hamiltonians.netket_builders import build_netket_rydberg_operator
from graphkan_amo_nqs.hamiltonians.rydberg import RydbergParams, build_dense_rydberg


def test_periodic_netket_matches_dense():
    g = make_periodic_chain_geometry(5)
    p = RydbergParams(omega=1.0, delta=1.5, c6=4.0)
    d = geometry_distance_matrix(g)
    _, h = build_netket_rydberg_operator(g.positions, p, distance_matrix=d, inverted_ordering=False)
    dense = build_dense_rydberg(g.positions, p, distance_matrix=d)
    assert np.max(np.abs(np.asarray(h.to_dense()) - dense)) < 1e-10
