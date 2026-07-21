import numpy as np

from graphkan_amo_nqs.geometry import make_chain_geometry, make_periodic_chain_geometry
from graphkan_amo_nqs.geometry.distances import geometry_distance_matrix, interaction_matrix
from graphkan_amo_nqs.geometry.hashing import geometry_id
from graphkan_amo_nqs.hamiltonians.rydberg import RydbergParams, build_dense_rydberg


def test_periodic_chain_minimum_image_and_identity():
    open_g = make_chain_geometry(6, spacing=1.0)
    periodic_g = make_periodic_chain_geometry(6, spacing=1.0)
    d_open = geometry_distance_matrix(open_g)
    d_periodic = geometry_distance_matrix(periodic_g)
    assert np.isclose(d_open[0, 5], 5.0)
    assert np.isclose(d_periodic[0, 5], 1.0)
    assert geometry_id(open_g) != geometry_id(periodic_g)


def test_periodic_hamiltonian_differs_from_open():
    params = RydbergParams(omega=1.0, delta=1.5, c6=4.0)
    open_g = make_chain_geometry(6)
    periodic_g = make_periodic_chain_geometry(6)
    ho = build_dense_rydberg(open_g.positions, params, distance_matrix=geometry_distance_matrix(open_g))
    hp = build_dense_rydberg(periodic_g.positions, params, distance_matrix=geometry_distance_matrix(periodic_g))
    assert not np.allclose(ho, hp)
