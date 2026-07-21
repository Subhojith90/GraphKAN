from graphkan_amo_nqs.geometry import make_irregular_geometry
from graphkan_amo_nqs.geometry.hashing import distance_matrix_hash

def test_geometry_hashing():
    a = make_irregular_geometry(6, seed=4)
    b = make_irregular_geometry(6, seed=4)
    c = make_irregular_geometry(6, seed=5)
    assert distance_matrix_hash(a.positions) == distance_matrix_hash(b.positions)
    assert distance_matrix_hash(a.positions) != distance_matrix_hash(c.positions)
