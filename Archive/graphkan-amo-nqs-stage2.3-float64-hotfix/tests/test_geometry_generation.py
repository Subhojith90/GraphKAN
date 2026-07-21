import numpy as np
from graphkan_amo_nqs.geometry import make_chain_geometry, make_ladder_geometry, make_square_grid_geometry, make_triangular_fragment, make_kagome_fragment, make_irregular_geometry
from graphkan_amo_nqs.geometry.distances import pairwise_distances

def test_geometry_generation():
    geoms=[make_chain_geometry(4), make_ladder_geometry(3,2), make_square_grid_geometry(2,2), make_triangular_fragment(2,2), make_kagome_fragment(1,1), make_irregular_geometry(5, seed=0)]
    for g in geoms:
        d=pairwise_distances(g.positions)
        assert d.shape == (g.n_sites, g.n_sites)
        assert np.allclose(d, d.T)
        assert len(set(g.edges)) == len(g.edges)
