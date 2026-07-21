import numpy as np
import pytest
from graphkan_amo_nqs.models.flax_nqs import pair_feature_arrays, build_flax_model

def test_pair_feature_arrays_are_deterministic():
    pos=np.array([[0.],[1.],[2.]])
    a=pair_feature_arrays(pos,n_basis=4,family='sine')
    b=pair_feature_arrays(pos,n_basis=4,family='sine')
    assert a[2].shape==(3,4)
    assert np.array_equal(a[2],b[2])

def test_flax_model_has_clear_optional_dependency_error():
    try:
        import flax  # noqa
    except Exception:
        with pytest.raises(Exception):
            build_flax_model('product',np.array([[0.],[1.]]))
