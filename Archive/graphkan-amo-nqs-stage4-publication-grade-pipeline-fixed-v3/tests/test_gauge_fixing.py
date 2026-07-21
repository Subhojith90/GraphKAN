import numpy as np
from graphkan_amo_nqs.models.gauge import gauge_fix_curve

def test_gauge_fix_zero_mean_unit():
    r=np.linspace(1,3,10); v=np.linspace(0,2,10)+5
    g=gauge_fix_curve(r,v,align_sign=False)
    assert abs(g.mean()) < 1e-10
    assert abs((g**2).mean() - 1) < 1e-10
