import numpy as np
from graphkan_amo_nqs.interpretability.stability import curve_alignment, stability_score

def test_curve_alignment():
    curves=[np.arange(5), np.arange(5)+1]
    assert curve_alignment(curves) > 0.99
    assert stability_score(curves) > 0
