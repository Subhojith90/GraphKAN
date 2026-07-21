import numpy as np
from graphkan_amo_nqs.stats.bootstrap import bootstrap_ci
from graphkan_amo_nqs.stats.paired_tests import wilcoxon_or_permutation

def test_stats():
    med,lo,hi=bootstrap_ci([1,2,3], n_resamples=20)
    assert lo <= med <= hi
    assert "p_value" in wilcoxon_or_permutation(np.array([1,2,3]))
