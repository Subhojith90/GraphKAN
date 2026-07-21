from graphkan_amo_nqs.baselines.fairness import fairness_matrix

def test_fairness_matrix():
    df=fairness_matrix(["mlp_pair","kan_jastrow"])
    assert df["same_instances"].all()
