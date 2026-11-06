from stage4i_protocol import read_config


def test_stage4i_threshold_semantics_are_prospective():
    cfg = read_config()
    gate = cfg["counterfactual"]["acceptance"]
    assert gate["energy_safe_seeds_required"] == 5
    assert gate["correlation_positive_seeds_required"] == 4
    assert gate["maximum_density_degradation"] == 1e-4
    assert gate["negative_closure_correlation_worse_seeds"] == 4
    assert cfg["conditional_g3"]["status"] == "NOT_LAUNCHED"
