from graphkan_amo_nqs.training.nqs_preflight import preflight_dict

def test_preflight_schema_is_stable():
    report=preflight_dict(require_netket=False)
    assert {'ok','python','platform','packages','jax_devices','errors','warnings'} <= set(report)
    assert isinstance(report['packages'], dict)
