from graphkan_amo_nqs.manifests import hash_dict

def test_hash_dict_stable():
    assert hash_dict({"b":2,"a":1}) == hash_dict({"a":1,"b":2})
