from graphkan_amo_nqs.geometry.splitters import detect_split_leakage

def test_split_leakage_detects_duplicates():
    recs=[{"distance_matrix_hash":"x","split":"train"},{"distance_matrix_hash":"x","split":"test"}]
    assert detect_split_leakage(recs) == ["x"]
