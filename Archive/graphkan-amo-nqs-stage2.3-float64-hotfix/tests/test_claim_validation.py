from graphkan_amo_nqs.reporting.claim_audit import validate_claims

def test_claim_validation():
    df=validate_claims()
    assert "quantum advantage" in " ".join(df["claim"])
