from stage4i_protocol import read_config, roster, validate_config


def test_stage4i_protocol_and_roster_are_frozen():
    cfg = read_config()
    result = validate_config(cfg)
    assert result["validation"] == "PASS"
    rows = roster(cfg)
    assert len(rows) == 15
    assert len({row["run_id"] for row in rows}) == 15
    assert {row["shard"] for row in rows} == {
        "stage4i_radial",
        "stage4i_endpoint_full_raw",
        "stage4i_endpoint_full_orthogonal",
    }
    assert {
        (row["n_sites"], row["boundary"], row["delta"], row["c6"]) for row in rows
    } == {(12, "open", 2.0, 5.0)}
