from pathlib import Path

def test_repaired_stage4c_script_is_shipped():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "summarize_stage4c_repaired.py"
    assert script.exists()
    text = script.read_text()
    for token in ("n_sites", "delta", "c6", "seed"):
        assert token in text
