import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_current_root_manifest_has_no_stale_stage():
    manifest = json.loads((ROOT / "release_manifest.json").read_text())
    assert manifest["authoritative_stage"] == "Stage 4F-R.1b"
    assert manifest["campaign"]["validation"] == "PASS"
    assert manifest["replay"]["validation"] == "PASS"
    assert manifest["immutable_release_tag"] == "stage4fr1b-final-20260723"


def test_historical_build_is_explicitly_superseded():
    build = json.loads(
        (ROOT / "releases/GraphKAN_Stage4FR1b_Hardened_20260722.build.json").read_text()
    )
    assert build["record_status"] == "SUPERSEDED_HISTORICAL_BUILD_RECORD"
    assert build["superseded_by"] == "GraphKAN_Stage4FR1b_Composite_20260723.build.json"


def test_workflow_actions_are_immutable_shas():
    workflow = (ROOT / ".github/workflows/stage4fr1b-independent-replay.yml").read_text()
    for mutable in ("actions/checkout@v4", "actions/setup-python@v5", "actions/upload-artifact@v4"):
        assert mutable not in workflow


def test_no_tracked_cache_files():
    forbidden = {"__pycache__", ".pytest_cache", ".cache"}
    tracked = [
        line
        for line in (
            __import__("subprocess")
            .run(
                ["git", "ls-files"],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )
            .stdout.splitlines()
        )
        if forbidden.intersection(Path(line).parts) or line.endswith((".pyc", ".pyo"))
    ]
    assert tracked == []
