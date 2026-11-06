from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from finalize_stage4g_local import discover, safe_extract


def make_zip(path: Path, member: str = "marker.txt") -> None:
    with zipfile.ZipFile(path, "w") as handle:
        handle.writestr(member, "ok")


def test_local_finalizer_discovers_exact_actions_artifact_set(tmp_path: Path):
    make_zip(tmp_path / "stage4g-final-preflight.zip")
    make_zip(tmp_path / "stage4g-final-control-selection.zip")
    for boundary in ("open", "periodic"):
        for index in range(6):
            make_zip(tmp_path / f"stage4g-confirmation-{boundary}-model{index}-final.zip")
    preflight, selection, confirmation = discover(tmp_path)
    assert preflight.name == "stage4g-final-preflight.zip"
    assert selection.name == "stage4g-final-control-selection.zip"
    assert len(confirmation) == 12


def test_local_finalizer_rejects_unsafe_zip_member(tmp_path: Path):
    archive = tmp_path / "unsafe.zip"
    make_zip(archive, "../escape.txt")
    with pytest.raises(ValueError, match="unsafe member"):
        safe_extract(archive, tmp_path / "output")
