from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
from pathlib import Path
from typing import Any

from .config import canonical_json


def sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def hash_dict(data: Any) -> str:
    return sha256_bytes(canonical_json(data).encode("utf-8"))


def hash_file(path: str | Path) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_text(path: str | Path, text: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=p.name, suffix=".tmp", dir=str(p.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, p)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_write_json(path: str | Path, data: Any, indent: int = 2) -> None:
    atomic_write_text(path, json.dumps(data, indent=indent, sort_keys=True, default=str) + "\n")


def atomic_write_dataframe(path: str | Path, df) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=p.name, suffix=".tmp", dir=str(p.parent))
    os.close(fd)
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, p)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def run_manifest(config: dict[str, Any], profile: str, source_files: list[str | Path] | None = None) -> dict[str, Any]:
    source_files = source_files or []
    return {
        "profile": profile,
        "config_hash": hash_dict(config),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "source_hashes": {str(p): hash_file(p) for p in source_files if Path(p).exists()},
    }
