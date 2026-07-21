from __future__ import annotations

from graphkan_amo_nqs.exceptions import DependencyUnavailable


def require_quspin():
    try:
        import quspin  # type: ignore
        return quspin
    except Exception as exc:
        raise DependencyUnavailable("QuSpin is unavailable. Install with `pip install '.[ed]'` if supported on your platform.") from exc


def quspin_crosscheck_available() -> bool:
    try:
        require_quspin(); return True
    except DependencyUnavailable:
        return False
