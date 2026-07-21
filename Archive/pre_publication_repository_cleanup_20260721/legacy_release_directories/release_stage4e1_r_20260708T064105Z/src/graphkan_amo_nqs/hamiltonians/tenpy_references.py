from __future__ import annotations

from graphkan_amo_nqs.exceptions import DependencyUnavailable


def require_tenpy():
    try:
        import tenpy  # type: ignore
        return tenpy
    except Exception as exc:
        raise DependencyUnavailable("TeNPy is unavailable. Install with `pip install '.[tensor]'` or follow TeNPy docs.") from exc


def tenpy_dmrg_available() -> bool:
    try:
        require_tenpy(); return True
    except DependencyUnavailable:
        return False
