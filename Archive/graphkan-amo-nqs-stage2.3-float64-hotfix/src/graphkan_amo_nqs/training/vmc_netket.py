from __future__ import annotations

from graphkan_amo_nqs.exceptions import DependencyUnavailable


def netket_available() -> bool:
    try:
        import netket  # noqa: F401
        import flax  # noqa: F401
        return True
    except Exception:
        return False


def run_netket_vmc(*args, **kwargs):
    if not netket_available():
        raise DependencyUnavailable("NetKet/Flax are not installed; VMC production profile was not executed.")
    raise NotImplementedError("NetKet VMC route is specified but not executed by the CPU pilot artifact.")
