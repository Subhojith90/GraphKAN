from __future__ import annotations

import importlib
import importlib.metadata
import platform
import sys
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class PreflightReport:
    ok: bool
    python: str
    platform: str
    packages: dict[str, str | None]
    jax_devices: list[str]
    errors: list[str]
    warnings: list[str]


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except Exception:
        return None


def run_preflight(require_netket: bool = True) -> PreflightReport:
    packages = {name: _version(name) for name in ("jax", "jaxlib", "flax", "optax", "netket", "numpy", "scipy")}
    errors: list[str] = []
    warnings: list[str] = []
    devices: list[str] = []
    if sys.version_info[:2] != (3, 11):
        warnings.append(f"Recommended interpreter is Python 3.11; detected {sys.version.split()[0]}.")
    if require_netket:
        missing = [name for name in ("jax", "flax", "optax", "netket") if packages[name] is None]
        if missing:
            errors.append("Missing NQS dependencies: " + ", ".join(missing) + ". Install with `pip install -e '.[nqs]'`.")
    try:
        import jax
        devices = [str(d) for d in jax.devices()]
        if not devices:
            errors.append("JAX reported no executable devices.")
        elif all("cpu" in d.lower() for d in devices):
            warnings.append("CPU-only JAX detected. Use the smoke profile before launching the registered VMC benchmark.")
    except Exception as exc:
        if require_netket:
            errors.append(f"JAX import/device discovery failed: {type(exc).__name__}: {exc}")
    return PreflightReport(
        ok=not errors,
        python=sys.version.split()[0],
        platform=platform.platform(),
        packages=packages,
        jax_devices=devices,
        errors=errors,
        warnings=warnings,
    )


def preflight_dict(require_netket: bool = True) -> dict[str, Any]:
    return asdict(run_preflight(require_netket=require_netket))
