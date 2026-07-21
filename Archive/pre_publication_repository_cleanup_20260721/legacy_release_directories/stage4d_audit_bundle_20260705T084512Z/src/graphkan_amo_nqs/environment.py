from __future__ import annotations

import importlib.util
import platform
import sys
from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import pandas as pd
import scipy


@dataclass(frozen=True)
class DependencyStatus:
    name: str
    available: bool
    version: str | None
    required_for: str


def _version(name: str) -> str | None:
    spec = importlib.util.find_spec(name)
    if spec is None:
        return None
    try:
        module = __import__(name)
        return str(getattr(module, "__version__", "unknown"))
    except Exception:
        return "import-error"


def dependency_status() -> list[DependencyStatus]:
    deps = [
        ("jax", "JAX array/autodiff optional NetKet-compatible development"),
        ("flax", "Flax model definitions for production VMC"),
        ("optax", "JAX optimisers"),
        ("netket", "production VMC"),
        ("quspin", "optional ED cross-checks"),
        ("tenpy", "optional 1D DMRG references"),
        ("qiskit", "optional IBM toy circuit sanity"),
        ("qiskit_ibm_runtime", "optional IBM runtime"),
        ("torch", "optional legacy prototype fallback"),
    ]
    out = []
    for name, purpose in deps:
        v = _version(name)
        out.append(DependencyStatus(name, v is not None and v != "import-error", v, purpose))
    return out


def collect_environment() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "optional_dependencies": [asdict(d) for d in dependency_status()],
    }
