#!/usr/bin/env python3
"""Record and enforce the exact independent replay software/host identity."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import platform
import sys
from importlib import metadata
from pathlib import Path


EXPECTED = {
    "jax": "0.7.1", "jaxlib": "0.7.1", "flax": "0.12.0", "optax": "0.2.8",
    "netket": "3.22.3", "numpy": "1.26.4", "scipy": "1.13.1", "pandas": "2.2.2",
    "matplotlib": "3.9.2", "PyYAML": "6.0.1", "pytest": "7.4.4",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    installed = {name: metadata.version(name) for name in EXPECTED}
    mismatches = {name: {"expected": EXPECTED[name], "installed": value} for name, value in installed.items() if value != EXPECTED[name]}
    import jax
    import numpy as np
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        np.show_config()
    output = {
        "schema_version": "stage4fr1b-environment-attestation-v1",
        "validation": "PASS" if not mismatches else "FAIL",
        "expected_packages": EXPECTED,
        "installed_packages": installed,
        "package_mismatches": mismatches,
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "numpy_blas_lapack": buffer.getvalue(),
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

