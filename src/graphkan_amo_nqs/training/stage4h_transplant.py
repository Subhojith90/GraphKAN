"""Exact radial-to-expanded-model parameter transplant for Stage 4H.

The Stage 4G checkpoints store JAX tree leaves in deterministic key order.  The
radial model has exactly two leaves: ``kan_coeff`` and ``site_bias``.  Stage 4H
copies those arrays by semantic name and sets every newly introduced feature
coefficient to literal zero.  No randomly initialised expanded parameter is
allowed to survive the transplant.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


EXPANSION_KEYS = {
    "radial_continuation": None,
    "endpoint_full_warmstart": "endpoint_coeff",
    "endpoint_hybrid_warmstart": "endpoint_coeff",
    "nongeometric_warmstart": "control_coeff",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_stage4g_radial_checkpoint(path: Path) -> dict[str, np.ndarray]:
    """Load and schema-check a Stage 4G radial checkpoint."""
    sidecar = path.with_suffix(".json")
    if not path.is_file() or not sidecar.is_file():
        raise FileNotFoundError(f"checkpoint or sidecar missing: {path}")
    metadata = json.loads(sidecar.read_text())
    if metadata.get("model_label") != "rbf_radial_b32":
        raise ValueError(f"not a Stage 4G radial checkpoint: {path}")
    with np.load(path, allow_pickle=False) as archive:
        names = sorted(archive.files)
        arrays = [np.asarray(archive[name]) for name in names]
    if names != ["leaf_0000", "leaf_0001"] or [x.shape for x in arrays] != [
        (32,),
        (12,),
    ]:
        raise ValueError(f"unexpected Stage 4G radial leaf schema: {path}")
    return {"kan_coeff": arrays[0], "site_bias": arrays[1]}


def transplant_parameters(
    template: Mapping[str, Any], radial: Mapping[str, np.ndarray], arm: str
) -> Any:
    """Return a target tree containing radial parameters plus zero extras."""
    if arm not in EXPANSION_KEYS:
        raise ValueError(f"unsupported Stage 4H arm: {arm}")
    expected = {"kan_coeff", "site_bias"}
    extra = EXPANSION_KEYS[arm]
    if extra is not None:
        expected.add(extra)
    if set(template) != expected:
        raise ValueError(
            f"target parameter schema for {arm} is {sorted(template)}, expected {sorted(expected)}"
        )
    if set(radial) != {"kan_coeff", "site_bias"}:
        raise ValueError(
            "radial parameter mapping must contain kan_coeff and site_bias only"
        )

    converted: dict[str, Any] = {}
    for name, target in template.items():
        source = (
            np.zeros(np.asarray(target).shape, dtype=np.asarray(target).dtype)
            if name == extra
            else np.asarray(radial[name])
        )
        if source.shape != np.asarray(target).shape:
            raise ValueError(
                f"shape mismatch for {name}: {source.shape} != {np.asarray(target).shape}"
            )
        try:
            import jax.numpy as jnp

            converted[name] = jnp.asarray(source, dtype=getattr(target, "dtype", None))
        except Exception:
            converted[name] = source.copy()

    try:
        from flax.core import FrozenDict, freeze

        return freeze(converted) if isinstance(template, FrozenDict) else converted
    except Exception:
        return converted


def full_basis_attestation(
    radial_model: Any,
    radial_parameters: Mapping[str, Any],
    target_model: Any,
    target_parameters: Mapping[str, Any],
    sigma_states: np.ndarray,
    *,
    tolerance: float = 1.0e-12,
) -> dict[str, Any]:
    """Compare raw log amplitudes on every supplied basis state."""
    import jax.numpy as jnp

    sigma = jnp.asarray(sigma_states, dtype=jnp.float64)
    radial_logs = np.asarray(
        radial_model.apply({"params": radial_parameters}, sigma), dtype=np.complex128
    )
    target_logs = np.asarray(
        target_model.apply({"params": target_parameters}, sigma), dtype=np.complex128
    )
    differences = np.abs(radial_logs - target_logs)
    maximum = float(np.max(differences))
    return {
        "schema_version": "stage4h-full-basis-transplant-v1",
        "states_compared": int(len(sigma_states)),
        "maximum_log_amplitude_difference": maximum,
        "tolerance": float(tolerance),
        "validation": "PASS" if maximum <= tolerance else "FAIL",
    }


def find_radial_checkpoint(root: Path, *, delta: float, c6: float, seed: int) -> Path:
    """Locate one Stage 4G N=12 open radial checkpoint by sidecar semantics."""
    candidates: list[Path] = []
    for sidecar in root.rglob("*.json"):
        try:
            metadata = json.loads(sidecar.read_text())
        except Exception:
            continue
        run_id = str(metadata.get("run_id", ""))
        if (
            metadata.get("model_label") == "rbf_radial_b32"
            and int(metadata.get("seed", -1)) == int(seed)
            and "N12" in run_id
            and "confirmation_open" in run_id
            and f"D{float(delta):g}_C6{float(c6):g}_" in run_id
        ):
            checkpoint = sidecar.with_suffix(".npz")
            if checkpoint.is_file():
                candidates.append(checkpoint)
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"expected one Stage 4G radial checkpoint for delta={delta}, c6={c6}, seed={seed}; found {len(candidates)}"
        )
    return candidates[0]
