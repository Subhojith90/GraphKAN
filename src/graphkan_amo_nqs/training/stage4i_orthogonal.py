"""Span-preserving endpoint orthogonalisation for Stage 4I.

The projection is defined on the complete Boolean basis, not on sampled VMC
states.  Endpoint sufficient statistics are residualised against the one-body
and radial sufficient-statistic design.  This changes coordinates and
conditioning, while preserving the exact hypothesis-space column span.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

from graphkan_amo_nqs.models.stage4g_flax_nqs import pair_feature_arrays


def boolean_states(n_sites: int) -> np.ndarray:
    integers = np.arange(1 << int(n_sites), dtype=np.uint64)[:, None]
    return ((integers >> np.arange(n_sites, dtype=np.uint64)) & 1).astype(float)


def endpoint_pair_features(
    n_sites: int, pair_i: np.ndarray, pair_j: np.ndarray
) -> np.ndarray:
    distance = np.minimum(np.arange(n_sites), n_sites - 1 - np.arange(n_sites)).astype(
        float
    )
    distance /= max(float(distance.max()), 1.0)
    di, dj = distance[pair_i], distance[pair_j]
    return np.stack([di + dj, np.abs(di - dj), di * dj], axis=1)


def orthogonalisation(
    n_sites: int, r_scale: float, *, n_basis: int = 32
) -> dict[str, Any]:
    states = boolean_states(n_sites)
    pair_i, pair_j, radial = pair_feature_arrays(
        np.arange(n_sites, dtype=float)[:, None],
        n_basis=n_basis,
        family="rbf",
        r_scale=float(r_scale),
    )
    pair_occ = states[:, pair_i] * states[:, pair_j]
    endpoint = endpoint_pair_features(n_sites, pair_i, pair_j)
    base = np.concatenate([states, pair_occ @ radial], axis=1)
    extra = pair_occ @ endpoint
    projection = np.linalg.lstsq(base, extra, rcond=None)[0]
    residual = extra - base @ projection
    raw = np.concatenate([base, extra], axis=1)
    orthogonal = np.concatenate([base, residual], axis=1)
    raw_rank = int(np.linalg.matrix_rank(raw))
    orthogonal_rank = int(np.linalg.matrix_rank(orthogonal))
    projector_error = float(
        np.linalg.norm(base.T @ residual)
        / max(np.linalg.norm(base) * np.linalg.norm(residual), 1e-300)
    )
    raw_projector = raw @ np.linalg.pinv(raw)
    orthogonal_projector = orthogonal @ np.linalg.pinv(orthogonal)
    span_error = float(np.max(np.abs(raw_projector - orthogonal_projector)))

    def condition(x: np.ndarray) -> float:
        singular = np.linalg.svd(x, compute_uv=False)
        tolerance = max(x.shape) * np.finfo(float).eps * singular[0]
        nonzero = singular[singular > tolerance]
        return float(nonzero[0] / nonzero[-1])

    arrays = {
        "one_body_projection": projection[:n_sites],
        "radial_projection": projection[n_sites:],
    }
    digest_payload = {
        name: np.asarray(value).round(16).tolist() for name, value in arrays.items()
    }
    return {
        **arrays,
        "schema_version": "stage4i-orthogonalisation-v1",
        "n_sites": int(n_sites),
        "n_basis": int(n_basis),
        "r_scale": float(r_scale),
        "raw_rank": raw_rank,
        "orthogonal_rank": orthogonal_rank,
        "orthogonality_error": projector_error,
        "column_space_projector_error": span_error,
        "raw_condition_number": condition(raw),
        "orthogonal_condition_number": condition(orthogonal),
        "projection_sha256": hashlib.sha256(
            (
                json.dumps(digest_payload, sort_keys=True, separators=(",", ":")) + "\n"
            ).encode()
        ).hexdigest(),
        "validation": "PASS"
        if raw_rank == orthogonal_rank
        and span_error <= 1e-10
        and projector_error <= 1e-12
        else "FAIL",
    }


def model_projection(result: dict[str, Any]) -> dict[str, np.ndarray]:
    if result.get("validation") != "PASS":
        raise ValueError("orthogonalisation attestation did not pass")
    return {
        "one_body_projection": np.asarray(result["one_body_projection"], dtype=float),
        "radial_projection": np.asarray(result["radial_projection"], dtype=float),
    }
