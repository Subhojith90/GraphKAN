#!/usr/bin/env python3
"""Verify periodic endpoint aliases on identical parameters and all N=10 states."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from graphkan_amo_nqs.geometry.chains import make_periodic_chain_geometry
from graphkan_amo_nqs.geometry.distances import geometry_distance_matrix
from graphkan_amo_nqs.models.stage4g_flax_nqs import build_flax_model


KINDS = {
    "rbf_radial_b32": "kan_jastrow",
    "endpoint_full_b32": "kan_endpoint_jastrow",
    "endpoint_hybrid_b32": "kan_endpoint_hybrid_jastrow",
}

jax.config.update("jax_enable_x64", True)


def validate(n_sites: int = 10) -> dict:
    n_sites = int(n_sites)
    geometry = make_periodic_chain_geometry(n_sites)
    distances = geometry_distance_matrix(geometry)
    states = np.asarray(
        [[1.0 if (value >> site) & 1 else -1.0 for site in range(n_sites)] for value in range(1 << n_sites)],
        dtype=np.float64,
    )
    rows = []
    for delta, c6 in ((1.25, 3.0), (1.50, 4.0), (2.00, 5.0)):
        r_scale = float(c6 ** (1.0 / 6.0))
        for seed in range(5):
            models = {
                label: build_flax_model(
                    kind, geometry.positions, n_basis=32, hidden=1, r_scale=r_scale,
                    omega=1.0, basis_family="rbf", distance_matrix=distances,
                    endpoint_features_enabled=False,
                )
                for label, kind in KINDS.items()
            }
            variables = {
                label: model.init(jax.random.PRNGKey(seed), jnp.asarray(states[:1]))
                for label, model in models.items()
            }
            radial_leaves, radial_tree = jax.tree_util.tree_flatten(variables["rbf_radial_b32"])
            radial_output = np.asarray(models["rbf_radial_b32"].apply(variables["rbf_radial_b32"], states))
            for label in ("endpoint_full_b32", "endpoint_hybrid_b32"):
                leaves, tree = jax.tree_util.tree_flatten(variables[label])
                parameter_identity = tree == radial_tree and len(leaves) == len(radial_leaves) and all(
                    np.array_equal(np.asarray(left), np.asarray(right))
                    for left, right in zip(radial_leaves, leaves)
                )
                output = np.asarray(models[label].apply(variables[label], states))
                maximum = float(np.max(np.abs(radial_output - output)))
                rows.append({
                    "delta": delta, "c6": c6, "seed": seed, "model": label,
                    "parameter_tree_identical": parameter_identity,
                    "states_compared": int(len(states)),
                    "maximum_log_amplitude_difference": maximum,
                    "pass": bool(parameter_identity and maximum == 0.0),
                })
    errors = [] if len(rows) == 30 and all(row["pass"] for row in rows) else [
        "periodic aliases are not exactly identical on common parameters and all N=10 states"
    ]
    return {
        "schema_version": "stage4g-g2c-periodic-alias-v1",
        "validation": "PASS" if not errors else "FAIL",
        "blocking_errors": errors,
        "comparisons": len(rows),
        "passes": sum(bool(row["pass"]) for row in rows),
        "n_sites": n_sites,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--n-sites", type=int, default=10)
    args = parser.parse_args()
    result = validate(args.n_sites)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["validation"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
