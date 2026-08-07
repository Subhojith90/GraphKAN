#!/usr/bin/env python3
"""Run all 60 no-VMC Stage 4H transplant equality attestations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from _common import params_from_values
from graphkan_amo_nqs.geometry import make_chain_geometry
from graphkan_amo_nqs.geometry.distances import geometry_distance_matrix
from graphkan_amo_nqs.hamiltonians.basis import enumerate_binary_states
from graphkan_amo_nqs.hamiltonians.netket_builders import build_netket_rydberg_operator
from graphkan_amo_nqs.hamiltonians.rydberg import build_dense_rydberg
from graphkan_amo_nqs.models.stage4g_flax_nqs import build_flax_model
from graphkan_amo_nqs.training.stage4h_transplant import (
    find_radial_checkpoint,
    full_basis_attestation,
    load_stage4g_radial_checkpoint,
    sha256_file,
    transplant_parameters,
)
from run_stage4g import physical_sigma_from_binary
from run_stage4h_warmstart import exact_observables
from stage4h_protocol import DEFAULT_CONFIG, read_config, validate_config


def run(cfg: dict, stage4g_root: Path, output: Path) -> dict:
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    output.mkdir(parents=True, exist_ok=True)
    geometry = make_chain_geometry(12)
    distances = geometry_distance_matrix(geometry)
    states = enumerate_binary_states(12)
    sigma = physical_sigma_from_binary(states, False)
    records = []
    for cell in cfg["physics"]["cells"]:
        params = params_from_values(
            cfg["physics"]["omega"], cell["delta"], cell["c6"], "parity"
        )
        _, hamiltonian = build_netket_rydberg_operator(
            geometry.positions,
            params,
            inverted_ordering=False,
            distance_matrix=distances,
        )
        dense_h = np.asarray(
            build_dense_rydberg(geometry.positions, params, distance_matrix=distances),
            dtype=np.complex128,
        )
        matrix_diff = float(
            np.max(
                np.abs(
                    dense_h - np.asarray(hamiltonian.to_dense(), dtype=np.complex128)
                )
            )
        )
        if matrix_diff > 1e-10:
            raise RuntimeError("dense/NetKet operator mismatch")
        radial_model = build_flax_model(
            "kan_jastrow",
            geometry.positions,
            n_basis=32,
            hidden=1,
            r_scale=params.r_omega,
            omega=params.omega,
            basis_family="rbf",
            distance_matrix=distances,
        )
        for seed in cfg["physics"]["seeds"]:
            source = find_radial_checkpoint(
                stage4g_root, delta=cell["delta"], c6=cell["c6"], seed=seed
            )
            radial_np = load_stage4g_radial_checkpoint(source)
            radial = {
                name: jnp.asarray(value, dtype=jnp.float64)
                for name, value in radial_np.items()
            }
            _, _, radial_e, radial_n, radial_c = exact_observables(
                radial_model, radial, sigma, states, dense_h
            )
            for arm, spec in cfg["arms"].items():
                model = build_flax_model(
                    spec["kind"],
                    geometry.positions,
                    n_basis=32,
                    hidden=1,
                    r_scale=params.r_omega,
                    omega=params.omega,
                    basis_family="rbf",
                    distance_matrix=distances,
                )
                template = model.init(
                    jax.random.PRNGKey(int(seed)), jnp.ones((1, 12), dtype=jnp.float64)
                )["params"]
                target = transplant_parameters(template, radial_np, arm)
                result = full_basis_attestation(
                    radial_model,
                    radial,
                    model,
                    target,
                    sigma,
                    tolerance=cfg["acceptance"][
                        "transplant_max_log_amplitude_difference"
                    ],
                )
                _, _, energy, density, corr = exact_observables(
                    model, target, sigma, states, dense_h
                )
                result.update(
                    {
                        "run_id": f"stage4h__{cell['tag']}__{arm}__S{seed}",
                        "cell_tag": cell["tag"],
                        "delta": cell["delta"],
                        "c6": cell["c6"],
                        "seed": seed,
                        "arm": arm,
                        "source_checkpoint": source.name,
                        "source_checkpoint_sha256": sha256_file(source),
                        "exact_energy_difference": abs(energy - radial_e),
                        "maximum_density_difference": float(
                            np.max(np.abs(density - radial_n))
                        ),
                        "maximum_correlation_difference": float(
                            np.max(np.abs(corr - radial_c))
                        ),
                    }
                )
                limits = cfg["acceptance"]
                if not (
                    result["exact_energy_difference"]
                    <= limits["transplant_max_exact_energy_difference"]
                    and result["maximum_density_difference"]
                    <= limits["transplant_max_density_difference"]
                    and result["maximum_correlation_difference"]
                    <= limits["transplant_max_correlation_difference"]
                ):
                    result["validation"] = "FAIL"
                records.append(result)
    frame = pd.DataFrame(records)
    frame.to_csv(output / "transplant_preflight_attestations.csv", index=False)
    passes = int((frame.validation == "PASS").sum())
    summary = {
        "schema_version": "stage4h-transplant-preflight-v1",
        "validation": "PASS" if passes == 60 else "FAIL",
        "attestations": len(frame),
        "passes": passes,
        "expected": 60,
        "maximum_log_amplitude_difference": float(
            frame.maximum_log_amplitude_difference.max()
        ),
        "maximum_exact_energy_difference": float(frame.exact_energy_difference.max()),
        "maximum_density_difference": float(frame.maximum_density_difference.max()),
        "maximum_correlation_difference": float(
            frame.maximum_correlation_difference.max()
        ),
    }
    (output / "transplant_preflight_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage4g-root", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("results_stage4h/environment")
    )
    args = parser.parse_args()
    cfg = read_config(args.config)
    if validate_config(cfg)["validation"] != "PASS":
        raise SystemExit("invalid Stage 4H config")
    result = run(cfg, args.stage4g_root, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["validation"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
