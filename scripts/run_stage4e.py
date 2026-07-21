#!/usr/bin/env python3
"""Stage 4E adjacent hard-regime patch runner with explicit dual health gates and checkpoints.

This script intentionally does not use the legacy Stage 3F raw filenames.  It writes
native Stage 4E tables and records sampled and exact all-state health separately.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from _common import build_geometries_from_config, hash_dict, load_config, parameter_grid
from graphkan_amo_nqs.hamiltonians.basis import enumerate_binary_states
from graphkan_amo_nqs.hamiltonians.netket_builders import build_netket_rydberg_operator
from graphkan_amo_nqs.hamiltonians.rydberg import build_dense_rydberg, instance_id
from graphkan_amo_nqs.models.flax_nqs import build_flax_model, resolve_basis_family
from graphkan_amo_nqs.geometry.distances import geometry_distance_matrix
from graphkan_amo_nqs.observables.correlations import connected_correlations_from_state, correlation_mae
from graphkan_amo_nqs.observables.density import density_from_state
from graphkan_amo_nqs.training.nqs_preflight import preflight_dict


def append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, mode="a" if path.exists() else "w", header=not path.exists(), index=False)


def exact_reference(positions: np.ndarray, params: Any, *, distance_matrix: np.ndarray | None = None):
    h = np.asarray(build_dense_rydberg(positions, params, distance_matrix=distance_matrix), dtype=np.complex128)
    vals, vecs = np.linalg.eigh(h)
    return float(vals[0]), h, vecs[:, 0]


def operator_check(positions: np.ndarray, params: Any, netket_h: Any, *, distance_matrix: np.ndarray | None = None):
    e0, dense_h, psi0 = exact_reference(positions, params, distance_matrix=distance_matrix)
    netket_dense = np.asarray(netket_h.to_dense(), dtype=np.complex128)
    matrix_diff = float(np.max(np.abs(dense_h - netket_dense)))
    spectrum_diff = float(np.max(np.abs(np.linalg.eigvalsh(dense_h) - np.linalg.eigvalsh(netket_dense))))
    return e0, dense_h, psi0, matrix_diff, spectrum_diff


def physical_sigma_from_binary(states: np.ndarray, inverted_ordering: bool) -> np.ndarray:
    return 2.0 * states - 1.0 if inverted_ordering else 1.0 - 2.0 * states


def exact_model_observables(vstate: Any, states: np.ndarray, dense_h: np.ndarray, inverted_ordering: bool):
    import jax.numpy as jnp

    sigma = physical_sigma_from_binary(states, inverted_ordering)
    logs = np.asarray(
        vstate.model.apply({"params": vstate.parameters}, jnp.asarray(sigma, dtype=jnp.float64)),
        dtype=np.complex128,
    )
    logs = logs - np.max(logs.real)
    psi = np.exp(logs)
    psi = psi / np.linalg.norm(psi)
    density = density_from_state(psi, states)
    corr = connected_correlations_from_state(psi, states)
    energy = float(np.real(np.vdot(psi, dense_h @ psi)))
    return psi, energy, density, corr


def sample_observables(vstate: Any, n_sites: int, inverted_ordering: bool):
    samples = np.asarray(vstate.samples).reshape((-1, n_sites))
    occupation = 0.5 * (samples + 1.0) if inverted_ordering else 0.5 * (1.0 - samples)
    density = occupation.mean(axis=0)
    corr = (occupation.T @ occupation) / len(occupation) - np.outer(density, density)
    unique = np.unique(samples, axis=0).shape[0]
    return density, corr, {
        "sample_count_observed": int(len(samples)),
        "unique_sampled_states": int(unique),
        "unique_state_fraction": float(unique / max(1, len(samples))),
    }


def classify_sample(energy_mean: float, energy_stderr: float, e0: float, density_mae: float, corr_mae: float, diversity: float, gate: dict[str, Any]):
    flags: list[str] = []
    if not (np.isfinite(energy_mean) and np.isfinite(energy_stderr)):
        flags.append("NONFINITE_ESTIMATOR")
    if energy_stderr <= gate.get("min_energy_error", 1e-10):
        flags.append("NEAR_ZERO_ERROR")
    if abs(energy_mean - e0) > gate.get("max_abs_energy_error_to_ed", 0.5):
        flags.append("LARGE_ED_ERROR")
    if density_mae > gate.get("max_density_mae_to_ed", 0.08):
        flags.append("LARGE_DENSITY_ERROR")
    if corr_mae > gate.get("max_connected_correlation_mae_to_ed", 0.08):
        flags.append("LARGE_CORRELATION_ERROR")
    if diversity < gate.get("min_unique_state_fraction", 0.005):
        flags.append("LOW_SAMPLE_DIVERSITY")
    return ("VALID" if not flags else "FLAGGED"), flags


def classify_exact(exact_energy: float, e0: float, density_mae: float, corr_mae: float, gate: dict[str, Any]):
    flags: list[str] = []
    if not np.isfinite(exact_energy):
        flags.append("NONFINITE_EXACT_OBSERVABLE")
    if abs(exact_energy - e0) > gate.get("max_abs_energy_error_to_ed", 0.5):
        flags.append("LARGE_EXACT_ED_ERROR")
    if density_mae > gate.get("max_density_mae_to_ed", 0.08):
        flags.append("LARGE_EXACT_DENSITY_ERROR")
    if corr_mae > gate.get("max_connected_correlation_mae_to_ed", 0.08):
        flags.append("LARGE_EXACT_CORRELATION_ERROR")
    return ("VALID" if not flags else "FLAGGED"), flags


def n_parameters(model: Any, n_sites: int, seed: int) -> int:
    import jax
    import jax.numpy as jnp

    params = model.init(jax.random.PRNGKey(seed), jnp.ones((1, n_sites), dtype=jnp.float64))
    return int(sum(np.asarray(x).size for x in jax.tree_util.tree_leaves(params["params"])))


def save_checkpoint(path: Path, vstate: Any, metadata: dict[str, Any]) -> None:
    """Save a portable Flax parameter-tree snapshot as npz plus JSON metadata."""
    import jax

    leaves, treedef = jax.tree_util.tree_flatten(vstate.parameters)
    arrays = {f"leaf_{i:04d}": np.asarray(value) for i, value in enumerate(leaves)}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    sidecar = dict(metadata)
    sidecar.update({"checkpoint_format": "jax_tree_leaves_npz_v1", "leaf_count": len(leaves), "treedef_repr": repr(treedef)})
    path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--only-models", nargs="*", default=None, help="Optional exact model labels to run.")
    parser.add_argument("--only-n-sites", nargs="*", type=int, default=None, help="Optional system sizes to run.")
    parser.add_argument("--only-deltas", nargs="*", type=float, default=None, help="Optional detunings to run.")
    parser.add_argument("--only-c6", nargs="*", type=float, default=None, help="Optional C6 values to run.")
    parser.add_argument("--resume", action="store_true", help="Skip run IDs already present in the native metrics file.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = Path(cfg.get("output_root", "results_stage4e"))
    raw = root / "raw"
    env = root / "environment"
    checkpoints = root / "checkpoints"
    raw.mkdir(parents=True, exist_ok=True)
    env.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)

    metrics_path = raw / "stage4e_metrics.csv"
    history_path = raw / "stage4e_history.csv"
    site_path = raw / "stage4e_site_observables.csv"
    corr_path = raw / "stage4e_corr_observables.csv"
    if args.overwrite:
        for path in [metrics_path, history_path, site_path, corr_path]:
            if path.exists():
                path.unlink()
    existing_run_ids: set[str] = set()
    if args.resume and metrics_path.exists():
        try:
            existing_run_ids = set(pd.read_csv(metrics_path, usecols=["run_id"]).run_id.dropna().astype(str))
        except Exception:
            existing_run_ids = set()

    preflight = preflight_dict(True)
    (env / "nqs_preflight.json").write_text(json.dumps(preflight, indent=2) + "\n")
    if not preflight["ok"]:
        raise SystemExit("Preflight failed; no Stage 4E VMC run generated.")
    try:
        (env / "pip_freeze_minimal.txt").write_text(
            subprocess.check_output([sys.executable, "-m", "pip", "freeze", "--exclude-editable"], text=True)
        )
    except Exception:
        pass

    profile = cfg["profile"]
    cfg_hash = hash_dict(cfg)
    manifest = {
        "stage": "4E",
        "profile": profile,
        "schema_version": "stage4e-native-v1",
        "config_hash": cfg_hash,
        "raw_files": [p.name for p in [metrics_path, history_path, site_path, corr_path]],
        "health_definition": "dual sampled and exact all-state validity columns; headline summaries use exact validity",
    }
    (env / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    import netket as nk
    import optax

    inverted = bool(cfg.get("netket", {}).get("inverted_ordering", False))
    sampled_gate = cfg.get("sample_health_gate", cfg.get("health_gate", {}))
    exact_gate = cfg.get("exact_health_gate", cfg.get("health_gate", {}))
    training = cfg["training"]
    specs = cfg["model_specs"]
    if args.only_models:
        requested = set(args.only_models)
        missing = requested - set(specs)
        if missing:
            raise SystemExit(f"Unknown requested model labels: {sorted(missing)}")
        specs = {label: spec for label, spec in specs.items() if label in requested}

    metrics: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []

    for geometry in build_geometries_from_config(cfg):
        if args.only_n_sites and geometry.n_sites not in set(args.only_n_sites):
            continue
        for params in parameter_grid(cfg):
            if args.only_deltas and float(params.delta) not in set(args.only_deltas):
                continue
            if args.only_c6 and float(params.c6) not in set(args.only_c6):
                continue
            distance_matrix = geometry_distance_matrix(geometry)
            endpoint_features_enabled = bool((geometry.metadata or {}).get("endpoint_features_enabled", True))
            hi, hamiltonian = build_netket_rydberg_operator(geometry.positions, params, inverted_ordering=inverted, distance_matrix=distance_matrix)
            e0, dense_h, psi0, matrix_diff, spectrum_diff = operator_check(geometry.positions, params, hamiltonian, distance_matrix=distance_matrix)
            if matrix_diff > 1e-10 or spectrum_diff > 1e-10:
                raise RuntimeError("Dense-versus-NetKet operator validation failed.")
            states = enumerate_binary_states(geometry.n_sites)
            ed_density = density_from_state(psi0, states)
            ed_corr = connected_correlations_from_state(psi0, states)

            for label, spec in specs.items():
                for n_samples in spec.get("n_samples", [training["n_samples"]]):
                    for diag_shift in spec.get("diag_shifts", [training["diag_shift"]]):
                        for seed in training["seeds"]:
                            run_id = f"{profile}:{instance_id(geometry, params)}:{label}:samples{n_samples}:shift{diag_shift}:seed{seed}"
                            if run_id in existing_run_ids:
                                print("SKIP existing", run_id)
                                continue
                            started = time.perf_counter()
                            basis_family = "UNRESOLVED"
                            try:
                                basis_family = resolve_basis_family(spec["kind"], spec.get("basis_family"))
                                model = build_flax_model(
                                    spec["kind"],
                                    geometry.positions,
                                    n_basis=int(spec.get("n_basis", training["n_basis"])),
                                    hidden=int(spec.get("hidden", training["hidden"])),
                                    r_scale=params.r_omega if np.isfinite(params.r_omega) else 1.0,
                                    omega=params.omega,
                                    inverted_ordering=inverted,
                                    basis_family=basis_family,
                                    distance_matrix=distance_matrix,
                                    endpoint_features_enabled=endpoint_features_enabled,
                                )
                                sampler = nk.sampler.MetropolisLocal(hi, n_chains_per_rank=int(training["n_chains"]))
                                state = nk.vqs.MCState(
                                    sampler,
                                    model,
                                    n_samples=int(n_samples),
                                    n_discard_per_chain=int(training["n_discard_per_chain"]),
                                    seed=int(seed),
                                )
                                optimiser = optax.chain(
                                    optax.clip_by_global_norm(float(training["grad_clip_norm"])),
                                    optax.adam(float(training["learning_rate"])),
                                )
                                driver = nk.driver.VMC(
                                    hamiltonian,
                                    optimiser,
                                    variational_state=state,
                                    preconditioner=nk.optimizer.SR(diag_shift=float(diag_shift)),
                                )
                                for step in range(0, int(training["n_iter"]), int(training["chunk_size"])):
                                    chunk = min(int(training["chunk_size"]), int(training["n_iter"]) - step)
                                    driver.run(n_iter=chunk, out=None, show_progress=False)
                                    estimate = driver.energy
                                    history.append({
                                        "run_id": run_id,
                                        "profile": profile,
                                        "model_label": label,
                                        "seed": seed,
                                        "step": step + chunk,
                                        "energy_mean": float(np.real(estimate.mean)),
                                        "energy_error": float(estimate.error_of_mean),
                                        "n_samples": n_samples,
                                        "diag_shift": diag_shift,
                                    })
                                estimate = driver.energy
                                sample_density, sample_corr, sample_diag = sample_observables(state, geometry.n_sites, inverted)
                                _, exact_energy, exact_density, exact_corr = exact_model_observables(state, states, dense_h, inverted)
                                sample_density_mae = float(np.mean(np.abs(sample_density - ed_density)))
                                sample_corr_mae = float(correlation_mae(sample_corr, ed_corr))
                                exact_density_mae = float(np.mean(np.abs(exact_density - ed_density)))
                                exact_corr_mae = float(correlation_mae(exact_corr, ed_corr))
                                sample_validity, sample_flags = classify_sample(
                                    float(np.real(estimate.mean)), float(estimate.error_of_mean), e0,
                                    sample_density_mae, sample_corr_mae, sample_diag["unique_state_fraction"], sampled_gate,
                                )
                                exact_validity, exact_flags = classify_exact(
                                    exact_energy, e0, exact_density_mae, exact_corr_mae, exact_gate
                                )
                                row = {
                                    "run_id": run_id,
                                    "profile": profile,
                                    "stage": "4E",
                                    "schema_version": "stage4e-native-v1",
                                    "status": "COMPLETED",
                                    "scientific_validity": exact_validity,
                                    "scientific_validity_exact": exact_validity,
                                    "scientific_validity_sample": sample_validity,
                                    "exact_health_flags": ";".join(exact_flags),
                                    "sample_health_flags": ";".join(sample_flags),
                                    "health_flags": ";".join(exact_flags),
                                    "model_label": label,
                                    "model_kind": spec["kind"],
                                    "basis_family": basis_family,
                                    "boundary_condition": (geometry.metadata or {}).get("boundary_condition", "open"),
                                    "periodic_length": (geometry.metadata or {}).get("periodic_length", ""),
                                    "endpoint_features_enabled": endpoint_features_enabled,
                                    "seed": seed,
                                    "n_sites": geometry.n_sites,
                                    "instance_id": instance_id(geometry, params),
                                    "omega": params.omega,
                                    "delta": params.delta,
                                    "c6": params.c6,
                                    "n_samples": n_samples,
                                    "diag_shift": diag_shift,
                                    "n_basis": spec.get("n_basis", training["n_basis"]),
                                    "hidden": spec.get("hidden", training["hidden"]),
                                    "n_parameters": n_parameters(model, geometry.n_sites, seed),
                                    "energy_mean": float(np.real(estimate.mean)),
                                    "energy_error": float(estimate.error_of_mean),
                                    "abs_energy_error_to_ed": abs(float(np.real(estimate.mean)) - e0),
                                    "exact_model_energy": exact_energy,
                                    "exact_model_abs_energy_error_to_ed": abs(exact_energy - e0),
                                    "sample_density_mae_to_ed": sample_density_mae,
                                    "sample_connected_correlation_mae_to_ed": sample_corr_mae,
                                    "exact_model_density_mae_to_ed": exact_density_mae,
                                    "exact_model_connected_correlation_mae_to_ed": exact_corr_mae,
                                    "exact_ground_energy": e0,
                                    "operator_matrix_max_abs_diff": matrix_diff,
                                    "operator_spectrum_max_abs_diff": spectrum_diff,
                                    "wall_seconds": time.perf_counter() - started,
                                    "config_hash": cfg_hash,
                                    **sample_diag,
                                }
                                metrics.append(row)
                                append_rows(metrics_path, [row])

                                if bool(spec.get("save_checkpoint", True)):
                                    safe_id = run_id.replace(":", "__").replace("/", "_")
                                    save_checkpoint(checkpoints / f"{safe_id}.npz", state, {
                                        "run_id": run_id,
                                        "profile": profile,
                                        "stage": "4E",
                                        "model_label": label,
                                        "model_kind": spec["kind"],
                                        "basis_family": basis_family,
                                        "seed": int(seed),
                                        "n_parameters": row["n_parameters"],
                                        "config_hash": cfg_hash,
                                    })

                                site_rows = []
                                corr_rows = []
                                common = {k: row[k] for k in [
                                    "run_id", "profile", "stage", "schema_version", "model_label", "model_kind", "basis_family",
                                    "seed", "n_sites", "instance_id", "omega", "delta", "c6", "n_samples", "diag_shift",
                                    "scientific_validity_exact", "scientific_validity_sample",
                                ]}
                                for i in range(geometry.n_sites):
                                    site_rows.append({
                                        **common, "site": i, "ed_density": ed_density[i], "sample_density": sample_density[i],
                                        "exact_model_density": exact_density[i], "sample_abs_error": abs(sample_density[i] - ed_density[i]),
                                        "exact_model_abs_error": abs(exact_density[i] - ed_density[i]),
                                    })
                                for i in range(geometry.n_sites):
                                    for j in range(geometry.n_sites):
                                        corr_rows.append({
                                            **common, "i": i, "j": j, "ed_connected": ed_corr[i, j],
                                            "sample_connected": sample_corr[i, j], "exact_model_connected": exact_corr[i, j],
                                            "sample_abs_error": abs(sample_corr[i, j] - ed_corr[i, j]),
                                            "exact_model_abs_error": abs(exact_corr[i, j] - ed_corr[i, j]),
                                        })
                                append_rows(site_path, site_rows)
                                append_rows(corr_path, corr_rows)
                                print(
                                    f"{exact_validity.lower()} {run_id}: sample|E-E0|={row['abs_energy_error_to_ed']:.4g}, "
                                    f"exact|E-E0|={row['exact_model_abs_energy_error_to_ed']:.4g}, "
                                    f"exact_flags={row['exact_health_flags'] or 'none'}, sample_flags={row['sample_health_flags'] or 'none'}"
                                )
                            except Exception as exc:
                                failed = {
                                    "run_id": run_id, "profile": profile, "stage": "4E", "schema_version": "stage4e-native-v1",
                                    "status": "FAILED", "scientific_validity": "INVALID", "scientific_validity_exact": "INVALID",
                                    "scientific_validity_sample": "INVALID", "exact_health_flags": "EXECUTION_FAILURE",
                                    "sample_health_flags": "EXECUTION_FAILURE", "health_flags": "EXECUTION_FAILURE",
                                    "model_label": label, "model_kind": spec["kind"], "basis_family": basis_family, "seed": seed,
                                    "n_sites": geometry.n_sites, "instance_id": instance_id(geometry, params), "omega": params.omega,
                                    "delta": params.delta, "c6": params.c6, "n_samples": n_samples, "diag_shift": diag_shift,
                                    "error_type": type(exc).__name__, "error_message": str(exc), "traceback": traceback.format_exc(),
                                    "config_hash": cfg_hash, "wall_seconds": time.perf_counter() - started,
                                }
                                metrics.append(failed)
                                append_rows(metrics_path, [failed])
                                print("FAILED", run_id, type(exc).__name__, exc)

    append_rows(history_path, history)
    pd.DataFrame(metrics).to_csv(raw / "stage4e_metrics_snapshot.csv", index=False)
    exact_valid = sum(row.get("scientific_validity_exact") == "VALID" for row in metrics)
    sample_valid = sum(row.get("scientific_validity_sample") == "VALID" for row in metrics)
    print(f"Stage 4E complete: exact_valid={exact_valid}/{len(metrics)}, sample_valid={sample_valid}/{len(metrics)}")


if __name__ == "__main__":
    main()
