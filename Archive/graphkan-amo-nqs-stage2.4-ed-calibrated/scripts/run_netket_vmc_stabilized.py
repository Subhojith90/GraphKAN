#!/usr/bin/env python
"""ED-calibrated NetKet VMC stabilisation runner.

This runner is intentionally separate from the Stage 2.3 registered sweep. It writes to a
new output root, validates the NetKet operator against the exact dense engine for every
small instance, records sampled-state diversity, and labels degenerate trajectories instead
of calling every completed process a valid scientific run.
"""
from __future__ import annotations
import argparse, json, math, time, traceback
from pathlib import Path
import numpy as np
import pandas as pd

from _common import load_config, build_geometries_from_config, parameter_grid, hash_dict
from graphkan_amo_nqs.hamiltonians.rydberg import RydbergParams, instance_id, build_dense_rydberg
from graphkan_amo_nqs.hamiltonians.netket_builders import build_netket_rydberg_operator
from graphkan_amo_nqs.models.flax_nqs import build_flax_model
from graphkan_amo_nqs.training.nqs_preflight import preflight_dict


def append_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if path.exists():
        frame.to_csv(path, mode="a", index=False, header=False)
    else:
        frame.to_csv(path, index=False)


def count_params(model, n_sites: int, seed: int) -> int:
    import jax
    import jax.numpy as jnp
    variables = model.init(jax.random.PRNGKey(seed), jnp.ones((1, n_sites), dtype=jnp.float64))
    return int(sum(np.asarray(x).size for x in jax.tree_util.tree_leaves(variables["params"])))


def exact_reference(positions: np.ndarray, params: RydbergParams) -> tuple[float, np.ndarray]:
    dense = np.asarray(build_dense_rydberg(positions, params), dtype=np.float64)
    eigs = np.linalg.eigvalsh(dense)
    return float(eigs[0]), dense


def operator_check(positions: np.ndarray, params: RydbergParams, h_netket) -> dict:
    exact_ground, exact_dense = exact_reference(positions, params)
    netket_dense = np.asarray(h_netket.to_dense(), dtype=np.complex128)
    matrix_max_abs_diff = float(np.max(np.abs(exact_dense.astype(np.complex128) - netket_dense)))
    spectrum_diff = float(np.max(np.abs(np.linalg.eigvalsh(exact_dense) - np.linalg.eigvalsh(netket_dense))))
    return {
        "exact_ground_energy": exact_ground,
        "operator_matrix_max_abs_diff": matrix_max_abs_diff,
        "operator_spectrum_max_abs_diff": spectrum_diff,
        "operator_valid": bool(matrix_max_abs_diff <= 1e-10 and spectrum_diff <= 1e-10),
    }


def sample_diagnostics(vstate) -> dict:
    """Extract lightweight, API-tolerant sampler diagnostics after a completed VMC run."""
    try:
        samples = np.asarray(vstate.samples)
        flat = samples.reshape((-1, samples.shape[-1]))
        unique = np.unique(flat, axis=0).shape[0]
        n_total = int(flat.shape[0])
        unique_fraction = float(unique / max(n_total, 1))
        occupation_mean = float(np.mean((flat + 1.0) * 0.5))
        occupation_var = float(np.var((flat + 1.0) * 0.5))
        return {
            "samples_available": True,
            "sample_count_observed": n_total,
            "unique_sampled_states": int(unique),
            "unique_state_fraction": unique_fraction,
            "occupation_mean": occupation_mean,
            "occupation_variance": occupation_var,
        }
    except Exception as exc:
        return {
            "samples_available": False,
            "sample_diagnostics_error": f"{type(exc).__name__}: {exc}",
            "sample_count_observed": np.nan,
            "unique_sampled_states": np.nan,
            "unique_state_fraction": np.nan,
            "occupation_mean": np.nan,
            "occupation_variance": np.nan,
        }


def classify_run(energy_mean: float, energy_error: float, energy_variance: float | None,
                 diag: dict, exact_ground: float, cfg: dict) -> tuple[str, list[str]]:
    flags: list[str] = []
    if not (math.isfinite(energy_mean) and math.isfinite(energy_error)):
        flags.append("NONFINITE_ESTIMATOR")
    if energy_error <= float(cfg.get("min_energy_error", 1e-10)):
        flags.append("NEAR_ZERO_ERROR")
    if energy_variance is not None and math.isfinite(energy_variance) and energy_variance <= float(cfg.get("min_energy_variance", 1e-12)):
        flags.append("NEAR_ZERO_ENERGY_VARIANCE")
    if diag.get("samples_available"):
        if diag.get("unique_sampled_states", 0) <= int(cfg.get("min_unique_states", 2)):
            flags.append("SAMPLER_SINGLE_STATE")
        elif diag.get("unique_state_fraction", 0.0) < float(cfg.get("min_unique_state_fraction", 0.01)):
            flags.append("LOW_SAMPLE_DIVERSITY")
    else:
        flags.append("NO_SAMPLE_DIAGNOSTICS")
    energy_error_to_ed = abs(energy_mean - exact_ground)
    if energy_error_to_ed > float(cfg.get("max_abs_energy_error_to_ed", 0.5)):
        flags.append("LARGE_ED_ERROR")
    return ("VALID" if not flags else "FLAGGED"), flags


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Stage 2.4 ED-calibrated VMC stabilisation profile.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    root = Path(cfg.get("output_root", "results_stage2_4"))
    profile = cfg.get("profile", "netket_stabilisation")
    preflight = preflight_dict(True)
    (root / "environment").mkdir(parents=True, exist_ok=True)
    (root / "environment" / "nqs_preflight.json").write_text(json.dumps(preflight, indent=2) + "\n")
    if not preflight["ok"]:
        raise SystemExit("Preflight failed. No Stage 2.4 result was generated.")
    if args.overwrite and (root / "raw").exists():
        for p in (root / "raw").glob("*.csv"):
            p.unlink()

    import optax
    import netket as nk

    training = cfg.get("training", {})
    health_cfg = cfg.get("health_gate", {})
    config_hash = hash_dict(cfg)
    rows: list[dict] = []
    history: list[dict] = []

    for geometry in build_geometries_from_config(cfg):
        for params in parameter_grid(cfg):
            hi, hamiltonian = build_netket_rydberg_operator(geometry.positions, params)
            op = operator_check(geometry.positions, params, hamiltonian)
            if not op["operator_valid"]:
                raise RuntimeError(f"Operator validation failed for {instance_id(geometry, params)}: {op}")
            r_scale = params.r_omega if np.isfinite(params.r_omega) else 1.0
            for model_name in cfg.get("models", []):
                model_cfg = {**training, **cfg.get("model_overrides", {}).get(model_name, {})}
                for seed in model_cfg.get("seeds", [0, 1, 2]):
                    run_id = f"{profile}:{instance_id(geometry, params)}:{model_name}:seed{seed}"
                    started = time.perf_counter()
                    try:
                        model = build_flax_model(
                            model_name,
                            geometry.positions,
                            n_basis=int(model_cfg.get("n_basis", 8)),
                            hidden=int(model_cfg.get("hidden", 16)),
                            r_scale=r_scale,
                        )
                        n_params = count_params(model, geometry.n_sites, int(seed))
                        sampler = nk.sampler.MetropolisLocal(
                            hi, n_chains_per_rank=int(model_cfg.get("n_chains", 32))
                        )
                        vstate = nk.vqs.MCState(
                            sampler,
                            model,
                            n_samples=int(model_cfg.get("n_samples", 2048)),
                            n_discard_per_chain=int(model_cfg.get("n_discard_per_chain", 32)),
                            seed=int(seed),
                        )
                        optimizer = optax.chain(
                            optax.clip_by_global_norm(float(model_cfg.get("grad_clip_norm", 5.0))),
                            optax.adam(float(model_cfg.get("learning_rate", 1e-3))),
                        )
                        sr = nk.optimizer.SR(diag_shift=float(model_cfg.get("diag_shift", 0.1)))
                        driver = nk.driver.VMC(hamiltonian, optimizer, variational_state=vstate, preconditioner=sr)
                        n_iter = int(model_cfg.get("n_iter", 250))
                        chunk = int(model_cfg.get("chunk_size", 25))
                        for step in range(0, n_iter, chunk):
                            driver.run(n_iter=min(chunk, n_iter - step), out=None, show_progress=False)
                            est = driver.energy
                            history.append({
                                "run_id": run_id, "step": step + min(chunk, n_iter - step),
                                "energy_mean": float(np.real(est.mean)), "energy_error": float(est.error_of_mean),
                                "energy_variance": float(getattr(est, "variance", np.nan)),
                                "profile": profile, "model": model_name, "seed": seed,
                                "config_hash": config_hash,
                            })
                        energy = driver.energy
                        energy_mean = float(np.real(energy.mean))
                        energy_error = float(energy.error_of_mean)
                        energy_variance = float(getattr(energy, "variance", np.nan))
                        diag = sample_diagnostics(vstate)
                        validity, flags = classify_run(
                            energy_mean, energy_error, energy_variance, diag,
                            op["exact_ground_energy"], health_cfg,
                        )
                        row = {
                            "run_id": run_id, "profile": profile, "status": "COMPLETED",
                            "scientific_validity": validity, "health_flags": ";".join(flags),
                            "model": model_name, "seed": seed, "n_sites": geometry.n_sites,
                            "instance_id": instance_id(geometry, params), "omega": params.omega,
                            "delta": params.delta, "c6": params.c6, "energy_mean": energy_mean,
                            "energy_error": energy_error, "energy_variance": energy_variance,
                            "abs_energy_error_to_ed": abs(energy_mean - op["exact_ground_energy"]),
                            "n_parameters": n_params, "n_samples": int(model_cfg.get("n_samples", 2048)),
                            "n_iter": n_iter, "wall_seconds": time.perf_counter() - started,
                            "config_hash": config_hash, "backend": "netket_flax_stage2_4", **op, **diag,
                        }
                        rows.append(row)
                        print(f"{validity.lower()} {run_id}: E={energy_mean:.8f}, |E-E0|={row['abs_energy_error_to_ed']:.4g}, flags={row['health_flags'] or 'none'}")
                    except Exception as exc:
                        rows.append({
                            "run_id": run_id, "profile": profile, "status": "FAILED",
                            "scientific_validity": "INVALID", "health_flags": "EXECUTION_FAILURE",
                            "model": model_name, "seed": seed, "n_sites": geometry.n_sites,
                            "instance_id": instance_id(geometry, params), "omega": params.omega,
                            "delta": params.delta, "c6": params.c6, "error_type": type(exc).__name__,
                            "error_message": str(exc), "traceback": traceback.format_exc(),
                            "config_hash": config_hash, "backend": "netket_flax_stage2_4", **op,
                        })
                        print(f"FAILED {run_id}: {type(exc).__name__}: {exc}")
    append_rows(root / "raw" / "netket_vmc_stabilized_metrics.csv", rows)
    append_rows(root / "raw" / "netket_vmc_stabilized_history.csv", history)
    pd.DataFrame(rows).to_csv(root / "raw" / "netket_vmc_stabilized_metrics_snapshot.csv", index=False)
    valid_count = sum(r.get("scientific_validity") == "VALID" for r in rows)
    print(f"Stage 2.4 complete: {valid_count}/{len(rows)} runs passed the scientific-health gate.")
    if valid_count == 0:
        raise SystemExit("No run passed the Stage 2.4 health gate. Tune before any benchmark claim.")


if __name__ == "__main__":
    main()
