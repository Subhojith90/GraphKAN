#!/usr/bin/env python3
"""Execute one immutable Stage 4H cell-by-arm shard."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from _common import params_from_values
from graphkan_amo_nqs.geometry import make_chain_geometry
from graphkan_amo_nqs.geometry.distances import geometry_distance_matrix
from graphkan_amo_nqs.hamiltonians.basis import enumerate_binary_states
from graphkan_amo_nqs.hamiltonians.netket_builders import build_netket_rydberg_operator
from graphkan_amo_nqs.hamiltonians.rydberg import build_dense_rydberg
from graphkan_amo_nqs.models.stage4g_flax_nqs import build_flax_model
from graphkan_amo_nqs.observables.correlations import (
    connected_correlations_from_state,
    correlation_mae,
)
from graphkan_amo_nqs.observables.density import density_from_state
from graphkan_amo_nqs.training.stage4h_transplant import (
    find_radial_checkpoint,
    full_basis_attestation,
    load_stage4g_radial_checkpoint,
    sha256_file,
    transplant_parameters,
)
from run_stage4g import classify_exact, physical_sigma_from_binary, sample_observables
from stage4h_protocol import (
    DEFAULT_CONFIG,
    read_config,
    sha256_bytes,
    canonical_bytes,
    validate_config,
)


def write_json(path: Path, value: Any, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "x" if exclusive else "w"
    with path.open(mode) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def save_named_checkpoint(
    path: Path, parameters: Any, metadata: dict[str, Any]
) -> dict[str, Any]:
    arrays = {str(name): np.asarray(value) for name, value in parameters.items()}
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    payload = buffer.getvalue()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
    manifest = {
        **metadata,
        "schema_version": "stage4h-named-checkpoint-v1",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "parameter_shapes": {name: list(value.shape) for name, value in arrays.items()},
    }
    write_json(path.with_suffix(".json"), manifest, exclusive=True)
    return manifest


def exact_observables(
    model: Any,
    parameters: Any,
    sigma: np.ndarray,
    states: np.ndarray,
    dense_h: np.ndarray,
):
    import jax.numpy as jnp

    logs = np.array(
        model.apply({"params": parameters}, jnp.asarray(sigma, dtype=jnp.float64)),
        dtype=np.complex128,
        copy=True,
    )
    logs -= np.max(logs.real)
    psi = np.exp(logs)
    psi /= np.linalg.norm(psi)
    density = density_from_state(psi, states)
    corr = connected_correlations_from_state(psi, states)
    energy = float(np.real(np.vdot(psi, dense_h @ psi)))
    return logs, psi, energy, density, corr


def next_attempt(run_root: Path) -> Path:
    attempts = run_root / "attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    numbers = [
        int(p.name.split("-")[-1]) for p in attempts.glob("attempt-[0-9][0-9][0-9]")
    ]
    path = attempts / f"attempt-{max(numbers, default=0) + 1:03d}"
    path.mkdir()
    return path


def region_rows(
    common: dict[str, Any], density: np.ndarray, reference: np.ndarray
) -> list[dict[str, Any]]:
    n = len(density)
    groups = {
        "edge": [0, n - 1],
        "near_edge": [1, n - 2],
        "bulk": list(range(2, n - 2)),
    }
    return [
        {
            **common,
            "region": name,
            "sites": ";".join(map(str, sites)),
            "exact_model_mae": float(
                np.mean(np.abs(density[sites] - reference[sites]))
            ),
            "signed_density_difference": float(
                np.mean(density[sites] - reference[sites])
            ),
        }
        for name, sites in groups.items()
    ]


def separation_rows(
    common: dict[str, Any], corr: np.ndarray, reference: np.ndarray
) -> list[dict[str, Any]]:
    rows = []
    for separation in range(1, len(corr)):
        i = np.arange(0, len(corr) - separation)
        j = i + separation
        rows.append(
            {
                **common,
                "separation": separation,
                "pair_count": len(i),
                "exact_model_mae": float(np.mean(np.abs(corr[i, j] - reference[i, j]))),
                "signed_correlation_difference": float(
                    np.mean(corr[i, j] - reference[i, j])
                ),
            }
        )
    return rows


def load_stage4g_reference(
    root: Path, *, delta: float, c6: float
) -> tuple[float, np.ndarray, np.ndarray]:
    """Load the accepted Stage 4G ED reference without repeating diagonalisation."""

    def frames(names: tuple[str, ...]) -> pd.DataFrame:
        paths = []
        for name in names:
            paths.extend(root.rglob(name))
        if not paths:
            raise FileNotFoundError(
                f"Stage 4G reference table missing under {root}: {names}"
            )
        return pd.concat(
            [pd.read_csv(path) for path in sorted(set(paths))], ignore_index=True
        ).drop_duplicates()

    metrics = frames(("reference_metrics.csv", "stage4g_metrics.csv"))
    mask = (
        (metrics.model_label == "rbf_radial_b32")
        & np.isclose(metrics.delta.astype(float), delta)
        & np.isclose(metrics.c6.astype(float), c6)
    )
    selected = metrics[mask]
    if "boundary_condition" in selected and len(selected):
        selected = selected[
            selected.boundary_condition.astype(str).str.contains("open")
        ]
    energies = selected.exact_ground_energy.dropna().astype(float).unique()
    if len(energies) != 1:
        raise ValueError(
            f"expected one Stage 4G exact ground energy for {delta}/{c6}, found {len(energies)}"
        )
    site = frames(("reference_site_observables.csv", "stage4g_site_observables.csv"))
    site = site[
        (site.model_label == "rbf_radial_b32")
        & np.isclose(site.delta.astype(float), delta)
        & np.isclose(site.c6.astype(float), c6)
    ]
    site = site.sort_values(["seed", "site"]).drop_duplicates("site")
    corr = frames(("reference_corr_observables.csv", "stage4g_corr_observables.csv"))
    corr = corr[
        (corr.model_label == "rbf_radial_b32")
        & np.isclose(corr.delta.astype(float), delta)
        & np.isclose(corr.c6.astype(float), c6)
    ]
    corr = corr.sort_values(["seed", "i", "j"]).drop_duplicates(["i", "j"])
    if len(site) != 12 or len(corr) != 144:
        raise ValueError(
            f"incomplete Stage 4G ED observable reference for {delta}/{c6}"
        )
    density = site.sort_values("site").ed_density.to_numpy(float)
    correlation = (
        corr.sort_values(["i", "j"]).ed_connected.to_numpy(float).reshape(12, 12)
    )
    return float(energies[0]), density, correlation


def run_one(
    row: dict[str, Any], cfg: dict[str, Any], stage4g_root: Path, output: Path
) -> dict[str, Any]:
    run_root = output / "runs" / row["run_id"]
    completion = run_root / "completion.json"
    if completion.is_file():
        prior = json.loads(completion.read_text())
        if prior.get("status") == "COMPLETED":
            print("SKIP completed", row["run_id"])
            return prior
    attempt = next_attempt(run_root)
    log_lines: list[str] = []
    resolved = {
        **row,
        "protocol_hash": sha256_bytes(canonical_bytes(cfg)),
        "training": cfg["training"],
        "arm_spec": cfg["arms"][row["arm"]],
    }
    write_json(attempt / "resolved_config.json", resolved, exclusive=True)
    started = time.perf_counter()
    try:
        import jax

        jax.config.update("jax_enable_x64", True)
        import netket as nk
        import optax
        import jax.numpy as jnp

        geometry = make_chain_geometry(12)
        params = params_from_values(
            cfg["physics"]["omega"], row["delta"], row["c6"], "parity"
        )
        distances = geometry_distance_matrix(geometry)
        hi, hamiltonian = build_netket_rydberg_operator(
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
        spectrum_diff = 0.0
        if matrix_diff > 1e-10:
            raise RuntimeError("dense/NetKet operator mismatch")
        states = enumerate_binary_states(12)
        sigma = physical_sigma_from_binary(states, False)
        e0, ed_density, ed_corr = load_stage4g_reference(
            stage4g_root, delta=row["delta"], c6=row["c6"]
        )

        source = find_radial_checkpoint(
            stage4g_root, delta=row["delta"], c6=row["c6"], seed=row["seed"]
        )
        radial_params_np = load_stage4g_radial_checkpoint(source)
        radial_params = {
            name: jnp.asarray(value, dtype=jnp.float64)
            for name, value in radial_params_np.items()
        }
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
        spec = cfg["arms"][row["arm"]]
        target_model = build_flax_model(
            spec["kind"],
            geometry.positions,
            n_basis=32,
            hidden=1,
            r_scale=params.r_omega,
            omega=params.omega,
            basis_family="rbf",
            distance_matrix=distances,
        )
        template = target_model.init(
            jax.random.PRNGKey(int(row["seed"])), jnp.ones((1, 12), dtype=jnp.float64)
        )["params"]
        target_params = transplant_parameters(template, radial_params_np, row["arm"])
        attestation = full_basis_attestation(
            radial_model,
            radial_params,
            target_model,
            target_params,
            sigma,
            tolerance=cfg["acceptance"]["transplant_max_log_amplitude_difference"],
        )
        _, _, radial_e, radial_n, radial_c = exact_observables(
            radial_model, radial_params, sigma, states, dense_h
        )
        _, _, target_e, target_n, target_c = exact_observables(
            target_model, target_params, sigma, states, dense_h
        )
        attestation.update(
            {
                "source_checkpoint": str(source),
                "source_checkpoint_sha256": sha256_file(source),
                "arm": row["arm"],
                "run_id": row["run_id"],
                "exact_energy_difference": abs(target_e - radial_e),
                "maximum_density_difference": float(
                    np.max(np.abs(target_n - radial_n))
                ),
                "maximum_correlation_difference": float(
                    np.max(np.abs(target_c - radial_c))
                ),
            }
        )
        limits = cfg["acceptance"]
        derived_pass = (
            attestation["exact_energy_difference"]
            <= limits["transplant_max_exact_energy_difference"]
            and attestation["maximum_density_difference"]
            <= limits["transplant_max_density_difference"]
            and attestation["maximum_correlation_difference"]
            <= limits["transplant_max_correlation_difference"]
        )
        attestation["validation"] = (
            "PASS" if attestation["validation"] == "PASS" and derived_pass else "FAIL"
        )
        write_json(attempt / "transplant_attestation.json", attestation, exclusive=True)
        if attestation["validation"] != "PASS":
            raise RuntimeError("full-basis transplant equality failed")

        sampler = nk.sampler.MetropolisLocal(
            hi, n_chains_per_rank=int(cfg["training"]["n_chains"])
        )
        state = nk.vqs.MCState(
            sampler,
            target_model,
            n_samples=int(cfg["training"]["n_samples"]),
            n_discard_per_chain=int(cfg["training"]["n_discard_per_chain"]),
            seed=int(row["seed"]),
        )
        state.parameters = target_params
        if callable(getattr(state, "reset", None)):
            state.reset()
        optimiser = optax.chain(
            optax.clip_by_global_norm(float(cfg["training"]["grad_clip_norm"])),
            optax.adam(float(cfg["training"]["learning_rate"])),
        )
        driver = nk.driver.VMC(
            hamiltonian,
            optimiser,
            variational_state=state,
            preconditioner=nk.optimizer.SR(
                diag_shift=float(cfg["training"]["diag_shift"])
            ),
        )
        histories, sentinels = [], []
        total, chunk = (
            int(cfg["training"]["additional_iterations"]),
            int(cfg["training"]["chunk_size"]),
        )
        checkpoint_interval = int(cfg["training"]["immutable_checkpoint_interval"])
        sentinel_steps = set(map(int, cfg["training"]["exact_sentinel_steps"]))
        for step in range(chunk, total + 1, chunk):
            driver.run(n_iter=chunk, out=None, show_progress=False)
            estimate = driver.energy
            histories.append(
                {
                    "run_id": row["run_id"],
                    "step": step,
                    "energy_mean": float(np.real(estimate.mean)),
                    "energy_error": float(estimate.error_of_mean),
                }
            )
            if step % checkpoint_interval == 0:
                save_named_checkpoint(
                    attempt / "checkpoints" / f"step-{step:05d}.npz",
                    state.parameters,
                    {
                        "run_id": row["run_id"],
                        "arm": row["arm"],
                        "seed": row["seed"],
                        "step": step,
                        "source_checkpoint_sha256": sha256_file(source),
                    },
                )
            if step in sentinel_steps:
                _, _, exact_e, exact_n, exact_c = exact_observables(
                    target_model, state.parameters, sigma, states, dense_h
                )
                sentinels.append(
                    {
                        "run_id": row["run_id"],
                        "arm": row["arm"],
                        "seed": row["seed"],
                        "delta": row["delta"],
                        "c6": row["c6"],
                        "step": step,
                        "exact_model_energy": exact_e,
                        "exact_model_abs_energy_error_to_ed": abs(exact_e - e0),
                        "exact_model_density_mae_to_ed": float(
                            np.mean(np.abs(exact_n - ed_density))
                        ),
                        "exact_model_connected_correlation_mae_to_ed": float(
                            correlation_mae(exact_c, ed_corr)
                        ),
                    }
                )
            if step % 250 == 0:
                message = f"{row['run_id']} step={step}/{total} E={float(np.real(estimate.mean)):.12g}"
                print(message, flush=True)
                log_lines.append(message)

        estimate = driver.energy
        sampled_n, sampled_c, sample_diag = sample_observables(state, 12, False)
        _, _, exact_e, exact_n, exact_c = exact_observables(
            target_model, state.parameters, sigma, states, dense_h
        )
        density_mae = float(np.mean(np.abs(exact_n - ed_density)))
        corr_mae = float(correlation_mae(exact_c, ed_corr))
        validity, flags = classify_exact(
            exact_e, e0, density_mae, corr_mae, cfg["health"]
        )
        metrics = {
            **row,
            "status": "COMPLETED",
            "scientific_validity_exact": validity,
            "exact_health_flags": ";".join(flags),
            "exact_ground_energy": e0,
            "exact_model_energy": exact_e,
            "exact_model_abs_energy_error_to_ed": abs(exact_e - e0),
            "exact_model_density_mae_to_ed": density_mae,
            "exact_model_connected_correlation_mae_to_ed": corr_mae,
            "sample_density_mae_to_ed": float(np.mean(np.abs(sampled_n - ed_density))),
            "sample_connected_correlation_mae_to_ed": float(
                correlation_mae(sampled_c, ed_corr)
            ),
            "energy_mean": float(np.real(estimate.mean)),
            "energy_error": float(estimate.error_of_mean),
            "operator_matrix_max_abs_diff": matrix_diff,
            "operator_spectrum_max_abs_diff": spectrum_diff,
            "source_checkpoint_sha256": sha256_file(source),
            "wall_seconds": time.perf_counter() - started,
            **sample_diag,
        }
        common = {k: metrics[k] for k in ("run_id", "arm", "seed", "delta", "c6")}
        site = [
            {
                **common,
                "site": i,
                "ed_density": ed_density[i],
                "exact_model_density": exact_n[i],
                "exact_model_abs_error": abs(exact_n[i] - ed_density[i]),
            }
            for i in range(12)
        ]
        corr = [
            {
                **common,
                "i": i,
                "j": j,
                "ed_connected": ed_corr[i, j],
                "exact_model_connected": exact_c[i, j],
                "exact_model_abs_error": abs(exact_c[i, j] - ed_corr[i, j]),
            }
            for i in range(12)
            for j in range(12)
        ]
        pd.DataFrame(histories).to_csv(attempt / "history.csv", index=False)
        pd.DataFrame(sentinels).to_csv(attempt / "exact_sentinels.csv", index=False)
        pd.DataFrame(site).to_csv(attempt / "site_observables.csv", index=False)
        pd.DataFrame(corr).to_csv(attempt / "correlation_observables.csv", index=False)
        pd.DataFrame(region_rows(common, exact_n, ed_density)).to_csv(
            attempt / "region_density.csv", index=False
        )
        pd.DataFrame(separation_rows(common, exact_c, ed_corr)).to_csv(
            attempt / "separation_correlation.csv", index=False
        )
        write_json(attempt / "metrics.json", metrics, exclusive=True)
        (attempt / "stdout.txt").write_text("\n".join(log_lines) + "\n")
        (attempt / "stderr.txt").write_text("")
        (attempt / "exit_code.txt").write_text("0\n")
        payloads = sorted(p for p in attempt.rglob("*") if p.is_file())
        record = {
            "schema_version": "stage4h-completion-v1",
            "status": "COMPLETED",
            "run_id": row["run_id"],
            "attempt": attempt.name,
            "payload_hashes": {
                str(p.relative_to(attempt)): sha256_file(p) for p in payloads
            },
        }
        write_json(completion, record, exclusive=True)
        return record
    except Exception as exc:
        failure = {
            **row,
            "status": "FAILED",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
            "wall_seconds": time.perf_counter() - started,
        }
        write_json(attempt / "failure.json", failure)
        (attempt / "stdout.txt").write_text("\n".join(log_lines) + "\n")
        (attempt / "stderr.txt").write_text(failure["traceback"])
        (attempt / "exit_code.txt").write_text("1\n")
        print("FAILED", row["run_id"], type(exc).__name__, exc, file=sys.stderr)
        return failure


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--shard", required=True)
    parser.add_argument("--stage4g-root", type=Path, required=True)
    parser.add_argument(
        "--output-root", type=Path, default=Path("results_stage4h/shards")
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg = read_config(args.config)
    validation = validate_config(cfg)
    if validation["validation"] != "PASS":
        raise SystemExit(json.dumps(validation, indent=2))
    from stage4h_protocol import roster

    selected = [row for row in roster(cfg) if row["shard"] == args.shard]
    if len(selected) != 5:
        raise SystemExit(
            f"unknown or malformed shard {args.shard!r}; expected five rows, found {len(selected)}"
        )
    output = args.output_root / args.shard
    output.mkdir(parents=True, exist_ok=True)
    environment = {
        "schema_version": "stage4h-environment-v1",
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "protocol_hash": validation["protocol_hash"],
        "shard": args.shard,
    }
    write_json(output / "environment.json", environment)
    results = [run_one(row, cfg, args.stage4g_root, output) for row in selected]
    summary = {
        "schema_version": "stage4h-shard-summary-v1",
        "shard": args.shard,
        "expected": 5,
        "completed": sum(r.get("status") == "COMPLETED" for r in results),
        "failed": sum(r.get("status") == "FAILED" for r in results),
    }
    write_json(output / "shard_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["completed"] != 5:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
