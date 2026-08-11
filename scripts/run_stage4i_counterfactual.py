#!/usr/bin/env python3
"""Run one resumable five-seed Stage 4I counterfactual shard locally."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
import traceback
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
from graphkan_amo_nqs.observables.correlations import correlation_mae
from graphkan_amo_nqs.training.stage4h_transplant import (
    find_radial_checkpoint,
    full_basis_attestation,
    load_stage4g_radial_checkpoint,
    sha256_file,
    transplant_parameters,
)
from graphkan_amo_nqs.training.stage4i_orthogonal import (
    model_projection,
    orthogonalisation,
)
from run_stage4g import classify_exact, physical_sigma_from_binary, sample_observables
from run_stage4h_warmstart import (
    exact_observables,
    load_stage4g_reference,
    next_attempt,
    region_rows,
    separation_rows,
    write_json,
)
from stage4i_protocol import (
    DEFAULT_CONFIG,
    canonical_bytes,
    read_config,
    roster,
    validate_config,
)


def save_stage4i_checkpoint(path: Path, parameters: dict, metadata: dict) -> dict:
    arrays = {str(name): np.asarray(value) for name, value in parameters.items()}
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    payload = buffer.getvalue()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
    manifest = {
        **metadata,
        "schema_version": "stage4i-named-checkpoint-v1",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "parameter_shapes": {name: list(value.shape) for name, value in arrays.items()},
    }
    write_json(path.with_suffix(".json"), manifest, exclusive=True)
    return manifest


def run_one(row: dict, cfg: dict, stage4g_root: Path, output: Path) -> dict:
    run_root = output / "runs" / row["run_id"]
    completion = run_root / "completion.json"
    if completion.is_file():
        prior = json.loads(completion.read_text())
        if prior.get("status") == "COMPLETED":
            print("SKIP completed", row["run_id"])
            return prior
    attempt = next_attempt(run_root)
    started = time.perf_counter()
    logs: list[str] = []
    pilot = cfg["counterfactual"]
    training = pilot["training"]
    acceptance = pilot["acceptance"]
    spec = pilot["arms"][row["arm"]]
    write_json(
        attempt / "resolved_config.json",
        {
            **row,
            "training": training,
            "arm_spec": spec,
            "protocol_schema": cfg["protocol"]["schema_version"],
            "protocol_hash": hashlib.sha256(canonical_bytes(cfg)).hexdigest(),
        },
        exclusive=True,
    )
    try:
        import jax

        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
        import netket as nk
        import optax

        geometry = make_chain_geometry(12)
        params = params_from_values(1.0, row["delta"], row["c6"], "parity")
        distances = geometry_distance_matrix(geometry)
        hi, hamiltonian = build_netket_rydberg_operator(
            geometry.positions,
            params,
            inverted_ordering=False,
            distance_matrix=distances,
        )
        dense_h = np.asarray(
            build_dense_rydberg(geometry.positions, params, distance_matrix=distances),
            complex,
        )
        if (
            float(np.max(np.abs(dense_h - np.asarray(hamiltonian.to_dense(), complex))))
            > 1e-10
        ):
            raise RuntimeError("dense/NetKet operator mismatch")
        states = enumerate_binary_states(12)
        sigma = physical_sigma_from_binary(states, False)
        e0, ed_density, ed_corr = load_stage4g_reference(
            stage4g_root, delta=row["delta"], c6=row["c6"]
        )
        source = find_radial_checkpoint(
            stage4g_root, delta=row["delta"], c6=row["c6"], seed=row["seed"]
        )
        radial_np = load_stage4g_radial_checkpoint(source)
        radial_params = {
            k: jnp.asarray(v, dtype=jnp.float64) for k, v in radial_np.items()
        }
        radial_model = build_flax_model(
            "kan_jastrow",
            geometry.positions,
            n_basis=32,
            hidden=1,
            r_scale=params.r_omega,
            omega=1.0,
            basis_family="rbf",
            distance_matrix=distances,
        )
        orth = orthogonalisation(12, params.r_omega)
        if orth["validation"] != "PASS":
            raise RuntimeError("span-preservation attestation failed")
        write_json(
            attempt / "orthogonalisation_attestation.json",
            {k: v for k, v in orth.items() if not isinstance(v, np.ndarray)},
            exclusive=True,
        )
        with (attempt / "orthogonalisation_projection.npz").open("xb") as handle:
            np.savez_compressed(
                handle,
                one_body_projection=orth["one_body_projection"],
                radial_projection=orth["radial_projection"],
            )
        target_model = build_flax_model(
            spec["kind"],
            geometry.positions,
            n_basis=32,
            hidden=1,
            r_scale=params.r_omega,
            omega=1.0,
            basis_family="rbf",
            distance_matrix=distances,
            endpoint_orthogonalisation=model_projection(orth)
            if row["arm"] == "stage4i_endpoint_full_orthogonal"
            else None,
        )
        template = target_model.init(
            jax.random.PRNGKey(row["seed"]), jnp.ones((1, 12), dtype=jnp.float64)
        )["params"]
        target_params = transplant_parameters(template, radial_np, row["arm"])
        transplant = full_basis_attestation(
            radial_model,
            radial_params,
            target_model,
            target_params,
            sigma,
            tolerance=float(acceptance["transplant_tolerance"]),
        )
        transplant["schema_version"] = "stage4i-full-basis-transplant-v1"
        _, _, radial_e, radial_n, radial_c = exact_observables(
            radial_model, radial_params, sigma, states, dense_h
        )
        _, _, target_e, target_n, target_c = exact_observables(
            target_model, target_params, sigma, states, dense_h
        )
        transplant.update(
            {
                "run_id": row["run_id"],
                "arm": row["arm"],
                "source_checkpoint": str(source),
                "source_checkpoint_sha256": sha256_file(source),
                "exact_energy_difference": abs(target_e - radial_e),
                "maximum_density_difference": float(
                    np.max(np.abs(target_n - radial_n))
                ),
                "maximum_correlation_difference": float(
                    np.max(np.abs(target_c - radial_c))
                ),
            }
        )
        if max(
            transplant["exact_energy_difference"],
            transplant["maximum_density_difference"],
            transplant["maximum_correlation_difference"],
        ) > float(acceptance["transplant_tolerance"]):
            transplant["validation"] = "FAIL"
        write_json(attempt / "transplant_attestation.json", transplant, exclusive=True)
        if transplant["validation"] != "PASS":
            raise RuntimeError("radial warm-start equality failed")

        sampler = nk.sampler.MetropolisLocal(
            hi, n_chains_per_rank=int(training["n_chains"])
        )
        state = nk.vqs.MCState(
            sampler,
            target_model,
            n_samples=int(training["n_samples"]),
            n_discard_per_chain=int(training["n_discard_per_chain"]),
            seed=row["seed"],
        )
        state.parameters = target_params
        if callable(getattr(state, "reset", None)):
            state.reset()
        optimiser = optax.chain(
            optax.clip_by_global_norm(float(training["grad_clip_norm"])),
            optax.adam(float(training["learning_rate"])),
        )
        driver = nk.driver.VMC(
            hamiltonian,
            optimiser,
            variational_state=state,
            preconditioner=nk.optimizer.SR(diag_shift=float(training["diag_shift"])),
        )
        history, sentinels = [], []
        total, chunk = (
            int(training["additional_iterations"]),
            int(training["chunk_size"]),
        )
        for step in range(chunk, total + 1, chunk):
            driver.run(n_iter=chunk, out=None, show_progress=False)
            estimate = driver.energy
            history.append(
                {
                    "run_id": row["run_id"],
                    "step": step,
                    "energy_mean": float(np.real(estimate.mean)),
                    "energy_error": float(estimate.error_of_mean),
                }
            )
            if step % int(training["immutable_checkpoint_interval"]) == 0:
                save_stage4i_checkpoint(
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
            if step in set(training["exact_sentinel_steps"]):
                _, _, ee, en, ec = exact_observables(
                    target_model, state.parameters, sigma, states, dense_h
                )
                sentinels.append(
                    {
                        **row,
                        "step": step,
                        "exact_model_energy": ee,
                        "exact_model_abs_energy_error_to_ed": abs(ee - e0),
                        "exact_model_density_mae_to_ed": float(
                            np.mean(np.abs(en - ed_density))
                        ),
                        "exact_model_connected_correlation_mae_to_ed": float(
                            correlation_mae(ec, ed_corr)
                        ),
                    }
                )
            if step % 250 == 0:
                message = f"{row['run_id']} step={step}/{total} E={float(np.real(estimate.mean)):.12g}"
                print(message, flush=True)
                logs.append(message)

        estimate = driver.energy
        sampled_n, sampled_c, sample_diag = sample_observables(state, 12, False)
        _, _, exact_e, exact_n, exact_c = exact_observables(
            target_model, state.parameters, sigma, states, dense_h
        )
        density_mae = float(np.mean(np.abs(exact_n - ed_density)))
        corr_mae = float(correlation_mae(exact_c, ed_corr))
        health = {
            "max_abs_energy_error_to_ed": 0.5,
            "max_density_mae_to_ed": 0.08,
            "max_connected_correlation_mae_to_ed": 0.08,
        }
        validity, flags = classify_exact(exact_e, e0, density_mae, corr_mae, health)
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
            "source_checkpoint_sha256": sha256_file(source),
            "orthogonalisation_sha256": orth["projection_sha256"],
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
        pd.DataFrame(history).to_csv(attempt / "history.csv", index=False)
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
        (attempt / "stdout.txt").write_text("\n".join(logs) + "\n")
        (attempt / "stderr.txt").write_text("")
        (attempt / "exit_code.txt").write_text("0\n")
        payloads = sorted(p for p in attempt.rglob("*") if p.is_file())
        record = {
            "schema_version": "stage4i-completion-v1",
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
        (attempt / "stdout.txt").write_text("\n".join(logs) + "\n")
        (attempt / "stderr.txt").write_text(failure["traceback"])
        (attempt / "exit_code.txt").write_text("1\n")
        print("FAILED", row["run_id"], exc, file=sys.stderr)
        return failure


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--shard", required=True)
    parser.add_argument("--stage4g-root", type=Path, required=True)
    parser.add_argument(
        "--output-root", type=Path, default=Path("results_stage4i/shards")
    )
    args = parser.parse_args()
    cfg = read_config(args.config)
    validation = validate_config(cfg)
    if validation["validation"] != "PASS":
        raise SystemExit(json.dumps(validation, indent=2))
    selected = [r for r in roster(cfg) if r["shard"] == args.shard]
    if len(selected) != 5:
        raise SystemExit(f"expected five rows for {args.shard}, found {len(selected)}")
    out = args.output_root / args.shard
    out.mkdir(parents=True, exist_ok=True)
    results = [run_one(row, cfg, args.stage4g_root, out) for row in selected]
    summary = {
        "schema_version": "stage4i-shard-summary-v1",
        "shard": args.shard,
        "expected": 5,
        "completed": sum(x.get("status") == "COMPLETED" for x in results),
        "failed": sum(x.get("status") == "FAILED" for x in results),
    }
    write_json(out / "shard_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["completed"] != 5:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
