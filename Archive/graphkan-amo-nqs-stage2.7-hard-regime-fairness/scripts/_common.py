from __future__ import annotations

from pathlib import Path
from time import perf_counter
import json
import traceback

import numpy as np
import pandas as pd

from graphkan_amo_nqs import __version__
from graphkan_amo_nqs.config import load_config
from graphkan_amo_nqs.manifests import hash_dict, hash_file, atomic_write_dataframe, atomic_write_json
from graphkan_amo_nqs.geometry import make_chain_geometry, make_square_grid_geometry, make_ladder_geometry
from graphkan_amo_nqs.geometry.hashing import distance_matrix_hash
from graphkan_amo_nqs.hamiltonians import RydbergParams, enumerate_binary_states, build_sparse_rydberg, build_dense_rydberg, analytic_diagonal_spectrum
from graphkan_amo_nqs.hamiltonians.rydberg import instance_id
from graphkan_amo_nqs.hamiltonians.exact_diag import compute_rydberg_reference, lowest_eigenpair
from graphkan_amo_nqs.models.registry import make_model
from graphkan_amo_nqs.training.exact_fullsum import train_feature_model, variational_state
from graphkan_amo_nqs.observables.density import density_from_state
from graphkan_amo_nqs.observables.correlations import connected_correlations_from_state, correlation_mae
from graphkan_amo_nqs.observables.fidelity import fidelity
from graphkan_amo_nqs.interpretability.extract_functions import extract_edge_function
from graphkan_amo_nqs.interpretability.correlation_matching import correlation_matching_score
from graphkan_amo_nqs.training.failed_run_policy import failed_run_row


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def source_hash() -> str:
    return hash_file(repo_root() / "src" / "graphkan_amo_nqs" / "training" / "exact_fullsum.py")[:16]


def ensure_result_dirs(root: Path):
    for sub in ["references", "raw", "aggregated", "interpretability", "ibm_optional"]:
        (root / sub).mkdir(parents=True, exist_ok=True)


def params_from_values(omega, delta, c6, sign_rule="parity"):
    return RydbergParams(omega=float(omega), delta=float(delta), c6=float(c6), sign_rule=sign_rule)


def metric_row(profile, inst_id, geometry, params, split, model, seed, reference_method, metric, value, config_hash, units=""):
    return {
        "profile": profile,
        "instance_id": inst_id,
        "geometry": geometry.family,
        "n_sites": geometry.n_sites,
        "omega": params.omega,
        "delta": params.delta,
        "c6": params.c6,
        "split": split,
        "model": model,
        "seed": seed,
        "reference_method": reference_method,
        "metric": metric,
        "value": value,
        "run_status": "completed",
        "code_version": __version__,
        "config_hash": config_hash,
        "source_file_hash": source_hash(),
        "units": units,
    }


def build_geometries_from_config(cfg: dict) -> list:
    geoms = []
    if "geometry" in cfg:
        g = cfg["geometry"]
        fam = g.get("family", "chain")
        if fam == "chain":
            geoms.append(make_chain_geometry(int(g["n_sites"]), spacing=float(g.get("spacing", 1.0)), seed=g.get("seed")))
        elif fam == "square":
            geoms.append(make_square_grid_geometry(int(g["rows"]), int(g["cols"]), spacing=float(g.get("spacing", 1.0)), seed=g.get("seed")))
    if "chains" in cfg:
        for n in cfg["chains"].get("n_sites", []):
            geoms.append(make_chain_geometry(int(n), spacing=float(cfg["chains"].get("spacing", 1.0))))
    if "instances" in cfg:
        inst = cfg["instances"]
        for n in inst.get("chains", {}).get("n_sites", []):
            geoms.append(make_chain_geometry(int(n), spacing=float(inst.get("chains", {}).get("spacing", 1.0))))
        for rows, cols in inst.get("squares", {}).get("shapes", []):
            geoms.append(make_square_grid_geometry(int(rows), int(cols)))
        for width, length in inst.get("ladders", {}).get("shapes", []):
            geoms.append(make_ladder_geometry(length=int(length), width=int(width)))
    # de-duplicate by distance matrix hash + family
    out = []
    seen = set()
    for g in geoms:
        key = (g.family, distance_matrix_hash(g.positions))
        if key not in seen:
            seen.add(key); out.append(g)
    return out


def parameter_grid(cfg: dict):
    if "hamiltonian" in cfg:
        h = cfg["hamiltonian"]
        yield params_from_values(h.get("omega", 1.0), h.get("delta", 0.0), h.get("c6", 1.0), h.get("sign_rule", "parity"))
        return
    p = cfg.get("parameters", {})
    omegas = p.get("omega", [1.0]); deltas = p.get("delta", [0.0]); c6s = p.get("c6", [1.0]); sign = p.get("sign_rule", "parity")
    for omega in omegas:
        for delta in deltas:
            for c6 in c6s:
                yield params_from_values(omega, delta, c6, sign)


def write_status_not_run(profile: str, reason: str, config_hash: str, results_root: Path) -> None:
    ensure_result_dirs(results_root)
    row = failed_run_row(profile, "profile", "all", -1, "profile", reason, config_hash, source_hash())
    path = results_root / "raw" / "failed_runs.csv"
    if path.exists():
        old = pd.read_csv(path); df = pd.concat([old, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    atomic_write_dataframe(path, df)


def run_model_on_instance(profile: str, geometry, params, model_name: str, seed: int, cfg: dict, config_hash: str, reference=None):
    states = enumerate_binary_states(geometry.n_sites)
    H = build_sparse_rydberg(geometry.positions, params)
    if reference is None:
        reference = compute_rydberg_reference(geometry, params, dense_threshold=512)
    model = make_model(model_name, seed=seed)
    result = train_feature_model(model, states, geometry, params, H, seed=seed, maxiter=int(cfg.get("training", {}).get("maxiter", 40)), ridge=float(cfg.get("training", {}).get("ridge", 1e-8)))
    psi = variational_state(result.theta, model, states, geometry, params)
    density = density_from_state(psi, states)
    connected = connected_correlations_from_state(psi, states)
    edens_error = abs(result.energy - reference.energy) / geometry.n_sites
    corr_error = correlation_mae(connected, reference.connected)
    dens_error = float(np.mean(np.abs(density - reference.density)))
    fid = fidelity(psi, reference.vector)
    inst_id = instance_id(geometry, params)
    metrics = [
        ("energy", result.energy, "Omega"),
        ("reference_energy", reference.energy, "Omega"),
        ("energy_density_error", edens_error, "Omega/site"),
        ("correlation_mae", corr_error, "dimensionless"),
        ("density_mae", dens_error, "dimensionless"),
        ("fidelity", fid, "dimensionless"),
        ("runtime_sec", result.runtime_sec, "s"),
        ("parameter_count", model.param_count(), "count"),
        ("iterations", result.n_iterations, "count"),
        ("gradient_norm", result.gradient_norm, "dimensionless"),
    ]
    rows = [metric_row(profile, inst_id, geometry, params, cfg.get("split", "pilot"), model.name, seed, reference.method, m, v, config_hash, u) for m, v, u in metrics]
    hist_rows = []
    for h in result.history:
        hist_rows.append({
            "profile": profile, "instance_id": inst_id, "model": model.name, "seed": seed,
            "iteration": h["iteration"], "energy": h["energy"], "gradient_norm": h["gradient_norm"],
            "step_time_sec": result.runtime_sec / max(len(result.history), 1), "run_status": "completed", "config_hash": config_hash,
        })
    interp_rows = []
    extracted = extract_edge_function(model, result.theta, params, r_min=0.5, r_max=max(2.0, geometry.n_sites * geometry.spacing), n_grid=80)
    if extracted is not None:
        for i in range(len(extracted["r"])):
            interp_rows.append({
                "profile": profile, "instance_id": inst_id, "geometry": geometry.family, "n_sites": geometry.n_sites,
                "omega": params.omega, "delta": params.delta, "c6": params.c6, "split": cfg.get("split", "pilot"),
                "model": model.name, "seed": seed, "r": extracted["r"][i], "rho_omega": extracted["rho_omega"][i],
                "rho_delta": extracted["rho_delta"][i], "raw_value": extracted["raw_value"][i], "gauge_fixed_value": extracted["gauge_fixed_value"][i],
                "run_status": "completed", "code_version": __version__, "config_hash": config_hash, "source_file_hash": source_hash(),
            })
        cm = correlation_matching_score(geometry.positions, reference.connected, extracted["r"], extracted["gauge_fixed_value"])
        rows.append(metric_row(profile, inst_id, geometry, params, "pilot", model.name, seed, reference.method, "correlation_matching_spearman", cm, config_hash, "rho"))
    return rows, hist_rows, interp_rows


def append_or_write(path: Path, rows: list[dict]):
    if not rows:
        return
    df = pd.DataFrame(rows)
    if path.exists():
        df = pd.concat([pd.read_csv(path), df], ignore_index=True)
    atomic_write_dataframe(path, df)
