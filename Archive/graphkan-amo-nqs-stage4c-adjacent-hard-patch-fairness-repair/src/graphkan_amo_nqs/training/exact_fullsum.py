from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import numpy as np
import scipy.optimize

from graphkan_amo_nqs.hamiltonians.basis import sign_vector
from graphkan_amo_nqs.observables.density import probabilities_from_state
from graphkan_amo_nqs.observables.local_energy import energy_from_state, local_energy


@dataclass
class ExactTrainingResult:
    model: str
    theta: np.ndarray
    energy: float
    success: bool
    message: str
    n_iterations: int
    runtime_sec: float
    gradient_norm: float
    history: list[dict[str, Any]] = field(default_factory=list)


def wavefunction_from_features(theta: np.ndarray, X: np.ndarray, signs: np.ndarray | None = None) -> np.ndarray:
    log_amp = np.asarray(X, dtype=float) @ np.asarray(theta, dtype=float)
    log_amp = np.clip(log_amp - np.max(log_amp), -700.0, 700.0)
    amp = np.exp(log_amp)
    if signs is not None:
        amp = amp * signs
    return amp.astype(float)


def energy_and_gradient(theta: np.ndarray, X: np.ndarray, H, signs: np.ndarray | None = None, ridge: float = 0.0) -> tuple[float, np.ndarray]:
    psi = wavefunction_from_features(theta, X, signs)
    Hpsi = H.dot(psi)
    z = float(np.dot(psi, psi))
    if z <= 0:
        raise ValueError("Zero norm during exact full-sum optimisation")
    energy = float(np.dot(psi, Hpsi) / z)
    eloc = Hpsi / psi
    weights = (psi * psi) / z
    centered = eloc - energy
    grad = 2.0 * (weights * centered) @ X
    if ridge > 0:
        energy += 0.5 * ridge * float(np.dot(theta, theta))
        grad = grad + ridge * theta
    return energy, np.asarray(grad, dtype=float)


def train_feature_model(model, states: np.ndarray, geometry, params, H, seed: int = 0, maxiter: int = 80,
                        ridge: float = 1e-8, sign_rule: str | None = None) -> ExactTrainingResult:
    X = model.feature_matrix(states, geometry, params)
    theta0 = model.initial_params(seed)
    if theta0.shape[0] != X.shape[1]:
        theta0 = np.resize(theta0, X.shape[1])
    signs = sign_vector(states, sign_rule or getattr(params, "sign_rule", "none"))
    history: list[dict[str, Any]] = []
    start = perf_counter()

    def fun(theta):
        e, g = energy_and_gradient(theta, X, H, signs=signs, ridge=ridge)
        history.append({"iteration": len(history), "energy": float(e), "gradient_norm": float(np.linalg.norm(g))})
        return e, g

    opt = scipy.optimize.minimize(fun, theta0, method="L-BFGS-B", jac=True, options={"maxiter": int(maxiter), "ftol": 1e-12, "gtol": 1e-7, "maxls": 30})
    runtime = perf_counter() - start
    energy, grad = energy_and_gradient(opt.x, X, H, signs=signs, ridge=0.0)
    return ExactTrainingResult(
        model=model.name, theta=np.asarray(opt.x, dtype=float), energy=float(energy), success=bool(opt.success),
        message=str(opt.message), n_iterations=int(getattr(opt, "nit", len(history))), runtime_sec=float(runtime),
        gradient_norm=float(np.linalg.norm(grad)), history=history,
    )


def variational_state(theta: np.ndarray, model, states: np.ndarray, geometry, params, sign_rule: str | None = None) -> np.ndarray:
    X = model.feature_matrix(states, geometry, params)
    signs = sign_vector(states, sign_rule or getattr(params, "sign_rule", "none"))
    psi = wavefunction_from_features(theta, X, signs)
    return psi / np.sqrt(np.dot(psi, psi))
