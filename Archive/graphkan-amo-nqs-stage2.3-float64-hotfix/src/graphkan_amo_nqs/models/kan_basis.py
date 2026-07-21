from __future__ import annotations

import numpy as np


def safe_scaled_distance(r: np.ndarray, scale: float) -> np.ndarray:
    if not np.isfinite(scale) or scale == 0:
        return np.asarray(r, dtype=float)
    return np.asarray(r, dtype=float) / float(scale)


def basis_matrix(x: np.ndarray, family: str = "sine", n_basis: int = 8) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    if n_basis <= 0:
        raise ValueError("n_basis must be positive")
    family = family.lower()
    if family in {"sine", "sinusoidal", "sinekan"}:
        cols = [np.sin((k + 1) * np.pi * x) for k in range(n_basis)]
    elif family in {"chebyshev", "cheb"}:
        z = np.clip(2 * (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x) + 1e-12) - 1, -1, 1)
        cols = [np.cos(k * np.arccos(z)) for k in range(n_basis)]
    elif family in {"rbf", "radial-basis"}:
        centres = np.linspace(float(np.nanmin(x)), float(np.nanmax(x) + 1e-12), n_basis)
        width = max(float(np.nanmax(x) - np.nanmin(x)), 1.0) / max(n_basis - 1, 1)
        cols = [np.exp(-0.5 * ((x - c) / (width + 1e-12)) ** 2) for c in centres]
    elif family in {"linear_spline", "linear-spline", "bspline", "b-spline"}:
        knots = np.linspace(float(np.nanmin(x)), float(np.nanmax(x) + 1e-12), n_basis)
        width = max(float(np.nanmax(x) - np.nanmin(x)), 1.0) / max(n_basis - 1, 1)
        cols = [np.maximum(1.0 - np.abs(x - c) / (width + 1e-12), 0.0) for c in knots]
    elif family == "poly":
        cols = [x ** k for k in range(n_basis)]
    else:
        raise ValueError(f"Unknown KAN basis family: {family}")
    return np.stack(cols, axis=1).astype(float)


def normalise_columns(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + eps)
