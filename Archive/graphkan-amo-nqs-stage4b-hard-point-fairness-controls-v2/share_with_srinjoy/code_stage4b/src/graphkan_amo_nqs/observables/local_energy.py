from __future__ import annotations

import numpy as np


def local_energy(matrix, psi: np.ndarray) -> np.ndarray:
    psi = np.asarray(psi, dtype=float)
    if np.any(np.abs(psi) < 1e-300):
        raise ValueError("Wavefunction contains near-zero amplitudes; local energy is ill-conditioned")
    return matrix.dot(psi) / psi


def energy_from_state(matrix, psi: np.ndarray) -> float:
    psi = np.asarray(psi, dtype=float)
    z = float(np.dot(psi, psi))
    if z <= 0:
        raise ValueError("Zero norm wavefunction")
    return float(np.dot(psi, matrix.dot(psi)) / z)
