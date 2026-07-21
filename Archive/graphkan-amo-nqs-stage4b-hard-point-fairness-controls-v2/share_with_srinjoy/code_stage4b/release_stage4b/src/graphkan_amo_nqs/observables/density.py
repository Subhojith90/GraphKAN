from __future__ import annotations

import numpy as np


def probabilities_from_state(vector: np.ndarray) -> np.ndarray:
    v = np.asarray(vector)
    p = np.abs(v) ** 2
    z = float(np.sum(p))
    if z <= 0:
        raise ValueError("State vector has zero norm")
    return p / z


def density_from_probabilities(probabilities: np.ndarray, states: np.ndarray) -> np.ndarray:
    return np.asarray(probabilities, dtype=float) @ np.asarray(states, dtype=float)


def density_from_state(vector: np.ndarray, states: np.ndarray) -> np.ndarray:
    return density_from_probabilities(probabilities_from_state(vector), states)
