from __future__ import annotations

import numpy as np

from graphkan_amo_nqs.models.gauge import gauge_fix_curve


def extract_edge_function(model, theta: np.ndarray, params, r_min: float = 0.5, r_max: float = 5.0, n_grid: int = 100) -> dict[str, np.ndarray] | None:
    r = np.linspace(r_min, r_max, n_grid)
    raw = model.edge_function(theta, r, params)
    if raw is None:
        return None
    fixed = gauge_fix_curve(r, raw)
    rho_o = r / params.r_omega if np.isfinite(params.r_omega) and params.r_omega != 0 else r
    rho_d = r / params.r_delta if np.isfinite(params.r_delta) and params.r_delta != 0 else r
    return {"r": r, "rho_omega": rho_o, "rho_delta": rho_d, "raw_value": raw, "gauge_fixed_value": fixed}
