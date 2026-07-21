from __future__ import annotations


def select_by_validation(rows, metric: str = "energy_density_error"):
    candidates = [r for r in rows if r.get("split") == "validation" and r.get("metric") == metric]
    if not candidates:
        return None
    return min(candidates, key=lambda r: r.get("value", float("inf")))
