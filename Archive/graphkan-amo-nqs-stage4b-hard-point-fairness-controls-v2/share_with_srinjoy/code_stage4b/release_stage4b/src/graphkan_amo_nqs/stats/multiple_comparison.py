from __future__ import annotations


def holm_correction(p_values: list[float]) -> list[float]:
    indexed = sorted([(p, i) for i, p in enumerate(p_values) if p == p])
    out = [float("nan")] * len(p_values)
    m = len(indexed)
    running = 0.0
    for rank, (p, i) in enumerate(indexed):
        adjusted = min(1.0, (m - rank) * p)
        running = max(running, adjusted)
        out[i] = running
    return out
