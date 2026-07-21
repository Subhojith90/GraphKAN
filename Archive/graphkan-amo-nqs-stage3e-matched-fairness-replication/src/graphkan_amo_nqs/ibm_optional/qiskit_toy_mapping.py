from __future__ import annotations

from graphkan_amo_nqs.exceptions import DependencyUnavailable


def ibm_sanity_status() -> dict:
    try:
        import qiskit  # type: ignore
        return {"available": True, "qiskit_version": getattr(qiskit, "__version__", "unknown"), "used_for_main_claims": False}
    except Exception:
        return {"available": False, "qiskit_version": None, "used_for_main_claims": False, "reason": "Qiskit unavailable; IBM optional sanity not run."}
