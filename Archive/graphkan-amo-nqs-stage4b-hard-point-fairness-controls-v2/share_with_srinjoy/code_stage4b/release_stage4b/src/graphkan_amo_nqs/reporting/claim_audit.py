from __future__ import annotations

import pandas as pd


def validate_claims(results_root: str = "results") -> pd.DataFrame:
    rows = [
        {"claim": "Hamiltonian validity", "status": "allowed", "required_evidence": "ED validation CSV and tests", "current_status": "check results/references/ed_validation.csv"},
        {"claim": "KAN-Jastrow pilot competitiveness", "status": "conditional", "required_evidence": "paired comparisons with CIs", "current_status": "pilot-only"},
        {"claim": "Graph-KAN value", "status": "not_allowed", "required_evidence": "fair graph-MPNN comparison", "current_status": "not run"},
        {"claim": "top-journal readiness", "status": "not_allowed", "required_evidence": "PRA-core/DMRG/NetKet baselines", "current_status": "not ready"},
        {"claim": "quantum advantage", "status": "forbidden", "required_evidence": "not in scope", "current_status": "forbidden"},
    ]
    return pd.DataFrame(rows)
