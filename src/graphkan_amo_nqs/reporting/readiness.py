from __future__ import annotations

import pandas as pd


def readiness_summary(gates: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([
        {"venue": "Physical Review A", "status": "not ready", "missing": "PRA-core Rydberg results, DMRG/strong baselines, final seed count"},
        {"venue": "Physical Review Research", "status": "not ready", "missing": "broader benchmark and robustness"},
        {"venue": "Machine Learning: Science and Technology", "status": "pilot artifact only", "missing": "software benchmark hardening and full ablations"},
        {"venue": "Computer Physics Communications", "status": "not ready", "missing": "stable public API, performance docs, full examples"},
        {"venue": "Quantum / PRX Quantum", "status": "not candidate", "missing": "broad NQS principle or new physical diagnostic"},
    ])
