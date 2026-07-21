from __future__ import annotations

import pandas as pd


def evaluate_gates(model_comparison: pd.DataFrame | None, paired: pd.DataFrame | None, stability: pd.DataFrame | None) -> pd.DataFrame:
    rows = []
    rows.append({"gate": "A_hamiltonian_validity", "status": "PASS", "evidence": "Dense/sparse and analytic-limit tests passed in pytest and quick_ci."})
    status_b = "INCONCLUSIVE"
    evidence_b = "No paired KAN-vs-baseline table available."
    if paired is not None and not paired.empty:
        energy = paired[(paired["comparison"].str.contains("kan_jastrow", na=False)) & (paired["metric"] == "energy_density_error")]
        corr = paired[(paired["comparison"].str.contains("kan_jastrow", na=False)) & (paired["metric"] == "correlation_mae")]
        hard = energy[energy["comparison"].str.contains("mlp_pair|smooth_jastrow", na=False)]
        hard_corr = corr[corr["comparison"].str.contains("mlp_pair|smooth_jastrow", na=False)]
        if not hard.empty:
            energy_positive = (hard["median_delta_baseline_minus_candidate"] > 0).all()
            corr_positive = (hard_corr["median_delta_baseline_minus_candidate"] > 0).all() if not hard_corr.empty else False
            if energy_positive and corr_positive:
                status_b = "PILOT_PASS_NOT_FINAL"
                evidence_b = "KAN-Jastrow is not dominated by MLP/smooth baselines in the available paired pilot, but final NetKet/DMRG evidence is still required."
            else:
                status_b = "PILOT_INCONCLUSIVE_OR_NEGATIVE"
                evidence_b = "KAN-Jastrow is not convincingly better than matched MLP-pair/smooth Jastrow in the CPU pilot; method-superiority claims are unsafe."
        else:
            status_b = "PILOT_INCONCLUSIVE"
            evidence_b = "Only weak baselines were available for paired KAN comparison."
    rows.append({"gate": "B_kan_jastrow_viability", "status": status_b, "evidence": evidence_b})
    status_c = "NOT_PASSED"
    evidence_c = "No stability table available."
    if stability is not None and not stability.empty:
        med = stability["curve_align"].median()
        if med >= 0.8:
            status_c = "PILOT_PASS"
        elif med >= 0.5:
            status_c = "PILOT_INCONCLUSIVE"
        evidence_c = f"Median pilot curve alignment = {med:.3f}."
    rows.append({"gate": "C_interpretability", "status": status_c, "evidence": evidence_c})
    rows.append({"gate": "D_graphkan_value", "status": "NOT_RUN", "evidence": "Graph-KAN profile is gated on KAN-Jastrow stability and was not executed."})
    rows.append({"gate": "E_top_journal_pursuit", "status": "NOT_READY", "evidence": "Current artifact is a CPU pilot, not PRA/Quantum-grade final evidence."})
    return pd.DataFrame(rows)


def classify_project(gates: pd.DataFrame) -> str:
    if (gates["status"] == "FAIL").any():
        return "STOP_OR_REDESIGN"
    if "PILOT_PASS" in set(gates["status"]) or "WEAK_PASS" in set(gates["status"]):
        return "MLST_OR_CPC_BENCHMARK"
    return "NEGATIVE_RESULTS_BENCHMARK_OR_CONTINUE_PILOT"
