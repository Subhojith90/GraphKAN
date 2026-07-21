import importlib.util
from pathlib import Path
import subprocess
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "summarize_stage4c_repaired.py"


def test_repaired_aggregator_pairs_within_cell_and_fails_below_threshold(tmp_path):
    results = tmp_path / "results"
    raw_dir = results / "raw"
    raw_dir.mkdir(parents=True)

    rows = [
        # Cell A: candidate is better; not dominated.
        {
            "run_id": "a_candidate", "status": "COMPLETED",
            "model_label": "candidate", "seed": 0,
            "n_sites": 6, "delta": 1.5, "c6": 3.0,
            "scientific_validity_exact": "VALID",
            "scientific_validity_sample": "VALID",
            "exact_model_abs_energy_error_to_ed": 0.10,
            "exact_model_density_mae_to_ed": 0.10,
            "exact_model_connected_correlation_mae_to_ed": 0.10,
            "wall_seconds": 10.0,
        },
        {
            "run_id": "a_control", "status": "COMPLETED",
            "model_label": "control", "seed": 0,
            "n_sites": 6, "delta": 1.5, "c6": 3.0,
            "scientific_validity_exact": "VALID",
            "scientific_validity_sample": "VALID",
            "exact_model_abs_energy_error_to_ed": 0.20,
            "exact_model_density_mae_to_ed": 0.20,
            "exact_model_connected_correlation_mae_to_ed": 0.20,
            "wall_seconds": 10.0,
        },
        # Cell B: control dominates candidate.
        {
            "run_id": "b_candidate", "status": "COMPLETED",
            "model_label": "candidate", "seed": 0,
            "n_sites": 8, "delta": 2.0, "c6": 4.0,
            "scientific_validity_exact": "VALID",
            "scientific_validity_sample": "VALID",
            "exact_model_abs_energy_error_to_ed": 0.30,
            "exact_model_density_mae_to_ed": 0.30,
            "exact_model_connected_correlation_mae_to_ed": 0.30,
            "wall_seconds": 10.0,
        },
        {
            "run_id": "b_control", "status": "COMPLETED",
            "model_label": "control", "seed": 0,
            "n_sites": 8, "delta": 2.0, "c6": 4.0,
            "scientific_validity_exact": "VALID",
            "scientific_validity_sample": "VALID",
            "exact_model_abs_energy_error_to_ed": 0.20,
            "exact_model_density_mae_to_ed": 0.20,
            "exact_model_connected_correlation_mae_to_ed": 0.20,
            "wall_seconds": 10.0,
        },
    ]

    pd.DataFrame(rows).to_csv(raw_dir / "stage4c_metrics.csv", index=False)

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--results", str(results),
            "--candidate", "candidate",
            "--min-candidate-exact-valid-rate", "0.90",
            "--min-non-dominated-cell-rate", "0.70",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    out = results / "aggregated_repaired"
    gate = pd.read_csv(out / "stage4c_decision_gate_repaired.csv").iloc[0]
    cells = pd.read_csv(out / "stage4c_cell_gate.csv")

    assert gate["pairing_key"] == "n_sites,delta,c6,seed"
    assert gate["stage4c_status"] == "FAIL"
    assert len(cells) == 2
    assert cells["candidate_non_dominated"].sum() == 1

    dominated = cells[
        (cells["n_sites"] == 8)
        & (cells["delta"] == 2.0)
        & (cells["c6"] == 4.0)
    ].iloc[0]

    assert bool(dominated["candidate_non_dominated"]) is False
    assert dominated["dominating_controls"] == "control"


def test_repaired_script_contains_full_cell_seed_pairing():
    text = SCRIPT.read_text()
    assert 'CELL = ["n_sites", "delta", "c6"]' in text
    assert 'KEY = CELL + ["seed"]' in text
    assert 'candidate_rate>=args.min_candidate_exact_valid_rate' in text
    assert 'Incomplete patch cells present' in text
