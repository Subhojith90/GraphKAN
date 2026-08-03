from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from graphkan_amo_nqs.models.stage4g_flax_nqs import nongeometric_pair_features
from stage4g_gauge import minimum_norm_gauge
from stage4g_protocol import freeze, read_config, roster_rows, select_controls, validate_config, verify_freeze
from run_stage4g import remove_run_rows
from stage4g_merge_shards import merge


def test_nongeometric_capacity_features_are_frozen_centered_and_orthogonal():
    i, j = np.triu_indices(12, k=1)
    first = nongeometric_pair_features(i, j)
    second = nongeometric_pair_features(i, j)
    assert first.shape == (66, 3)
    assert np.array_equal(first, second)
    assert np.allclose(first.mean(axis=0), 0.0, atol=1e-15)
    assert np.allclose((first.T @ first) / len(first), np.eye(3), atol=1e-14)


def test_minimum_norm_gauge_preserves_design_and_removes_null_component():
    design = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    theta = np.array([2.0, -1.0, 4.0])
    projected, diagnostic = minimum_norm_gauge(design, theta)
    assert diagnostic["rank"] == 2
    assert diagnostic["nullity"] == 1
    assert np.allclose(design @ projected, design @ theta)
    assert np.linalg.norm(projected) < np.linalg.norm(theta)


def test_stage4g_freeze_has_disjoint_cells_and_expected_rosters(tmp_path: Path):
    cfg = read_config(Path("configs/stage4g_frozen.yaml"))
    assert validate_config(cfg)["validation"] == "PASS"
    result = freeze(cfg, tmp_path)
    assert result["expected_convergence_runs"] == 120
    assert result["expected_calibration_runs"] == 72
    assert result["expected_confirmation_runs_after_selection"] == 180
    roster = pd.read_csv(tmp_path / "preselection_roster.csv")
    assert len(roster) == 192
    assert set(roster.phase) == {"convergence", "calibration"}
    assert verify_freeze(cfg, tmp_path)["validation"] == "PASS"


def test_control_selection_is_calibration_only_and_materialises_confirmation(tmp_path: Path):
    cfg = read_config(Path("configs/stage4g_frozen.yaml"))
    freeze(cfg, tmp_path)
    candidates = cfg["models"]["control_candidates"]
    rows = roster_rows(
        "calibration", cfg["physics"]["calibration_cells"], cfg["physics"]["boundaries"],
        list(candidates), cfg["physics"]["calibration_seeds"], cfg["physics"]["calibration_n_sites"],
    )
    preferred = {"mlp_b12_h7", "rbm_h32"}
    metrics = []
    for index, row in enumerate(rows):
        base = 0.01 if row["model"] in preferred else 0.1
        metrics.append({
            "run_id": row["run_id"], "boundary_condition": row["boundary"],
            "delta": row["delta"], "c6": row["c6"], "model_label": row["model"], "seed": row["seed"],
            "n_parameters": 100 + index % 7,
            "exact_model_abs_energy_error_to_ed": base,
            "exact_model_density_mae_to_ed": base,
            "exact_model_connected_correlation_mae_to_ed": base,
        })
    path = tmp_path / "calibration.csv"
    pd.DataFrame(metrics).to_csv(path, index=False)
    decision = select_controls(cfg, tmp_path, [path])
    assert decision["selected"] == {"mlp": "mlp_b12_h7", "rbm": "rbm_h32"}
    confirmation = pd.read_csv(tmp_path / "confirmation_roster.csv")
    assert len(confirmation) == 180
    assert set(confirmation.model) == {
        "rbf_radial_b32", "endpoint_full_b32", "endpoint_hybrid_b32",
        "nongeometric_b32_p3", "mlp_b12_h7", "rbm_h32",
    }


def test_resume_cleanup_removes_only_the_interrupted_run(tmp_path: Path):
    path = tmp_path / "history.csv"
    pd.DataFrame([
        {"run_id": "complete", "step": 1},
        {"run_id": "interrupted", "step": 1},
        {"run_id": "interrupted", "step": 2},
    ]).to_csv(path, index=False)
    remove_run_rows(path, "interrupted")
    remaining = pd.read_csv(path)
    assert remaining.to_dict(orient="records") == [{"run_id": "complete", "step": 1}]


def test_github_shards_merge_to_the_complete_convergence_roster(tmp_path: Path):
    cfg = read_config(Path("configs/stage4g_frozen.yaml"))
    results = tmp_path / "results_stage4g"
    freeze(cfg, results / "protocol")
    roster = pd.read_csv(results / "protocol" / "preselection_roster.csv")
    roster = roster[roster.phase == "convergence"]
    downloaded = tmp_path / "downloaded"; downloaded.mkdir()
    for (boundary, model), group in roster.groupby(["boundary", "model"]):
        raw = downloaded / f"stage4g-convergence-{boundary}-{model}" / "raw"
        raw.mkdir(parents=True)
        rows = [{
            "run_id": row.run_id, "status": "COMPLETED", "boundary_condition": boundary,
            "delta": row.delta, "c6": row.c6, "model_label": model, "seed": row.seed,
        } for row in group.itertuples(index=False)]
        pd.DataFrame(rows).to_csv(raw / "stage4g_metrics.csv", index=False)
        for filename in (
            "stage4g_history.csv", "stage4g_site_observables.csv", "stage4g_corr_observables.csv",
            "stage4g_exact_sentinels.csv",
        ):
            pd.DataFrame(rows[:1]).to_csv(raw / filename, index=False)
    summary = merge(downloaded, results, "convergence")
    assert summary["validation"] == "PASS"
    assert summary["artifacts"] == 8
    assert summary["merged_runs"] == 120


def test_stage4g_workflow_uses_pinned_actions_and_dynamic_confirmation_matrix():
    text = Path(".github/workflows/stage4g-confirmation.yml").read_text()
    assert "actions/checkout@11d5960a326750d5838078e36cf38b85af677262" in text
    assert "actions/download-artifact@d3f86a106a0bac45b974a628896c90dbdf5c8093" in text
    assert "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02" in text
    assert "matrix: ${{ fromJSON(needs.select_controls.outputs.matrix) }}" in text
    assert "cancel-in-progress: false" in text
