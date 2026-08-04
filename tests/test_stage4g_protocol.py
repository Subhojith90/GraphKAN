from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from graphkan_amo_nqs.models.stage4g_flax_nqs import nongeometric_pair_features
from stage4g_gauge import minimum_norm_gauge
from stage4g_protocol import freeze, read_config, roster_rows, select_controls, validate_config, verify_freeze
from stage4g_g2b_protocol import freeze as freeze_g2b, read_amendment, validate as validate_g2b, verify as verify_g2b
from validate_stage4g_convergence import enrich_sentinel_metadata
from validate_stage4g_g2b import validate as validate_g2b_results
from run_stage4g import remove_run_rows
from stage4g_merge_shards import merge, resolve_protocol_directory


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
    assert "predecessor_run_id:" in text
    assert "--phase convergence_g2b" in text
    assert "stage4g-convergence-g2b-merged" in text
    assert "g2b_artifact_run_id:" in text
    assert "run-id: ${{ inputs.g2b_artifact_run_id }}" in text


def test_shard_merge_prefers_the_amended_protocol_directory(tmp_path: Path):
    amended = tmp_path / "g2b_protocol"
    amended.mkdir()
    (amended / "preselection_roster.csv").write_text("phase,run_id\n")
    original = tmp_path / "protocol"
    original.mkdir()
    (original / "preselection_roster.csv").write_text("phase,run_id\n")
    assert resolve_protocol_directory(tmp_path) == amended


def test_g2b_amendment_is_bound_to_predecessor_and_freezes_120_fresh_runs(tmp_path: Path):
    base = read_config(Path("configs/stage4g_frozen.yaml"))
    amendment = read_amendment(Path("configs/stage4g_g2b_amendment.yaml"))
    assert validate_g2b(base, amendment)["validation"] == "PASS"
    manifest = freeze_g2b(base, amendment, tmp_path)
    assert manifest["expected_g2b_runs"] == 120
    config = read_config(tmp_path / "runner_configs" / "convergence_g2b_open.yaml")
    assert config["training"]["n_iter"] == 4200
    assert config["training"]["exact_sentinel_steps"] == [2800, 3500, 4200]
    roster = pd.read_csv(tmp_path / "preselection_roster.csv")
    assert len(roster[roster.phase == "convergence_g2b"]) == 120
    assert verify_g2b(base, amendment, tmp_path)["validation"] == "PASS"


def test_legacy_sentinel_metadata_is_reconstructed_from_metrics():
    metrics = pd.DataFrame([
        {"run_id": "one", "delta": 1.25, "c6": 3.0, "boundary_condition": "open", "n_sites": 10}
    ])
    sentinels = pd.DataFrame([{"run_id": "one", "step": 700}])
    enriched = enrich_sentinel_metadata(sentinels, metrics)
    assert enriched.loc[0, "delta"] == 1.25
    assert enriched.loc[0, "c6"] == 3.0


def test_g2b_gate_checks_all_groups_conclusion_stability_and_periodic_collapse(tmp_path: Path):
    amendment = read_amendment(Path("configs/stage4g_g2b_amendment.yaml"))
    models = ["rbf_radial_b32", "endpoint_full_b32", "endpoint_hybrid_b32", "mlp_b12_h7"]
    cells = [(1.25, 3.0), (1.50, 4.0), (2.00, 5.0)]
    for boundary in ("open", "periodic"):
        metrics, sentinels = [], []
        for delta, c6 in cells:
            for model in models:
                for seed in range(5):
                    run_id = f"{boundary}:{delta}:{c6}:{model}:{seed}"
                    metrics.append({
                        "run_id": run_id, "status": "COMPLETED", "scientific_validity_exact": "VALID",
                        "boundary_condition": boundary, "delta": delta, "c6": c6,
                        "model_label": model, "seed": seed, "n_sites": 10,
                    })
                    endpoint_repair = boundary == "open" and model.startswith("endpoint_")
                    for step in (2800, 3500, 4200):
                        sentinels.append({
                            "run_id": run_id, "model_label": model, "seed": seed, "step": step,
                            "exact_model_abs_energy_error_to_ed": 0.1,
                            "exact_model_density_mae_to_ed": 0.01 if endpoint_repair else 0.02,
                            "exact_model_connected_correlation_mae_to_ed": 0.005 if endpoint_repair else 0.01,
                        })
        raw = tmp_path / "convergence_g2b" / boundary / "raw"
        raw.mkdir(parents=True)
        pd.DataFrame(metrics).to_csv(raw / "stage4g_metrics.csv", index=False)
        pd.DataFrame(sentinels).to_csv(raw / "stage4g_exact_sentinels.csv", index=False)
    result = validate_g2b_results(amendment, tmp_path)
    assert result["validation"] == "PASS"
    assert result["group_passes"] == 72
    assert result["conclusion_stability_passes"] == 4
    assert result["periodic_collapse_passes"] == 90
