from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from graphkan_amo_nqs.models.stage4g_flax_nqs import nongeometric_pair_features
from stage4g_gauge import minimum_norm_gauge
from stage4g_protocol import freeze, read_config, roster_rows, select_controls, validate_config, verify_freeze
from stage4g_g2b_protocol import freeze as freeze_g2b, read_amendment, validate as validate_g2b, verify as verify_g2b
from stage4g_g2c_protocol import (
    freeze as freeze_g2c,
    read_amendment as read_g2c_amendment,
    validate as validate_g2c,
    verify as verify_g2c,
)
from validate_stage4g_convergence import enrich_sentinel_metadata
from validate_stage4g_g2b import validate as validate_g2b_results
from validate_stage4g_g2c import validate as validate_g2c_results
from validate_stage4g_periodic_alias import validate as validate_periodic_alias
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
    assert result["decision_class"] == "SCIENTIFIC_GATE_PASS"
    assert result["failed_group_comparisons"] == []
    assert result["conclusion_stability_by_model"] == {
        "endpoint_full_b32": {"comparisons": 2, "passes": 2, "pass": True},
        "endpoint_hybrid_b32": {"comparisons": 2, "passes": 2, "pass": True},
    }
    assert result["periodic_collapse_maximum_exact_error_difference"] == 0.0


def test_g2c_recovery_is_frozen_without_relaxing_g2b_limits(tmp_path: Path):
    base = read_config(Path("configs/stage4g_frozen.yaml"))
    amendment = read_g2c_amendment(Path("configs/stage4g_g2c_recovery.yaml"))
    assert validate_g2c(base, amendment)["validation"] == "PASS"
    manifest = freeze_g2c(base, amendment, tmp_path)
    assert manifest["expected_target_runs"] == 25
    roster = pd.read_csv(tmp_path / "preselection_roster.csv")
    assert len(roster) == 25
    assert roster.groupby(["boundary", "delta", "c6", "model"]).ngroups == 5
    assert verify_g2c(base, amendment, tmp_path)["validation"] == "PASS"


def test_periodic_alias_gate_uses_common_parameters_and_all_states():
    result = validate_periodic_alias()
    assert result["validation"] == "PASS"
    assert result["passes"] == result["comparisons"] == 30
    assert {row["states_compared"] for row in result["rows"]} == {1024}
    assert max(row["maximum_log_amplitude_difference"] for row in result["rows"]) == 0.0


def test_g2c_composite_gate_preserves_g2b_and_closes_targeted_groups(tmp_path: Path):
    import json

    amendment = read_g2c_amendment(Path("configs/stage4g_g2c_recovery.yaml"))
    g2b, g2c = tmp_path / "g2b", tmp_path / "g2c"
    g2b.mkdir(); g2c.mkdir()
    cells = [(1.25, 3.0), (1.50, 4.0), (2.00, 5.0)]
    models = ["rbf_radial_b32", "endpoint_full_b32", "endpoint_hybrid_b32", "mlp_b12_h7"]
    group_rows = []
    for boundary in ("open", "periodic"):
        for delta, c6 in cells:
            for model in models:
                for outcome in ("energy", "density", "correlation"):
                    group_rows.append({
                        "stage4g_boundary": boundary, "delta": delta, "c6": c6,
                        "model_label": model, "outcome": outcome, "absolute_change": 0.0,
                        "maximum_absolute_change": 1.0, "tolerance_utilisation": 0.0, "pass": True,
                    })
    pd.DataFrame(group_rows).to_csv(g2b / "convergence_g2b_group_gate.csv", index=False)
    repair_rows = []
    for delta, c6 in cells[:2]:
        for seed in range(5):
            repair_rows.append({
                "stage4g_boundary": "open", "delta": delta, "c6": c6, "seed": seed,
                "step": 4200, "model": "endpoint_full_b32", "repair_energy": 0.0,
                "repair_density": 0.01, "repair_correlation": 0.005,
            })
    pd.DataFrame(repair_rows).to_csv(g2b / "convergence_g2b_endpoint_repairs.csv", index=False)
    (g2b / "convergence_g2b_gate.json").write_text(json.dumps({"validation": "FAIL"}))

    metric_frames, sentinel_frames = {"open": [], "periodic": []}, {"open": [], "periodic": []}
    for target in amendment["execution"]["targets"]:
        boundary = target["boundary"]
        for seed in range(5):
            run_id = f"{target['tag']}:{seed}"
            metric_frames[boundary].append({
                "run_id": run_id, "scientific_validity_exact": "VALID", "model_label": target["model"],
                "delta": target["delta"], "c6": target["c6"], "seed": seed,
            })
            for step in (5600, 6300, 7000):
                density = 0.01 if target["model"] == "endpoint_full_b32" else 0.02
                corr = 0.005 if target["model"] == "endpoint_full_b32" else 0.01
                sentinel_frames[boundary].append({
                    "run_id": run_id, "model_label": target["model"], "delta": target["delta"],
                    "c6": target["c6"], "seed": seed, "step": step,
                    "exact_model_abs_energy_error_to_ed": 0.1,
                    "exact_model_density_mae_to_ed": density,
                    "exact_model_connected_correlation_mae_to_ed": corr,
                })
    for boundary in ("open", "periodic"):
        raw = g2c / boundary / "raw"; raw.mkdir(parents=True)
        pd.DataFrame(metric_frames[boundary]).to_csv(raw / "stage4g_metrics.csv", index=False)
        pd.DataFrame(sentinel_frames[boundary]).to_csv(raw / "stage4g_exact_sentinels.csv", index=False)
    alias = tmp_path / "alias.json"
    alias.write_text(json.dumps({"validation": "PASS", "passes": 30}))
    result = validate_g2c_results(amendment, g2b, g2c, alias)
    assert result["validation"] == "PASS"
    assert result["g2b_preserved_validation"] == "FAIL"
    assert result["target_group_passes"] == 15
    assert result["composite_group_passes"] == 72
    assert result["endpoint_full_open_cells_passed"] == 3
