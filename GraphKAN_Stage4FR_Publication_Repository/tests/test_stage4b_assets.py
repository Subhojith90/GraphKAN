from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def read_yaml(name):
    return yaml.safe_load((ROOT / "configs" / name).read_text())


def test_stage4b_configs_use_native_schema_and_exact_health_gate():
    for name in [
        "netket_stage4b_n8_component_replay.yaml",
        "netket_stage4b_n8_controls_core.yaml",
        "netket_stage4b_n8_capacity_compute.yaml",
    ]:
        cfg = read_yaml(name)
        assert cfg["stage"] == "4B"
        assert str(cfg["output_root"]).startswith("results_stage4b/")
        assert "sample_health_gate" in cfg
        assert "exact_health_gate" in cfg
        assert len(cfg["training"]["seeds"]) == 5
        assert all(spec.get("save_checkpoint") for spec in cfg["model_specs"].values())


def test_core_controls_include_requested_fairness_models():
    models = read_yaml("netket_stage4b_n8_controls_core.yaml")["model_specs"]
    for label in ["rbf_endpoint_sum_21p", "mlp_radial_21p", "mlp_endpoint_21p", "mlp_radial_64p", "mlp_endpoint_64p", "mlp_radial_128p", "mlp_endpoint_128p", "smooth_radial_21p", "smooth_endpoint_sum_21p"]:
        assert label in models


def test_component_replay_contains_all_endpoint_terms():
    models = read_yaml("netket_stage4b_n8_component_replay.yaml")["model_specs"]
    for label in ["rbf_radial_21p", "rbf_endpoint_sum_21p", "rbf_endpoint_diff_21p", "rbf_endpoint_product_21p", "rbf_endpoint_full_21p"]:
        assert label in models


def test_runner_exports_dual_health_and_native_names():
    text = (ROOT / "scripts" / "run_stage4b_controls.py").read_text()
    assert "scientific_validity_exact" in text
    assert "scientific_validity_sample" in text
    assert "stage4b_metrics.csv" in text
    assert "save_checkpoint" in text
    assert "stage3f_metrics.csv" not in text


def test_stage4b_summary_does_not_emit_legacy_singular_gate_name():
    text = (ROOT / "scripts" / "summarize_stage4b.py").read_text()
    assert "n8_joint_gate.csv" not in text
    assert "stage4b_decision_gate.csv" in text
