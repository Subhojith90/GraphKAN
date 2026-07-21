
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def test_package_imports_from_complete_src_tree():
    import graphkan_amo_nqs
    assert graphkan_amo_nqs is not None

def test_endpoint_summary_exists_and_nonempty():
    path = ROOT / "analysis" / "stage4e_endpoint_feature_summary.csv"
    assert path.exists()
    df = pd.read_csv(path)
    assert len(df) > 0
    assert {"open", "periodic"}.issubset(set(df["boundary_batch"]))

def test_open_endpoint_product_improves_matched_radial_exact_metrics():
    path = ROOT / "evidence" / "stage4e1" / "n10_open_boundary_paired5" / "raw" / "stage4e_metrics.csv"
    df = pd.read_csv(path)
    pairs = [
        ("rbf_radial_b24", "endpoint_product_b24"),
        ("rbf_radial_b32", "endpoint_product_b32"),
    ]
    metrics = [
        "exact_model_abs_energy_error_to_ed",
        "exact_model_density_mae_to_ed",
        "exact_model_connected_correlation_mae_to_ed",
    ]
    for radial, endpoint in pairs:
        left = df[df["model_label"] == radial].sort_values("seed").set_index("seed")
        right = df[df["model_label"] == endpoint].sort_values("seed").set_index("seed")
        common = sorted(set(left.index).intersection(right.index))
        assert common == [0, 1, 2, 3, 4]
        for metric in metrics:
            delta = right.loc[common, metric].to_numpy() - left.loc[common, metric].to_numpy()
            assert (delta < 0).all(), (radial, endpoint, metric, delta)

def test_periodic_endpoint_collapse_matches_radial_exact_metrics_and_params():
    path = ROOT / "evidence" / "stage4e1" / "n10_periodic_boundary_paired5" / "raw" / "stage4e_metrics.csv"
    df = pd.read_csv(path)
    pairs = [
        ("rbf_radial_b24", "endpoint_sum_b24"),
        ("rbf_radial_b24", "endpoint_product_b24"),
        ("rbf_radial_b32", "endpoint_sum_b32"),
        ("rbf_radial_b32", "endpoint_product_b32"),
    ]
    metrics = [
        "exact_model_abs_energy_error_to_ed",
        "exact_model_density_mae_to_ed",
        "exact_model_connected_correlation_mae_to_ed",
    ]
    for radial, endpoint in pairs:
        left = df[df["model_label"] == radial].sort_values("seed").set_index("seed")
        right = df[df["model_label"] == endpoint].sort_values("seed").set_index("seed")
        common = sorted(set(left.index).intersection(right.index))
        assert common == [0, 1, 2, 3, 4]
        assert sorted(left.loc[common, "n_parameters"].unique()) == sorted(right.loc[common, "n_parameters"].unique())
        for metric in metrics:
            diff = left.loc[common, metric].to_numpy() - right.loc[common, metric].to_numpy()
            assert np.max(np.abs(diff)) == 0.0, (radial, endpoint, metric, diff)
