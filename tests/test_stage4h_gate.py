from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import analyse_stage4h as module
from stage4h_protocol import freeze, read_config, roster


def prepare(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, rescue: bool
) -> tuple[dict, Path, Path, Path]:
    cfg = read_config(Path("configs/stage4h_warmstart.yaml"))
    protocol = tmp_path / "protocol"
    freeze(cfg, protocol)
    merged = tmp_path / "merged"
    tables = merged / "tables"
    tables.mkdir(parents=True)
    (merged / "merge_summary.json").write_text(json.dumps({"validation": "PASS"}))
    metrics, attestations, regions = [], [], []
    for row in roster(cfg):
        radial_error = {"energy": 0.01, "density": 0.02, "correlation": 0.01}
        is_rescued = (
            rescue
            and row["arm"] == "endpoint_full_warmstart"
            and row["cell_tag"] in {"d1p25-c3", "d1p50-c4"}
        )
        errors = (
            radial_error
            if row["arm"] == "radial_continuation"
            else {
                "energy": 0.01,
                "density": 0.01 if is_rescued else 0.03,
                "correlation": 0.005 if is_rescued else 0.02,
            }
        )
        metrics.append(
            {
                **row,
                "status": "COMPLETED",
                "scientific_validity_exact": "VALID",
                "exact_model_energy": -10.0,
                "exact_model_abs_energy_error_to_ed": errors["energy"],
                "exact_model_density_mae_to_ed": errors["density"],
                "exact_model_connected_correlation_mae_to_ed": errors["correlation"],
            }
        )
        attestations.append({"run_id": row["run_id"], "validation": "PASS"})
        for region in ("edge", "near_edge", "bulk"):
            regions.append(
                {
                    **{k: row[k] for k in ("run_id", "arm", "seed", "delta", "c6")},
                    "region": region,
                    "exact_model_mae": errors["density"],
                }
            )
    pd.DataFrame(metrics).to_csv(tables / "metrics.csv", index=False)
    pd.DataFrame(attestations).to_csv(
        tables / "transplant_attestations.csv", index=False
    )
    pd.DataFrame(regions).to_csv(tables / "region_density.csv", index=False)
    periodic = tmp_path / "periodic.json"
    periodic.write_text(json.dumps({"validation": "PASS", "passes": 30}))
    monkeypatch.setattr(
        module, "conditioning_table", lambda cfg: pd.DataFrame([{"validation": "PASS"}])
    )
    monkeypatch.setattr(
        module,
        "gauge_tables",
        lambda merged, cfg: (
            pd.DataFrame([{"validation": "PASS"}]),
            pd.DataFrame([{"value": 0.0}]),
        ),
    )
    return cfg, merged, protocol, periodic


@pytest.mark.parametrize(
    "rescue,decision,g3,g4",
    [
        (True, "SCIENTIFIC_RESCUE_PASS", "PASS", "NOT_APPLICABLE"),
        (False, "NEGATIVE_CLOSURE_PASS", "FAIL", "PASS"),
    ],
)
def test_stage4h_gate_has_two_valid_publication_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rescue: bool,
    decision: str,
    g3: str,
    g4: str,
):
    cfg, merged, protocol, periodic = prepare(tmp_path, monkeypatch, rescue=rescue)
    result = module.analyse(merged, protocol, periodic, cfg)
    assert result["validation"] == "PASS"
    assert result["decision_class"] == decision
    assert result["G2_energy_safety"] == "PASS"
    assert result["G3_scientific_rescue"] == g3
    assert result["G4_negative_closure"] == g4
