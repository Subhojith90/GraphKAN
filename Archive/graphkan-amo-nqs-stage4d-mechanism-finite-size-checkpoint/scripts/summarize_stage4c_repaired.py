#!/usr/bin/env python3
"""Audit-safe Stage 4C aggregation repair.
Pairs only within (n_sites, delta, c6, seed); never pairs same seed across cells.
Does not alter raw VMC data.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

EXACT = ["exact_model_abs_energy_error_to_ed", "exact_model_density_mae_to_ed", "exact_model_connected_correlation_mae_to_ed"]
CELL = ["n_sites", "delta", "c6"]
KEY = CELL + ["seed"]
TOL = 1e-12

def compare_cell(cell: pd.DataFrame, candidate: str, control: str) -> dict:
    c = cell[cell.model_label == candidate].set_index("seed")
    b = cell[cell.model_label == control].set_index("seed")
    common = c.index.intersection(b.index)
    base = {"pairs": int(len(common))}
    if len(common) == 0:
        return base | {"status":"NOT_AVAILABLE"}
    c, b = c.loc[common], b.loc[common]
    cand_valid = bool((c.scientific_validity_exact == "VALID").all())
    ctrl_valid = bool((b.scientific_validity_exact == "VALID").all())
    ds = {name: c[col] - b[col] for name, col in zip(["energy","density","corr"], EXACT)}
    rt = b.wall_seconds / c.wall_seconds.replace(0, np.nan)
    # Control dominates only when valid, no worse on all exact metrics seed-by-seed,
    # strictly better at least once, and within stated compute tolerance.
    dominates = (cand_valid and ctrl_valid and
                 all((d >= -TOL).all() for d in ds.values()) and
                 any((d > TOL).any() for d in ds.values()) and
                 bool((rt <= 1.25 + TOL).all()))
    return base | {
        "candidate_exact_valid_all": cand_valid,
        "control_exact_valid_all": ctrl_valid,
        "control_over_candidate_runtime_ratio_median": float(rt.median()),
        "median_candidate_minus_energy": float(ds["energy"].median()),
        "median_candidate_minus_density": float(ds["density"].median()),
        "median_candidate_minus_corr": float(ds["corr"].median()),
        "control_dominates_candidate": bool(dominates),
        "status": "DOMINATED" if dominates else "NOT_DOMINATED",
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--candidate", default="rbf_endpoint_sum_21p")
    ap.add_argument("--min-candidate-exact-valid-rate", type=float, default=0.90)
    ap.add_argument("--min-non-dominated-cell-rate", type=float, default=0.70)
    args = ap.parse_args()
    root = Path(args.results)
    raw = root / "raw" / "stage4c_metrics.csv"
    out = root / "aggregated_repaired"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(raw)
    df = df[df.status == "COMPLETED"].copy()
    if df.empty: raise SystemExit("No COMPLETED rows.")
    expected = int(df.groupby(CELL).size().mode().iloc[0])
    counts = df.groupby(CELL).size().rename("rows").reset_index()
    counts["complete"] = counts.rows.eq(expected)
    counts.to_csv(out / "patch_cell_inventory.csv", index=False)
    if not counts.complete.all():
        raise SystemExit("Incomplete patch cells present; aggregation stopped.")
    labels = sorted(df.model_label.unique())
    if args.candidate not in labels: raise SystemExit("Candidate absent.")
    controls = [x for x in labels if x != args.candidate]
    per_model = df.groupby(CELL + ["model_label"], dropna=False).agg(
        runs=("run_id","size"), exact_valid=("scientific_validity_exact",lambda s:int((s=="VALID").sum())),
        sample_valid=("scientific_validity_sample",lambda s:int((s=="VALID").sum())),
        median_energy=(EXACT[0],"median"), median_density=(EXACT[1],"median"),
        median_corr=(EXACT[2],"median"), median_runtime=("wall_seconds","median")).reset_index()
    per_model["exact_valid_rate"] = per_model.exact_valid/per_model.runs
    per_model["sample_valid_rate"] = per_model.sample_valid/per_model.runs
    per_model.to_csv(out/"patch_cell_model_summary.csv",index=False)
    rows=[]
    for cell_key, cell in df.groupby(CELL, sort=True):
        for control in controls:
            r = dict(zip(CELL, cell_key)) | {"candidate":args.candidate,"control":control}
            rows.append(r | compare_cell(cell,args.candidate,control))
    comp=pd.DataFrame(rows)
    comp.to_csv(out/"per_cell_paired_non_domination.csv",index=False)
    candidate_cells = per_model[per_model.model_label==args.candidate].copy()
    candidate_cells["candidate_cell_exact_valid"] = candidate_cells.exact_valid_rate >= args.min_candidate_exact_valid_rate
    cell_gate=[]
    for cell_key, row in candidate_cells.set_index(CELL).iterrows():
        subset=comp[(comp.n_sites==cell_key[0])&(comp.delta==cell_key[1])&(comp.c6==cell_key[2])]
        dominated=bool((subset.status=="DOMINATED").any())
        cell_gate.append(dict(zip(CELL,cell_key)) | {
            "candidate_exact_valid_rate":float(row.exact_valid_rate),
            "candidate_sample_valid_rate":float(row.sample_valid_rate),
            "candidate_cell_exact_valid":bool(row.candidate_cell_exact_valid),
            "candidate_non_dominated":not dominated,
            "dominating_controls":";".join(subset.loc[subset.status=="DOMINATED","control"].tolist())})
    cell_gate=pd.DataFrame(cell_gate)
    cell_gate.to_csv(out/"stage4c_cell_gate.csv",index=False)
    valid_cells=cell_gate[cell_gate.candidate_cell_exact_valid]
    candidate_rate=float((df[df.model_label==args.candidate].scientific_validity_exact=="VALID").mean())
    nd_rate=float(valid_cells.candidate_non_dominated.mean()) if len(valid_cells) else 0.0
    status="PASS" if candidate_rate>=args.min_candidate_exact_valid_rate and nd_rate>=args.min_non_dominated_cell_rate else "FAIL"
    gate={"candidate":args.candidate,"expected_rows_per_cell":expected,"n_patch_cells":int(len(cell_gate)),
          "candidate_exact_valid_rate":candidate_rate,"candidate_sample_valid_rate":float((df[df.model_label==args.candidate].scientific_validity_sample=="VALID").mean()),
          "exact_valid_patch_cells":int(len(valid_cells)),"non_dominated_exact_valid_cells":int(valid_cells.candidate_non_dominated.sum()),
          "non_dominated_cell_rate":nd_rate,"min_candidate_exact_valid_rate":args.min_candidate_exact_valid_rate,
          "min_non_dominated_cell_rate":args.min_non_dominated_cell_rate,
          "runtime_convention":"control/candidate wall_seconds <= 1.25",
          "pairing_key":"n_sites,delta,c6,seed","stage4c_status":status,
          "scope_note":"Gate is a predefined Pareto non-domination diagnostic; it does not establish universal superiority."}
    pd.DataFrame([gate]).to_csv(out/"stage4c_decision_gate_repaired.csv",index=False)
    (out/"stage4c_decision_gate_repaired.json").write_text(json.dumps(gate,indent=2)+"\n")
    print(pd.DataFrame([gate]).to_string(index=False))
    print("Wrote",out)
if __name__=="__main__": main()
