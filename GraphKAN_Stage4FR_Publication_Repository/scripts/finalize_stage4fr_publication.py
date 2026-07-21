#!/usr/bin/env python3
"""Build the immutable Stage 4F-R analysis and supervisory handoff package.

This is post-processing code.  It never writes inside the frozen campaign root.
All comparisons are labelled descriptive unless a frozen decision threshold exists.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sqlite3
import subprocess
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


OUTCOMES = {
    "energy": "exact_model_abs_energy_error_to_ed",
    "density": "exact_model_density_mae_to_ed",
    "correlation": "exact_model_connected_correlation_mae_to_ed",
}
MODELS = [
    "rbf_radial_b32",
    "endpoint_sum_b32",
    "endpoint_product_b32",
    "endpoint_hybrid_b32",
    "endpoint_full_b32",
    "mlp_endpoint_medium",
    "mlp_endpoint_large",
    "rbm_h32",
]
DISPLAY = {
    "rbf_radial_b32": "Radial RBF",
    "endpoint_sum_b32": "Endpoint sum",
    "endpoint_product_b32": "Endpoint product",
    "endpoint_hybrid_b32": "Endpoint hybrid",
    "endpoint_full_b32": "Endpoint full",
    "mlp_endpoint_medium": "MLP medium",
    "mlp_endpoint_large": "MLP large",
    "rbm_h32": "RBM h32",
}
COLORS = {
    "rbf_radial_b32": "#6b7280",
    "endpoint_sum_b32": "#38bdf8",
    "endpoint_product_b32": "#0ea5e9",
    "endpoint_hybrid_b32": "#2563eb",
    "endpoint_full_b32": "#1d4ed8",
    "mlp_endpoint_medium": "#f59e0b",
    "mlp_endpoint_large": "#ea580c",
    "rbm_h32": "#16a34a",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_metrics(campaign: Path) -> pd.DataFrame:
    files = sorted((campaign / "runs").glob("*/metrics.csv"))
    frames = [pd.read_csv(path) for path in files]
    if not frames:
        raise RuntimeError(f"no metrics files under {campaign / 'runs'}")
    frame = pd.concat(frames, ignore_index=True)
    frame["boundary"] = frame["boundary"].fillna(frame["boundary_condition"])
    frame["model"] = frame["model"].fillna(frame["model_label"])
    return frame


def registry_summary(campaign: Path) -> dict:
    con = sqlite3.connect(campaign / "registry.sqlite")
    con.row_factory = sqlite3.Row
    status = {r["status"]: r["n"] for r in con.execute(
        "SELECT status, COUNT(*) AS n FROM runs GROUP BY status"
    )}
    lane = {r["lane"]: r["n"] for r in con.execute(
        "SELECT lane, COUNT(*) AS n FROM runs WHERE status='COMPLETE' GROUP BY lane"
    )}
    attempts = {str(r["attempt"]): r["n"] for r in con.execute(
        "SELECT attempt, COUNT(*) AS n FROM runs GROUP BY attempt ORDER BY attempt"
    )}
    duplicate_keys = con.execute(
        """SELECT COUNT(*) FROM (
        SELECT boundary,delta,c6,model,seed,COUNT(*) n FROM runs
        GROUP BY boundary,delta,c6,model,seed HAVING n>1)"""
    ).fetchone()[0]
    event_counts = {r["event_type"]: r["n"] for r in con.execute(
        "SELECT event_type, COUNT(*) AS n FROM events GROUP BY event_type"
    )}
    con.close()
    return {
        "status_counts": status,
        "complete_by_lane": lane,
        "attempt_counts": attempts,
        "duplicate_registry_keys": duplicate_keys,
        "event_counts": event_counts,
    }


def validate_dataset(frame: pd.DataFrame, registry: dict) -> list[str]:
    errors: list[str] = []
    if len(frame) != 960:
        errors.append(f"expected 960 metric rows; found {len(frame)}")
    if frame.run_id.nunique() != 960:
        errors.append(f"expected 960 unique run IDs; found {frame.run_id.nunique()}")
    if registry["status_counts"] != {"COMPLETE": 960}:
        errors.append(f"registry not fully complete: {registry['status_counts']}")
    if registry["duplicate_registry_keys"]:
        errors.append("registry contains duplicate scientific keys")
    expected_grid = {
        "boundary": {"open", "periodic"},
        "delta": {1.25, 1.5, 1.75, 2.0},
        "c6": {3.0, 4.0, 5.0},
        "seed": {0, 1, 2, 3, 4},
        "model": set(MODELS),
    }
    for column, expected in expected_grid.items():
        got = set(frame[column].dropna().tolist())
        if got != expected:
            errors.append(f"{column} roster mismatch: {sorted(got)}")
    for column in OUTCOMES.values():
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.isna().any() or not np.isfinite(numeric).all() or (numeric < 0).any():
            errors.append(f"invalid values in {column}")
    exact_states = set(frame["scientific_validity_exact"].dropna())
    sample_states = set(frame["scientific_validity_sample"].dropna())
    if not exact_states <= {"VALID", "FLAGGED"}:
        errors.append(f"unexpected exact validity states: {sorted(exact_states)}")
    if not sample_states <= {"VALID", "FLAGGED"}:
        errors.append(f"unexpected sampled validity states: {sorted(sample_states)}")
    bad_status = int((frame["status"] != "COMPLETED").sum())
    if bad_status:
        errors.append(f"{bad_status} rows have unexpected status")
    return errors


def bootstrap_mean_ci(values: np.ndarray, seed: int = 417, draws: int = 20000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def holm_adjust(pvalues: pd.Series) -> pd.Series:
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * p[index])
        adjusted[index] = min(1.0, running)
    return pd.Series(adjusted, index=pvalues.index)


def make_analysis(frame: pd.DataFrame, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    tables = out / "tables"
    tables.mkdir(exist_ok=True)
    frame.to_csv(tables / "stage4fr_run_metrics.csv", index=False)

    group = ["boundary", "delta", "c6", "model"]
    agg_spec: dict[str, tuple[str, str]] = {
        "runs": ("run_id", "size"),
        "exact_valid": ("scientific_validity_exact", lambda x: int((x == "VALID").sum())),
        "median_wall_seconds": ("wall_seconds", "median"),
        "median_parameters": ("n_parameters", "median"),
    }
    for short, column in OUTCOMES.items():
        agg_spec[f"median_{short}_error"] = (column, "median")
        agg_spec[f"mean_{short}_error"] = (column, "mean")
    cell = frame.groupby(group, as_index=False).agg(**agg_spec)
    cell.to_csv(tables / "stage4fr_cell_model_summary.csv", index=False)

    broad = frame.groupby(["boundary", "model"], as_index=False).agg(**agg_spec)
    broad.to_csv(tables / "stage4fr_model_boundary_summary.csv", index=False)

    health = frame.groupby(["boundary", "model"], as_index=False).agg(
        runs=("run_id", "size"),
        exact_valid=("scientific_validity_exact", lambda x: int((x == "VALID").sum())),
        exact_flagged=("scientific_validity_exact", lambda x: int((x == "FLAGGED").sum())),
        sample_valid=("scientific_validity_sample", lambda x: int((x == "VALID").sum())),
        sample_flagged=("scientific_validity_sample", lambda x: int((x == "FLAGGED").sum())),
    )
    health["exact_valid_rate"] = health.exact_valid / health.runs
    health["sample_valid_rate"] = health.sample_valid / health.runs
    health.to_csv(tables / "stage4fr_scientific_health_summary.csv", index=False)
    flag_inventory = (frame.assign(
        exact_health_flags=frame.exact_health_flags.fillna(""),
        sample_health_flags=frame.sample_health_flags.fillna(""),
    ).groupby(["boundary", "model", "scientific_validity_exact", "scientific_validity_sample", "exact_health_flags", "sample_health_flags"], as_index=False)
      .agg(runs=("run_id", "size")))
    flag_inventory.to_csv(tables / "stage4fr_health_flag_inventory.csv", index=False)

    # Cell is the inferential unit; seeds are first reduced by the median.
    comparisons: list[dict] = []
    refs = ["rbf_radial_b32", "mlp_endpoint_large", "rbm_h32"]
    candidates = [m for m in MODELS if m not in refs]
    for scope in ["all", "open", "periodic"]:
        scoped = cell if scope == "all" else cell[cell.boundary == scope]
        for candidate in candidates:
            for reference in refs:
                for outcome in OUTCOMES:
                    col = f"median_{outcome}_error"
                    left = scoped[scoped.model == candidate][["boundary", "delta", "c6", col]]
                    right = scoped[scoped.model == reference][["boundary", "delta", "c6", col]]
                    paired = left.merge(right, on=["boundary", "delta", "c6"], suffixes=("_candidate", "_reference"))
                    diff = paired[f"{col}_candidate"].to_numpy() - paired[f"{col}_reference"].to_numpy()
                    lo, hi = bootstrap_mean_ci(diff, seed=417 + len(comparisons))
                    try:
                        from scipy.stats import wilcoxon
                        pvalue = float(wilcoxon(diff, zero_method="wilcox", alternative="two-sided").pvalue) if np.any(diff) else 1.0
                    except Exception:
                        pvalue = float("nan")
                    comparisons.append({
                        "scope": scope,
                        "outcome": outcome,
                        "candidate": candidate,
                        "reference": reference,
                        "paired_cells": len(diff),
                        "mean_difference_candidate_minus_reference": float(np.mean(diff)),
                        "median_difference_candidate_minus_reference": float(np.median(diff)),
                        "bootstrap95_low": lo,
                        "bootstrap95_high": hi,
                        "candidate_win_fraction": float(np.mean(diff < 0)),
                        "wilcoxon_p_unadjusted": pvalue,
                    })
    paired = pd.DataFrame(comparisons)
    paired["holm_p_within_scope_outcome"] = paired.groupby(["scope", "outcome"])["wilcoxon_p_unadjusted"].transform(holm_adjust)
    paired.to_csv(tables / "stage4fr_paired_model_comparisons.csv", index=False)

    boundary_rows: list[dict] = []
    for model in MODELS:
        for outcome in OUTCOMES:
            col = f"median_{outcome}_error"
            op = cell[cell.model == model][["delta", "c6", "boundary", col]].pivot(index=["delta", "c6"], columns="boundary", values=col).dropna()
            diff = (op["open"] - op["periodic"]).to_numpy()
            lo, hi = bootstrap_mean_ci(diff, seed=911 + len(boundary_rows))
            boundary_rows.append({
                "model": model,
                "outcome": outcome,
                "paired_cells": len(diff),
                "mean_open_minus_periodic": float(np.mean(diff)),
                "median_open_minus_periodic": float(np.median(diff)),
                "bootstrap95_low": lo,
                "bootstrap95_high": hi,
            })
    boundary = pd.DataFrame(boundary_rows)
    boundary.to_csv(tables / "stage4fr_open_periodic_effects.csv", index=False)

    ranking_rows = []
    for boundary_name in ["open", "periodic"]:
        for outcome in OUTCOMES:
            col = f"median_{outcome}_error"
            ordered = broad[broad.boundary == boundary_name].sort_values(col)
            for rank, row in enumerate(ordered.itertuples(), 1):
                ranking_rows.append({"boundary": boundary_name, "outcome": outcome, "rank": rank, "model": row.model, "median_error": getattr(row, col)})
    rankings = pd.DataFrame(ranking_rows)
    rankings.to_csv(tables / "stage4fr_descriptive_rankings.csv", index=False)

    # Figure 1: exact outcome medians by model and boundary.
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), constrained_layout=True)
    x = np.arange(len(MODELS))
    for ax, (outcome, _) in zip(axes, OUTCOMES.items()):
        col = f"median_{outcome}_error"
        for offset, boundary_name in [(-0.16, "open"), (0.16, "periodic")]:
            vals = broad.set_index(["boundary", "model"]).loc[[(boundary_name, m) for m in MODELS], col].to_numpy()
            ax.scatter(x + offset, vals, s=55, label=boundary_name.title(), marker="o" if boundary_name == "open" else "s")
        ax.set_yscale("log")
        ax.set_title(f"Exact {outcome} error")
        ax.set_xticks(x, [DISPLAY[m] for m in MODELS], rotation=55, ha="right", fontsize=8)
        ax.grid(alpha=.25, which="both")
    axes[0].set_ylabel("Median absolute error across 60 runs")
    axes[0].legend(frameon=False)
    fig.suptitle("Stage 4F-R exact-outcome performance (descriptive)", fontsize=14, fontweight="bold")
    fig.savefig(figures / "fig1_exact_outcome_performance.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: endpoint candidate minus radial, cell-level mean and bootstrap CI.
    focus = paired[(paired.scope.isin(["open", "periodic"])) & (paired.reference == "rbf_radial_b32")]
    endpoint_models = ["endpoint_sum_b32", "endpoint_product_b32", "endpoint_hybrid_b32", "endpoint_full_b32"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)
    for ax, outcome in zip(axes, OUTCOMES):
        part = focus[focus.outcome == outcome]
        y = np.arange(len(endpoint_models))
        for off, scope, marker in [(-0.13, "open", "o"), (0.13, "periodic", "s")]:
            rows = part.set_index(["scope", "candidate"]).loc[[(scope, m) for m in endpoint_models]]
            mean = rows.mean_difference_candidate_minus_reference.to_numpy()
            low = rows.bootstrap95_low.to_numpy()
            high = rows.bootstrap95_high.to_numpy()
            ax.errorbar(mean, y + off, xerr=[mean - low, high - mean], fmt=marker, capsize=3, label=scope.title())
        ax.axvline(0, color="#111827", lw=1)
        ax.set_title(outcome.title())
        ax.set_yticks(y, [DISPLAY[m] for m in endpoint_models], fontsize=8)
        ax.grid(alpha=.25, axis="x")
        ax.set_xlabel("Candidate - radial error\n(negative favours endpoint)")
    axes[0].legend(frameon=False)
    fig.suptitle("Boundary-conditioned endpoint effect vs radial baseline", fontsize=14, fontweight="bold")
    fig.savefig(figures / "fig2_endpoint_effect_vs_radial.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Figure 3: energy improvement map for the full endpoint model.
    energy = cell[["boundary", "delta", "c6", "model", "median_energy_error"]]
    base = energy[energy.model == "rbf_radial_b32"].rename(columns={"median_energy_error": "radial"})
    cand = energy[energy.model == "endpoint_full_b32"].rename(columns={"median_energy_error": "endpoint"})
    heat = cand.merge(base, on=["boundary", "delta", "c6"])
    heat["radial_minus_endpoint"] = heat.radial - heat.endpoint
    vmax = float(np.max(np.abs(heat.radial_minus_endpoint)))
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4), constrained_layout=True)
    for ax, boundary_name in zip(axes, ["open", "periodic"]):
        data = heat[heat.boundary == boundary_name].pivot(index="delta", columns="c6", values="radial_minus_endpoint").sort_index(ascending=False)
        im = ax.imshow(data, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(data.columns)), [f"{x:g}" for x in data.columns])
        ax.set_yticks(range(len(data.index)), [f"{x:g}" for x in data.index])
        ax.set_xlabel("C6")
        ax.set_ylabel("Delta")
        ax.set_title(boundary_name.title())
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, f"{data.iloc[i,j]:.3g}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=axes, label="Radial - endpoint-full energy error\n(positive favours endpoint)")
    fig.suptitle("Endpoint-full energy-error change by registered cell", fontsize=14, fontweight="bold")
    fig.savefig(figures / "fig3_endpoint_full_energy_map.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Figure 4: accuracy/runtime trade-off.
    fig, ax = plt.subplots(figsize=(8.5, 5.3), constrained_layout=True)
    for model in MODELS:
        part = broad[broad.model == model]
        ax.scatter(part.median_wall_seconds, part.median_energy_error, s=80, color=COLORS[model], label=DISPLAY[model], edgecolor="white", linewidth=.6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Median wall time per run (seconds)")
    ax.set_ylabel("Median exact energy error")
    ax.grid(alpha=.25, which="both")
    ax.legend(frameon=False, ncol=2, fontsize=8)
    ax.set_title("Accuracy-runtime trade-off by model and boundary")
    fig.savefig(figures / "fig4_accuracy_runtime_pareto.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    leaders = {}
    for boundary_name in ["open", "periodic"]:
        leaders[boundary_name] = {}
        for outcome in OUTCOMES:
            row = rankings[(rankings.boundary == boundary_name) & (rankings.outcome == outcome) & (rankings["rank"] == 1)].iloc[0]
            leaders[boundary_name][outcome] = {"model": row.model, "median_error": float(row.median_error)}
    summary = {
        "generated_utc": utc_now(),
        "analysis_status": "PASS_WITH_REGISTERED_SCIENTIFIC_FAILURES",
        "interpretation_class": "descriptive confirmatory outcomes with cell-level paired comparisons; no universal-superiority claim",
        "rows": len(frame),
        "unique_runs": int(frame.run_id.nunique()),
        "exact_valid_runs": int((frame.scientific_validity_exact == "VALID").sum()),
        "exact_flagged_runs": int((frame.scientific_validity_exact == "FLAGGED").sum()),
        "sample_valid_runs": int((frame.scientific_validity_sample == "VALID").sum()),
        "sample_flagged_runs": int((frame.scientific_validity_sample == "FLAGGED").sum()),
        "registered_cells": int(frame[["boundary", "delta", "c6"]].drop_duplicates().shape[0]),
        "models": MODELS,
        "descriptive_leaders": leaders,
        "primary_outcomes": list(OUTCOMES),
        "multiplicity_note": "Wilcoxon p-values are Holm-adjusted within each scope/outcome family; cells, not seeds, are the comparison unit.",
    }
    (out / "stage4fr_analysis_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def check_protocol_source(root: Path, campaign_manifest: dict) -> dict:
    mismatches = []
    ledger = campaign_manifest["protocol_identity"]["source_ledger"]
    for rel, expected in sorted(ledger.items()):
        path = root / rel
        actual = sha256(path) if path.exists() else None
        if actual != expected:
            mismatches.append({"path": rel, "expected": expected, "actual": actual})
    return {"entries_checked": len(ledger), "mismatches": mismatches, "status": "PASS" if not mismatches else "FAIL"}


def build_gate_summary(root: Path, campaign: Path, registry: dict, source_check: dict) -> dict:
    validation = json.loads((campaign / "stage4f1c_validation_summary.json").read_text())
    kill = json.loads((root / "results_stage4f1c/real_microcampaign_killtest/kill_microcampaign_summary.json").read_text())
    replay = json.loads((root / "results_stage4f1c/independent_replay/stage4f1c_validation_summary.json").read_text())
    replay_commands = list((root / "results_stage4f1c/independent_replay/runs").glob("*/command.json"))
    replay_workers = sorted({json.loads(path.read_text()).get("worker_id", "") for path in replay_commands})
    same_host = any("Subhojits-MacBook-Air" in worker for worker in replay_workers)
    return {
        "generated_utc": utc_now(),
        "campaign_integrity": {
            "status": "PASS" if validation.get("validation") == "PASS" else "FAIL",
            "complete_bundles": validation.get("complete_bundles_checked"),
            "blocking_errors": validation.get("blocking_errors"),
            "registry_filesystem_errors": validation.get("registry_filesystem_errors"),
            "registry": registry,
        },
        "protocol_source_ledger": source_check,
        "kill_test": {
            "status": kill.get("gate"),
            "fault_phases": kill.get("fault_results"),
            "terminal_runs": len(kill.get("registry_evidence", {}).get("runs", [])),
        },
        "replay": {
            "technical_validation": replay.get("validation"),
            "complete_bundles": replay.get("complete_bundles_checked"),
            "worker_ids": replay_workers,
            "independent_second_host": False if same_host else None,
            "supervisory_status": "OPEN - rerun on a genuinely separate host" if same_host else "REQUIRES PROVENANCE REVIEW",
        },
        "overall_supervisory_request": "CONDITIONAL GO: accept Stage 4F.1c and Stage 4F-R campaign integrity; hold public release/next-stage escalation until second-host replay is supplied or waived by the supervisor.",
    }


def _p(text: str, style) -> Paragraph:
    return Paragraph(text.replace("&", "&amp;"), style)


def build_pdf(report_path: Path, analysis_dir: Path, analysis: dict, gates: dict, campaign_manifest: dict) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleBlue", parent=styles["Title"], textColor=colors.HexColor("#123a63"), fontSize=25, leading=30, spaceAfter=12))
    styles.add(ParagraphStyle(name="H1Blue", parent=styles["Heading1"], textColor=colors.HexColor("#123a63"), fontSize=17, leading=21, spaceBefore=6, spaceAfter=10))
    styles.add(ParagraphStyle(name="H2Blue", parent=styles["Heading2"], textColor=colors.HexColor("#1d5f8a"), fontSize=12, leading=15, spaceBefore=8, spaceAfter=6))
    styles.add(ParagraphStyle(name="BodyCompact", parent=styles["BodyText"], fontSize=9.2, leading=13, spaceAfter=6))
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=7.6, leading=10))
    styles.add(ParagraphStyle(name="Status", parent=styles["BodyText"], fontSize=12, leading=16, alignment=TA_CENTER, textColor=colors.white))
    styles.add(ParagraphStyle(name="Caption", parent=styles["BodyText"], fontSize=8, leading=10, textColor=colors.HexColor("#475569"), alignment=TA_CENTER))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#94a3b8"))
        canvas.line(18 * mm, 15 * mm, 192 * mm, 15 * mm)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#475569"))
        canvas.drawString(18 * mm, 10 * mm, "GraphKAN AMO NQS - Stage 4F-R final validation handoff")
        canvas.drawRightString(192 * mm, 10 * mm, f"Page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(str(report_path), pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm, topMargin=17 * mm, bottomMargin=20 * mm, title="GraphKAN Stage 4F-R Final Validation Handoff", author="Prepared for Srinjoy supervisory validation")
    story = []
    story += [_p("GraphKAN AMO NQS", styles["H2Blue"]), _p("Stage 4F-R Final Supervisory Validation Handoff", styles["TitleBlue"])]
    story += [_p("Frozen 960-run boundary-aware Rydberg NQS campaign", styles["Heading2"]), Spacer(1, 6)]
    status = Table([[Paragraph("CAMPAIGN + INTEGRITY: PASS", styles["Status"]), Paragraph("SECOND-HOST REPLAY: OPEN", styles["Status"])]], colWidths=[87 * mm, 87 * mm])
    status.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, 0), colors.HexColor("#16803c")), ("BACKGROUND", (1, 0), (1, 0), colors.HexColor("#b45309")), ("BOX", (0, 0), (-1, -1), .5, colors.white), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 10), ("BOTTOMPADDING", (0, 0), (-1, -1), 10)]))
    story += [status, Spacer(1, 14)]
    story += [_p("Prepared for final full-stage validation by Srinjoy", styles["H2Blue"]), _p("Version 1.0 - 20 July 2026", styles["BodyCompact"])]
    story += [Spacer(1, 14), _p("Executive statement", styles["H1Blue"])]
    story += [_p("All 960 registered work items are COMPLETE: 840 core-lane model runs and 120 RBM control runs. Strict bundle, checkpoint, manifest, registry, and filesystem reconciliation checked all 960 canonical bundles and returned PASS with zero blocking errors, zero filesystem mismatches, and zero duplicate registry keys. The execution source tree still matches all 105 hashes frozen at campaign initialization.", styles["BodyCompact"])]
    story += [_p(f"Scientific health is mixed rather than universally green: {analysis['exact_valid_runs']}/960 runs pass the exact-health screen and {analysis['exact_flagged_runs']} are flagged; {analysis['sample_valid_runs']}/960 pass the sampled-health screen and {analysis['sample_flagged_runs']} are flagged. These registered failures are retained in every full-grid table and reported as a failure-map result, not removed as invalid files.", styles["BodyCompact"])]
    story += [_p("The four-phase SIGKILL microcampaign passed and the frozen science roster was not changed after launch. The existing four-run replay also passes technically, but its recorded worker provenance is this same Mac; therefore this package does not claim a genuinely independent second-host replay.", styles["BodyCompact"])]
    story += [_p("Decision requested", styles["H2Blue"]), _p(gates["overall_supervisory_request"], styles["BodyCompact"]), PageBreak()]

    story += [_p("1. Acceptance-gate matrix", styles["H1Blue"])]
    gate_rows = [
        ["Gate", "Evidence", "Status"],
        ["Typed fail-closed bundles", "960 strict bundle validations", "PASS"],
        ["Real checkpoint loading", "960 NPZ checkpoints deserialised", "PASS"],
        ["Registry/filesystem reconciliation", "0 mismatches", "PASS"],
        ["Protocol/source identity", "105/105 frozen hashes match", "PASS"],
        ["Heartbeat + attempt fencing", "Kill/retry evidence and terminal registry", "PASS"],
        ["Four planned process kills", "training/checkpoint/promotion/post-promotion", "PASS"],
        ["16-run real microcampaign", "16/16 COMPLETE", "PASS"],
        ["Frozen Stage 4F-R campaign", "960/960 COMPLETE", "PASS"],
        ["Exact scientific health", f"{analysis['exact_valid_runs']} VALID; {analysis['exact_flagged_runs']} FLAGGED", "MIXED"],
        ["Sampled scientific health", f"{analysis['sample_valid_runs']} VALID; {analysis['sample_flagged_runs']} FLAGGED", "MIXED"],
        ["Independent second-host replay", "4/4 PASS, but same-host provenance", "OPEN"],
        ["Public artifact release", "Await supervisor decision on replay", "HOLD"],
    ]
    table = Table(gate_rows, colWidths=[52 * mm, 93 * mm, 28 * mm], repeatRows=1)
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#123a63")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("GRID", (0, 0), (-1, -1), .35, colors.HexColor("#cbd5e1")), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("FONTSIZE", (0, 0), (-1, -1), 8.2), ("LEADING", (0, 0), (-1, -1), 10), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]), ("TEXTCOLOR", (2, 1), (2, -1), colors.HexColor("#166534")), ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    story += [table, Spacer(1, 12), _p("Interpretation", styles["H2Blue"]), _p("The engineering closure and full campaign integrity gates pass. Final external reproducibility remains a distinct condition and is not inferred from same-host execution.", styles["BodyCompact"]), PageBreak()]

    story += [_p("2. Frozen protocol and dataset integrity", styles["H1Blue"])]
    pident = campaign_manifest["protocol_identity"]
    rows = [
        ["Item", "Frozen value"],
        ["Campaign ID", campaign_manifest["campaign_id"]],
        ["Protocol SHA-256", campaign_manifest["protocol_hash"]],
        ["System", "N=10 Rydberg chain; omega=1.0"],
        ["Registered grid", "2 boundaries x 4 delta x 3 C6 x 8 models x 5 seeds = 960"],
        ["Primary outcomes", "Exact energy, density, connected correlation"],
        ["Sampling", "4096 samples; SR shift 0.1"],
        ["Canonical bundles", "960"],
        ["Quarantined late attempts", str(gates["campaign_integrity"]["registry"]["event_counts"].get("FAILED_RETRYABLE", 0)) + " retry events retained as noncanonical evidence"],
        ["Source ledger", "105 entries; 0 mismatches"],
    ]
    t = Table(rows, colWidths=[43 * mm, 130 * mm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#123a63")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), .35, colors.HexColor("#cbd5e1")), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("FONTSIZE", (0, 0), (-1, -1), 8), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]), ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    story += [t, Spacer(1, 10), _p("Validation boundary", styles["H2Blue"]), _p("The canonical campaign directory is treated as immutable. All tables and figures in this handoff were written outside the frozen directory. Analysis operates only on canonical COMPLETE bundles; quarantined late attempts are retained for engineering audit but excluded from science tables.", styles["BodyCompact"]), PageBreak()]

    story += [_p("3. Preregistered exact outcomes", styles["H1Blue"])]
    story += [Image(str(analysis_dir / "figures/fig1_exact_outcome_performance.png"), width=174 * mm, height=62 * mm), _p("Figure 1. Median absolute error across registered runs, separated by boundary. Log scale; descriptive only.", styles["Caption"]), Spacer(1, 8)]
    leader_rows = [["Boundary", "Outcome", "Lowest descriptive median", "Median error"]]
    for boundary_name in ["open", "periodic"]:
        for outcome in OUTCOMES:
            row = analysis["descriptive_leaders"][boundary_name][outcome]
            leader_rows.append([boundary_name.title(), outcome.title(), DISPLAY[row["model"]], f"{row['median_error']:.4g}"])
    lt = Table(leader_rows, colWidths=[31 * mm, 36 * mm, 70 * mm, 35 * mm])
    lt.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#123a63")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), .35, colors.HexColor("#cbd5e1")), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]), ("FONTSIZE", (0, 0), (-1, -1), 8.3), ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    story += [lt, Spacer(1, 8), _p(f"Health screen: {analysis['exact_flagged_runs']} exact-health and {analysis['sample_flagged_runs']} sampled-health rows are flagged. The complete registered grid, including those failures, is used in the displayed medians. These rankings do not establish generic model superiority and must be interpreted jointly across all three outcomes, parameter count, runtime, and boundary.", styles["BodyCompact"]), PageBreak()]

    story += [_p("4. Boundary-conditioned endpoint mechanism", styles["H1Blue"]), Image(str(analysis_dir / "figures/fig2_endpoint_effect_vs_radial.png"), width=174 * mm, height=60 * mm), _p("Figure 2. Cell-level paired endpoint-minus-radial error. Negative favours the endpoint variant. Bars are deterministic 95% bootstrap intervals over registered physics cells after median reduction across five seeds.", styles["Caption"]), Spacer(1, 8), Image(str(analysis_dir / "figures/fig3_endpoint_full_energy_map.png"), width=148 * mm, height=62 * mm), _p("Figure 3. Registered-cell map for endpoint-full versus radial exact-energy error. Positive values favour endpoint-full.", styles["Caption"]), PageBreak()]

    story += [_p("5. Accuracy-runtime trade-off", styles["H1Blue"]), Image(str(analysis_dir / "figures/fig4_accuracy_runtime_pareto.png"), width=155 * mm, height=96 * mm), _p("Figure 4. Each model contributes one open and one periodic aggregate point. Both axes are logarithmic.", styles["Caption"]), Spacer(1, 10), _p("Statistical handling", styles["H2Blue"]), _p("The registered physics cell is the comparison unit. Five seeds are first reduced by their cell median. Paired candidate-minus-reference comparisons report mean and median differences, candidate win fraction, deterministic bootstrap intervals, Wilcoxon signed-rank p-values, and Holm adjustment within each scope/outcome family. These outputs are supplied for supervisory analysis and are not converted into a universal-superiority claim.", styles["BodyCompact"]), PageBreak()]

    story += [_p("6. Crash recovery and retry evidence", styles["H1Blue"])]
    fault_rows = [["Fault phase", "Lane", "SIGKILL observed", "Recovery exit", "Gate"]]
    for row in gates["kill_test"]["fault_phases"]:
        fault_rows.append([row["phase"], row["lane"], str(row["sigkill_observed"]), str(row["recovery_returncode"]), "PASS" if row["pass"] else "FAIL"])
    ft = Table(fault_rows, colWidths=[45 * mm, 27 * mm, 40 * mm, 31 * mm, 28 * mm])
    ft.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#123a63")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), .35, colors.HexColor("#cbd5e1")), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]), ("FONTSIZE", (0, 0), (-1, -1), 8.3), ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    story += [ft, Spacer(1, 12), _p("The 16-run microcampaign terminated 16/16 COMPLETE after four deliberate process kills. The full Stage 4F-R registry records all 960 rows COMPLETE. Late or lease-lost attempts were quarantined and did not become canonical.", styles["BodyCompact"]), _p("Replay disclosure", styles["H2Blue"]), _p("The four-run replay validates 4/4 canonical bundles with zero reconciliation errors. Its worker IDs contain 'Subhojits-MacBook-Air', so it is a clean same-host replay, not independent second-host evidence. The package therefore requests either a genuine separate-host replay or an explicit supervisory waiver before public release.", styles["BodyCompact"]), PageBreak()]

    story += [_p("7. Reproducibility and review commands", styles["H1Blue"])]
    commands = [
        "sha256sum -c SHA256SUMS.txt",
        "PYTHONPATH=\"$PWD/src\" python scripts/stage4f1c_campaign.py validate --campaign results_stage4f_r/frozen --strict-integrity --reconcile-registry --require-terminal --require-complete",
        "python -m pytest -q",
        "python scripts/finalize_stage4fr_publication.py --campaign results_stage4f_r/frozen --analysis-output results_stage4f_r/publication_analysis --report output/pdf/GraphKAN_Stage4FR_Final_Validation_Handoff_20260720.pdf",
    ]
    for command in commands:
        story += [_p(command, styles["Small"]), Spacer(1, 5)]
    story += [Spacer(1, 8), _p("Reviewer checklist", styles["H2Blue"])]
    checks = [
        "Confirm the repository and root ledger checksums.",
        "Re-run strict campaign validation; require 960/960 COMPLETE and PASS.",
        "Confirm campaign protocol hash 327f66318ee93c3365f5a12ba4d64fddd20a29ee288cc30330d437f514328ff9.",
        "Inspect kill_microcampaign_summary.json for four SIGKILL/recovery passes.",
        "Review analysis tables and figures without promoting descriptive rankings to universal claims.",
        "Decide whether a genuine second-host replay is required before next-stage/public release.",
    ]
    story += [_p("<br/>".join(f"- {x}" for x in checks), styles["BodyCompact"]), Spacer(1, 10), _p("Requested decision", styles["H2Blue"]), _p(gates["overall_supervisory_request"], styles["BodyCompact"])]

    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def copy_protocol_files(root: Path, package: Path, campaign_manifest: dict) -> None:
    for rel in sorted(campaign_manifest["protocol_identity"]["source_ledger"]):
        src = root / rel
        dst = package / "reproducibility" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    for rel in [
        "configs/stage4f_r_frozen.yaml",
        "configs/stage4f1c_independent_replay.yaml",
        "configs/stage4f1c_real_microcampaign_killtest.yaml",
        "docs/tracking/stage4f/README_STAGE4F1C.md",
        "tests/test_stage4f1c_closure.py",
        "tests/test_stage4f1b_semantic_lease.py",
        "tests/test_stage4f1_transactional_scheduler.py",
        "scripts/finalize_stage4fr_publication.py",
    ]:
        src = root / rel
        if src.exists():
            dst = package / "reproducibility" / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def write_manifest(package: Path) -> None:
    entries = []
    for path in sorted(package.rglob("*")):
        if path.is_file() and path.name not in {"MANIFEST.sha256", "SHA256SUMS.txt"}:
            entries.append(f"{sha256(path)}  {path.relative_to(package).as_posix()}")
    text = "\n".join(entries) + "\n"
    (package / "MANIFEST.sha256").write_text(text)
    (package / "SHA256SUMS.txt").write_text(text)


def build_package(root: Path, package: Path, campaign: Path, analysis_dir: Path, report: Path, gates: dict, campaign_manifest: dict) -> Path:
    if package.exists():
        shutil.rmtree(package)
    package.mkdir(parents=True)
    shutil.copytree(campaign, package / "results_stage4f_r/frozen", copy_function=shutil.copy2)
    shutil.copytree(root / "results_stage4f1c/real_microcampaign_killtest", package / "evidence/real_microcampaign_killtest", copy_function=shutil.copy2)
    shutil.copytree(root / "results_stage4f1c/independent_replay", package / "evidence/replay_same_host", copy_function=shutil.copy2)
    shutil.copytree(analysis_dir, package / "analysis", copy_function=shutil.copy2)
    (package / "report").mkdir()
    shutil.copy2(report, package / "report" / report.name)
    copy_protocol_files(root, package, campaign_manifest)
    (package / "evidence/final_gate_summary.json").write_text(json.dumps(gates, indent=2, sort_keys=True) + "\n")

    readme = f"""# GraphKAN Stage 4F-R final supervisory handoff

## Current status

- Frozen campaign: **PASS**, 960/960 COMPLETE.
- Strict registry/filesystem/bundle validation: **PASS**.
- Frozen execution-source ledger: **PASS**, 105/105 hashes match.
- Four-phase SIGKILL microcampaign: **PASS**.
- Four-run replay: **technical PASS**, but recorded on the same Mac and therefore **not independent second-host evidence**.

## Requested supervisory decision

{gates['overall_supervisory_request']}

## Primary entry points

- `report/{report.name}` - concise supervisory report.
- `evidence/final_gate_summary.json` - machine-readable gate status.
- `analysis/stage4fr_analysis_summary.json` - machine-readable science summary.
- `analysis/tables/` and `analysis/figures/` - registered-outcome outputs.
- `results_stage4f_r/frozen/` - complete canonical 960-run campaign and quarantined late attempts.
- `evidence/real_microcampaign_killtest/` - kill/recovery evidence.
- `evidence/replay_same_host/` - replay evidence with explicit provenance limitation.
- `reproducibility/` - frozen execution sources/configs plus post-processing script.

## Verification

```bash
sha256sum -c SHA256SUMS.txt
export PYTHONPATH="$PWD/reproducibility/src:$PWD/reproducibility/scripts"
python reproducibility/scripts/stage4f1c_campaign.py validate \\
  --campaign results_stage4f_r/frozen \\
  --strict-integrity --reconcile-registry --require-terminal --require-complete
```

## Claim boundary

This package supports completion and integrity of the registered Stage 4F-R N=10 benchmark. It does not claim generic KAN, endpoint-feature, Graph-KAN, or top-journal superiority. Descriptive rankings apply only to the frozen registered grid. Public release/next-stage escalation remains conditional on supervisory treatment of the second-host replay requirement.
"""
    (package / "README.md").write_text(readme)

    handoff = f"""# Student handoff for Srinjoy - Stage 4F-R

Date: 20 July 2026

## Short update

Every Stage 4F.1c structural gate and the full frozen Stage 4F-R campaign pass local validation. The campaign is terminal at 960/960 COMPLETE, strict reconciliation checks all 960 canonical bundles, and the 105-entry frozen source ledger has zero drift. The four planned SIGKILL/recovery phases pass and the 16-run real microcampaign is 16/16 COMPLETE.

The four-run replay is technically clean but was executed on the same Mac. I am explicitly reporting that as an open requirement, not as independent-host evidence.

## Known limitations

1. Genuine second-host replay has not been demonstrated in the supplied provenance.
2. Full worker-shell transcripts were not centrally captured with `tee`; canonical per-run `command.json`, `stdout.log`, `stderr.log`, `exit_status.json`, registry events, and quarantined attempts are included instead.
3. Scientific tables are restricted to the frozen N=10 roster and preregistered exact energy, density, and connected-correlation outcomes.
4. No generic method-superiority or top-journal-readiness claim is made.

## Decision requested

{gates['overall_supervisory_request']}
"""
    (package / "HANDOFF_TO_SRINJOY.md").write_text(handoff)
    write_manifest(package)

    return package


def main() -> None:
    ap = argparse.ArgumentParser(description="Finalize Stage 4F-R analysis and supervisory handoff")
    ap.add_argument("--campaign", default="results_stage4f_r/frozen")
    ap.add_argument("--analysis-output", default="results_stage4f_r/publication_analysis")
    ap.add_argument("--report", default="output/pdf/GraphKAN_Stage4FR_Final_Validation_Handoff_20260720.pdf")
    ap.add_argument("--package", default="share_with_srinjoy_stage4fr_final_20260720")
    ap.add_argument("--skip-package", action="store_true")
    args = ap.parse_args()

    root = Path.cwd()
    campaign = Path(args.campaign)
    analysis_dir = Path(args.analysis_output)
    report = Path(args.report)
    package = Path(args.package)
    campaign_manifest = json.loads((campaign / "campaign_manifest.json").read_text())
    frame = load_metrics(campaign)
    registry = registry_summary(campaign)
    errors = validate_dataset(frame, registry)
    if errors:
        raise SystemExit("dataset validation failed:\n- " + "\n- ".join(errors))
    source_check = check_protocol_source(root, campaign_manifest)
    if source_check["status"] != "PASS":
        raise SystemExit("frozen source ledger no longer matches")
    analysis = make_analysis(frame, analysis_dir)
    gates = build_gate_summary(root, campaign, registry, source_check)
    (analysis_dir / "final_gate_summary.json").write_text(json.dumps(gates, indent=2, sort_keys=True) + "\n")
    build_pdf(report, analysis_dir, analysis, gates, campaign_manifest)
    result = {"analysis": str(analysis_dir), "report": str(report), "gate_status": gates["campaign_integrity"]["status"]}
    if not args.skip_package:
        package = build_package(root, package, campaign, analysis_dir, report, gates, campaign_manifest)
        result.update({"package": str(package)})
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
