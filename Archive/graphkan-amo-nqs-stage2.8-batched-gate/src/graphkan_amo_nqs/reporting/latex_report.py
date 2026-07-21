from __future__ import annotations

from pathlib import Path

import pandas as pd


def _tex_escape(s: object) -> str:
    text = str(s)
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")]:
        text = text.replace(a, b)
    return text


def make_report_tex(repo_root: str | Path = ".") -> Path:
    root = Path(repo_root)
    paper = root / "paper"
    paper.mkdir(exist_ok=True)
    fig_rel = lambda name: f"../figures/{name}"
    metrics = root / "results" / "aggregated" / "model_comparison.csv"
    paired = root / "results" / "aggregated" / "paired_comparisons.csv"
    gates = root / "results" / "aggregated" / "readiness_gates.csv"
    fail = root / "results" / "raw" / "failed_runs.csv"
    env = root / "release" / "manifest.json"
    mc = pd.read_csv(metrics) if metrics.exists() else pd.DataFrame()
    pc = pd.read_csv(paired) if paired.exists() else pd.DataFrame()
    gt = pd.read_csv(gates) if gates.exists() else pd.DataFrame()
    fr = pd.read_csv(fail) if fail.exists() else pd.DataFrame()
    def tab(df, cols=None, max_rows=12):
        if df.empty:
            return "\\begin{tabular}{l}not available\\end{tabular}"
        d = df.copy()
        if cols:
            d = d[[c for c in cols if c in d.columns]]
        latex = d.head(max_rows).to_latex(index=False, escape=True)
        return "\\resizebox{\\linewidth}{!}{%\n" + latex + "\n}"
    tex = rf'''
\documentclass[11pt]{{article}}
\usepackage[a4paper,margin=1in]{{geometry}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{hyperref}}
\usepackage{{longtable}}
\usepackage{{amsmath}}
\usepackage{{float}}
\title{{Production Analysis of Blockade-Scale-Normalised KAN-Jastrow and Graph-KAN Neural Quantum States for Rydberg AMO Benchmarks}}
\author{{GraphKAN AMO NQS production artifact}}
\date{{20 June 2026}}
\begin{{document}}
\maketitle
\begin{{abstract}}
This report audits the repaired proposal and the uploaded prototype code, implements a hardware-free CPU production-pilot engine, and records the experiments that could be executed in the current environment. The artifact validates Rydberg Hamiltonian construction and exact diagonalisation on small systems, runs exact full-basis variational pilots for product, Jastrow, MLP-pair random-feature and KAN-Jastrow models, extracts gauge-fixed learned radial functions, and generates reproducible figures and tables. It does not claim top-journal readiness, first KAN-NQS, Graph-KAN value, quantum advantage, or hardware validation.
\end{{abstract}}

\section{{Executive summary}}
The result is a real CPU benchmark engine, not a proposal-only scaffold. The available environment lacked NetKet, Flax, QuSpin and TeNPy, so production VMC and DMRG profiles are recorded as not executed. The feasible path produced exact references and exact-fullsum pilot evidence. The correct classification is pilot benchmark continuation, not paper-ready PRA or Quantum evidence.

\section{{Code and proposal audit}}
The uploaded proposal was already anti-scoop repaired: it forbids first KAN-NQS and KAN-superiority claims and centres the project on blockade-scale normalisation, gauge fixing, and fair Rydberg benchmarks. The uploaded codebase passed its original tests and smoke script but was a PyTorch smoke scaffold. The present repository replaces the scientific core with deterministic geometry, Hamiltonian, ED, exact-fullsum, statistics and reporting modules.

\section{{Software environment}}
The release manifest records Python, NumPy, SciPy, pandas and optional dependency status. Optional NetKet/JAX/Flax VMC remains a required next step for a final paper. The CPU implementation is intentionally hardware-free.

\section{{Hamiltonian validation}}
The Rydberg Hamiltonian used here is
\[
H/\hbar=\sum_i \frac{{\Omega_i}}{{2}}\sigma_i^x-\sum_i\Delta_i n_i+\sum_{{i<j}}\frac{{C_6}}{{r_{{ij}}^6}}n_i n_j.
\]
Dense and sparse builders are checked against each other, the $\Omega=0$ diagonal limit is checked analytically, and exact references are generated where feasible.

\begin{{figure}}[H]\centering\includegraphics[width=.75\linewidth]{{{fig_rel('fig_ed_validation.pdf')}}}\caption{{Hamiltonian validation from generated reference files.}}\end{{figure}}

\section{{Exact-reference generation}}
References were generated for the configured chain, square and ladder instances. Exact diagonalisation uses dense diagonalisation below a dimension threshold and SciPy sparse ARPACK above it. Large-system truth is not claimed.

\section{{Model and baseline definitions}}
The CPU pilot uses linear-in-parameters exact-basis feature models: product, radial/smooth Jastrow, MLP-pair random features with the same blockade-normalised inputs as KAN-Jastrow, and KAN-Jastrow radial basis features. This is sufficient for Hamiltonian/model plumbing and pilot comparisons, but it is not a substitute for final NetKet/Flax trainable baselines.

\section{{Experimental profiles executed}}
Executed profiles are quick CI, ED references, exact full-sum pilot, Rydberg chain pilot, aggregation, figures, tables, report, manifest, claim validation and readiness checks. PRA-core, geometry transfer, Graph-KAN, robustness, Bose-Hubbard and IBM profiles are not executed in this CPU artifact.

\section{{Results}}
\subsection{{Energy and correlations}}
\begin{{center}}
{tab(mc, ['profile','model','metric','median','ci_low','ci_high','n'], 20)}
\end{{center}}
\begin{{figure}}[H]\centering\includegraphics[width=.8\linewidth]{{{fig_rel('fig_pilot_energy_correlation.pdf')}}}\caption{{Pilot median energy-density and connected-correlation errors.}}\end{{figure}}

\subsection{{Paired statistics}}
\begin{{center}}
{tab(pc, ['comparison','metric','median_delta_baseline_minus_candidate','ci_low','ci_high','n_pairs','p_value'], 20)}
\end{{center}}
\begin{{figure}}[H]\centering\includegraphics[width=.8\linewidth]{{{fig_rel('fig_kan_vs_mlp_pair_ablation.pdf')}}}\caption{{Paired pilot ablations. Positive deltas favour KAN-Jastrow over the named baseline.}}\end{{figure}}

\subsection{{Compute}}
\begin{{figure}}[H]\centering\includegraphics[width=.75\linewidth]{{{fig_rel('fig_compute_pareto.pdf')}}}\caption{{Pilot compute-error Pareto plot.}}\end{{figure}}

\section{{Interpretability analysis}}
Gauge fixing subtracts a constant offset, centres the learned pair function, normalises amplitude, and aligns a sign convention. This prevents raw redundant parameterisations from being interpreted as physical functions.
\begin{{figure}}[H]\centering\includegraphics[width=.8\linewidth]{{{fig_rel('fig_gauge_fixed_radial_functions.pdf')}}}\caption{{Gauge-fixed KAN-Jastrow radial functions from the CPU pilot.}}\end{{figure}}
\begin{{figure}}[H]\centering\includegraphics[width=.8\linewidth]{{{fig_rel('fig_scale_collapse.pdf')}}}\caption{{Pilot scale-collapse diagnostic plotted against $r/R_\Omega$.}}\end{{figure}}

\section{{Ablations, robustness and failed runs}}
The CPU pilot includes the central KAN-vs-MLP-pair ablation under matched blockade-normalised input features. Full basis-family, edge-cutoff, geometry-transfer and sampler robustness studies remain next experiments. Failed or unexecuted runs are retained rather than hidden.
\begin{{center}}
{tab(fr, ['profile','instance_id','model','seed','stage','run_status','reason'], 10)}
\end{{center}}

\section{{Decision gates and venue readiness}}
\begin{{center}}
{tab(gt, ['gate','status','evidence'], 10)}
\end{{center}}
\begin{{figure}}[H]\centering\includegraphics[width=.8\linewidth]{{{fig_rel('fig_claim_ladder_decision.pdf')}}}\caption{{Claim ladder and readiness gates.}}\end{{figure}}

\section{{What went well}}
The Rydberg Hamiltonian path is testable and reproducible. The exact-fullsum engine makes model evaluation noise-free on small systems. The output schemas are long-format and figure/table generation is script-driven. Gauge-fixed learned functions are exported as data rather than manually drawn plots.

\section{{What went badly}}
The environment did not include NetKet, Flax, QuSpin or TeNPy. Therefore no production VMC, QuSpin cross-check, or DMRG reference was run. The CPU MLP and RBM baselines are pilot substitutes, not final strong baselines. Graph-KAN was not run because its value should be gated on KAN-Jastrow stability and a fair graph-MPNN comparison.

\section{{What is inconclusive}}
KAN-Jastrow viability remains inconclusive until more seeds, full trainable baselines, larger parameter sweeps, DMRG references for chains, and NetKet VMC are run. Interpretability is also pilot-level: a stable-looking curve is not enough without broad seed and parameter stability.

\section{{Pursue or pivot}}
The project is worth pursuing as a rigorous benchmark/software paper if the next phase confirms Hamiltonian validation, baseline fairness, and learned-function stability. It is not yet a method-superiority paper. If KAN-Jastrow is dominated by MLP-pair and smooth Jastrow after full fair comparison, pivot to a negative-results benchmark.

\section{{Safe and unsafe claims}}
Safe claims: the artifact implements and tests a hardware-free exact Rydberg pilot; the proposal's anti-scoop framing remains necessary; optional hardware is unnecessary for main claims. Unsafe claims: first KAN-NQS, general KAN superiority, Graph-KAN value, quantum advantage, IBM validation of Rydberg NQS, and top-journal readiness.

\section{{Required next experiments}}
Run NetKet/Flax VMC, QuSpin cross-checks, TeNPy DMRG for 1D, five to ten seeds, full RBM/MLP/MPNN baselines, geometry transfer, Graph-KAN only after KAN-Jastrow stability, and robustness/failure maps.

\begin{{figure}}[H]\centering\includegraphics[width=.8\linewidth]{{{fig_rel('fig_failure_map.pdf')}}}\caption{{Pilot failure/best-model map from actual result rows.}}\end{{figure}}
\begin{{figure}}[H]\centering\includegraphics[width=.8\linewidth]{{{fig_rel('fig_reproducibility_workflow.pdf')}}}\caption{{Hardware-free reproducibility workflow.}}\end{{figure}}

\section{{Conclusion}}
The artifact moves the project from Level 1 smoke scaffold to a Level 2 CPU exact-reference pilot engine. It is production-coding useful and scientifically honest, but not journal-ready. The next gate is a NetKet/Flax-backed Rydberg chain benchmark with strong trainable baselines and DMRG cross-checks.
\end{{document}}
'''
    out = paper / "production_analysis_report.tex"
    out.write_text(tex, encoding="utf-8")
    return out
