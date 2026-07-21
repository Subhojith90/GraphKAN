Stage 4C Evidence Release
Created UTC: 20260702T064016Z

Scope:
- 630/630 registered VMC trajectories completed.
- Candidate: rbf_endpoint_sum_21p.
- Exact-valid patch cells: 18/18.
- Sample-valid patch cells: 18/18.
- Pareto non-dominated exact-valid cells: 16/18 (88.89%).
- Predefined non-domination threshold: 70%.
- Repaired Stage 4C gate status: PASS.

Interpretation:
The compact endpoint-sum RBF ansatz is a robust Pareto trade-off candidate
under the registered Stage 4C protocol. It is not universally superior:
stronger 128-parameter MLP controls dominate it in two N=8, low-detuning cells.

Excluded:
- results_stage4c/fast_triage is intentionally excluded because it used an
  under-budget engineering diagnostic protocol and failed the registered gates.
