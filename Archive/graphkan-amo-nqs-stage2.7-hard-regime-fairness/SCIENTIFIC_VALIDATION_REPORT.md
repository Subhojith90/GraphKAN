# Scientific Validation Report

## Validation status

The production artifact validates the hardware-free Rydberg CPU path through deterministic geometry generation, dense and sparse Hamiltonian builders, analytic diagonal-limit checks, exact diagonalisation, exact full-basis variational evaluation, observable computation, paired statistics, and figure/table regeneration.

## Hamiltonian gate

Decision gate A is `PASS`.

Evidence:

- Dense and sparse Rydberg Hamiltonians agree in tests.
- The $\Omega=0$ spectrum matches diagonal analytic energies.
- Basis indexing is invertible.
- Local-energy/full-matrix consistency tests pass.
- ED references were generated for configured chains, squares, and ladder instances.

## Exact-fullsum pilot

Energy-density error medians:

| model          |    median |    ci_low |   ci_high |   n |
|:---------------|----------:|----------:|----------:|----:|
| kan_jastrow    | 0.0162026 | 0.0139179 | 0.0199632 |  54 |
| radial_jastrow | 0.0162026 | 0.0139179 | 0.0199632 |  54 |
| mlp_pair       | 0.0169754 | 0.0152844 | 0.0199632 |  54 |
| product        | 0.111226  | 0.0715244 | 0.151655  |  54 |

Correlation-MAE medians:

| model          |    median |     ci_low |   ci_high |   n |
|:---------------|----------:|-----------:|----------:|----:|
| radial_jastrow | 0.0141411 | 0.00949334 | 0.0154967 |  54 |
| kan_jastrow    | 0.0142018 | 0.00944577 | 0.0155877 |  54 |
| mlp_pair       | 0.0154597 | 0.00943997 | 0.0191919 |  54 |
| product        | 0.0259775 | 0.021337   | 0.0326998 |  54 |

The exact-fullsum pilot shows that KAN-Jastrow, radial Jastrow, and MLP-pair feature models all produce sensible small-system variational values. It does not by itself establish a production VMC result.

## Rydberg chain CPU pilot

Energy-density error medians:

| model          |     median |     ci_low |    ci_high |   n |
|:---------------|-----------:|-----------:|-----------:|----:|
| mlp_pair       | 0.00502355 | 0.00336081 | 0.00686093 |  16 |
| smooth_jastrow | 0.00540912 | 0.00413179 | 0.0102461  |  16 |
| kan_jastrow    | 0.00978985 | 0.00342562 | 0.0229917  |  16 |
| product        | 0.0271758  | 0.0144947  | 0.0413855  |  16 |

Correlation-MAE medians:

| model          |     median |     ci_low |    ci_high |   n |
|:---------------|-----------:|-----------:|-----------:|----:|
| mlp_pair       | 0.00453003 | 0.00212911 | 0.00779872 |  16 |
| smooth_jastrow | 0.00517573 | 0.00221938 | 0.0138679  |  16 |
| product        | 0.00966544 | 0.00169155 | 0.0203444  |  16 |
| kan_jastrow    | 0.0104092  | 0.00213004 | 0.0211973  |  16 |

In the reduced two-seed CPU pilot, MLP-pair and smooth Jastrow have lower median energy-density error and lower median correlation error than KAN-Jastrow. KAN-Jastrow is therefore not safe to describe as superior or competitive in a final-paper sense.

## Paired comparisons

| comparison                       | metric               |   median_delta_baseline_minus_candidate |       ci_low |      ci_high |   n_pairs |     p_value | test     |      p_holm |
|:---------------------------------|:---------------------|----------------------------------------:|-------------:|-------------:|----------:|------------:|:---------|------------:|
| mlp_pair_minus_kan_jastrow       | energy_density_error |                             2.85401e-07 |  3.56291e-08 |  2.64004e-05 |        71 | 4.56451e-06 | wilcoxon | 2.73871e-05 |
| product_minus_kan_jastrow        | energy_density_error |                             0.0580243   |  0.0314464   |  0.111529    |        71 | 2.42922e-13 | wilcoxon | 1.94337e-12 |
| radial_jastrow_minus_kan_jastrow | energy_density_error |                             2.40086e-15 |  2.00361e-16 |  5.40054e-14 |        55 | 0.0004758   | wilcoxon | 0.002379    |
| smooth_jastrow_minus_kan_jastrow | energy_density_error |                            -0.000275074 | -0.000944525 |  0.00101986  |        16 | 1           | wilcoxon | 1           |
| mlp_pair_minus_kan_jastrow       | correlation_mae      |                            -2.33219e-09 | -5.7763e-07  |  4.86072e-09 |        71 | 0.578352    | wilcoxon | 1           |
| product_minus_kan_jastrow        | correlation_mae      |                             0.00706893  |  0.0005684   |  0.00949367  |        71 | 5.94576e-08 | wilcoxon | 4.16203e-07 |
| radial_jastrow_minus_kan_jastrow | correlation_mae      |                             3.00545e-10 | -6.81806e-10 |  1.27051e-09 |        55 | 0.81452     | wilcoxon | 1           |
| smooth_jastrow_minus_kan_jastrow | correlation_mae      |                            -0.000366726 | -0.00537321  | -2.77119e-06 |        16 | 0.12973     | wilcoxon | 0.518921    |

The paired comparisons are dominated by the exact-fullsum pilot where KAN and radial-Jastrow feature spaces are very close; the reduced Rydberg chain pilot is the more relevant warning signal for future development.

## Interpretability pilot

Median KAN-Jastrow curve alignment: 0.999539.
Median KAN-Jastrow stability score: 0.999670.

The gauge-fixed learned functions are seed-stable in this pilot feature-model setting, but this is not yet evidence of physical interpretability. A final claim requires correlation matching, scale tracking across larger parameter sweeps, stronger baselines, and no cherry-picked curves.

## Graph-KAN status

Graph-KAN was not executed. Decision gate D is `NOT_RUN`. No claim about Graph-KAN value is safe.

## Hardware status

No IBM or AMO hardware result was used. The package is hardware-free.

## Overall scientific classification

Current evidence level: Level 2 CPU exact-reference/pilot benchmark. It is not PRA-ready, MLST-ready, CPC-ready, Quantum-ready, or PRX-Quantum-ready.
