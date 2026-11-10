# CNF-DampSine: completed non-OOD experiments

This report summarizes the archived P0–P3 results as of 2026-11-10. The
experiment definitions come from [SPEC](../docs/SPEC/CNF_DampSine_SPEC.md);
the [implementation plan](../docs/IMPLEMENTATION_PLAN/IMPLEMENTATION_PLAN.md)
is a proposed workflow. Gamma, omega and offset are saved oracle parameters
used to construct the sample ODE and boundaries. They are not inferred from
observations. The condition-number and hard-residual cutoffs are project
engineering criteria, not claims made by the CNF source paper.

## Results and statistical scope

| Group | Completed scope | Main finding and qualification |
| --- | --- | --- |
| P0 | One seed-42 curve, four methods | Gaussian, CNF-SIREN and PINN valid; Fourier condition ~5.61e10, numerical failure. See [P0](p0.md). |
| P1 clean main | Five split seeds × 100 curves for each of three methods | Gaussian 500/500 valid, mean RMSE 0.219449, 29 warnings; CNF-SIREN 500/500 valid, mean RMSE 0.156219, 132 warnings and nine RMSE >1 (max 20.6975); PINN 500/500 valid, mean RMSE 0.208331, zero warnings. See [full report](p1-clean-formal-all.md). |
| P1 one-curve sweeps | Capacity, observations, layout, sigma, encoder, SIREN basis/omega/width/layers, PINN width/layers/ODE weight | Full-step scans, one curve per setting; not five-seed estimates. CNF-SIREN width 32 has RMSE 46.1286. See the [report index](README.md). |
| P1 additional numerical ablations | Gaussian condition weight 0/1e-8/1e-6/1e-4; Fourier frequency steps π/2, π, 2π and 24π/40π high-frequency pressure; separate CNF-Tikhonov λ=1e-10/1e-8/1e-6/1e-4 | The four Gaussian condition-weight runs are valid, with little RMSE change; all five Fourier runs exceed the unstable condition threshold. All four Tikhonov runs are labeled regularized. See [condition weight](p1-condition-weight-detail.md), [frequency step](p1-fourier-frequency-step.md), [pressure](p1-fourier-pressure-detail.md), and [Tikhonov](p1-tikhonov.md). |
| P1 fixed K=33 | M=15/31/63 for Fourier, both CNF methods, and PINN soft collocation | CNF M=63 both hard-infeasible (max residual ~0.044); Fourier M=31/63 unstable. PINN is a soft-loss comparison. See [CNF rectangular report](p1-rectangular-cnf.md). |
| P2 noise | 1% and 5%, five split seeds × 100 curves per noise level for Gaussian, CNF-SIREN and PINN | 3,000 saved method/curve/noise runs numerically valid; CNF-SIREN has severe fit outliers. Fourier 40-run smoke is condition-unstable and was not promoted to the formal comparison. See [CNF](p2-formal-multiseed.md) and [PINN](p2-pinn-formal-multiseed.md). |
| P3 frozen shared basis | Gaussian and CNF-SIREN, five seeds, 800 train / 100 validation / 100 test curves per split, 5,000 global steps | Mean test RMSE 0.212987 and 0.117374, respectively. The logged rerun matches the reproducible runs. See [P3](p3-logged-multiseed.md). |

The P2 OOD frequency and decay tests remain deferred at the user's request.
No OOD generalization claim is supported by these results.
The SPEC lists Tikhonov coefficients as a separately labeled diagnostic
ablation. The new CNF-Tikhonov runs use a regularized solve; all reported main
CNF results use the direct solve. No Tikhonov run is mixed into the hard-CNF
tables.

## Constraints and failure interpretation

CNF and Fourier solve hard equations at finitely many collocation and boundary
points. Their saved matrix residual, boundary error, and 2048-point dense ODE
residual measure different properties. PINN uses a weighted ODE penalty and
does not provide a hard equality guarantee. A finite reconstruction RMSE does
not turn a hard-infeasible rectangular solve into a success. Condition warnings
remain separate from numerical failures; all high-error fits remain in the
reported means and maxima. The project rule marks condition numbers above
1e10 as unstable candidates. Square direct solves remain the main CNF
protocol; rectangular minimum-norm and least-squares systems are reported
separately. The figures and tables do not establish unknown-physics recovery.

## SPEC section 8.1 figure audit

| Required diagnostic | Archived figure | Scope |
| --- | --- | --- |
| Four-method reconstruction | [P0 comparison](p0-comparison.png) | Fourier prediction retained with its failure label in P0 report |
| Pointwise absolute error | [P0 Gaussian](p0-gaussian-absolute-error.png) | One representative valid run |
| Dense ODE residual, log scale | [P0 Gaussian](p0-gaussian-dense-residual.png) | One representative valid run |
| Both boundary close-ups | [Left](p0-gaussian-left-boundary.png), [right](p0-gaussian-right-boundary.png) | One representative valid run |
| Condition vs K | [P1 capacity](condition-vs-k.png) | Single-curve sweep |
| Condition vs M | [P1 fixed-K](condition-vs-m.png) | Infeasible M=63 cases labeled |
| RMSE vs K | [P1 capacity](p1-capacity-rmse.png) | Single-curve sweep; use report for failure status |
| Dense residual vs K | [P1 capacity](dense-residual-vs-k.png) | Single-curve sweep |
| Gaussian sigma sensitivity | [P1 sigma](gaussian-sigma-sensitivity.png) | Single-curve sweep |
| SIREN omega sensitivity | [P1 omega](siren-omega-sensitivity.png) | Single-curve sweep |
| Beta coefficient heatmap | [P0 beta](p0-beta-heatmap.png) | Saved solved coefficients |
| Constraint matrix heatmap | [P0 Fourier matrix](p0-fourier-matrix-heatmap.png) | Reconstructed fixed dictionary; unstable solve labeled |
| Singular-value spectrum | [P0 Gaussian](p0-gaussian-singular-values.png) | One representative valid run |
| Noise degradation | [Formal P1/P2 means](noise-degradation.png) | Mean of five split means; severe CNF-SIREN outliers retained |
| Time vs relative L2 | [P0 time/error](p0-time-error.png) | Descriptive three-point plot, not a statistically established Pareto frontier |

All raw summaries, predictions, logs, loss plots and diagnostics are in the
[single artifact ZIP](https://github.com/jho887891-sudo/cnf-dampsine/releases/download/experiment-artifacts-2026-11-05/cnf-dampsine-all-completed-artifacts.zip).
