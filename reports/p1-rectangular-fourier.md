# CNF-DampSine experiment report

Physics parameters are oracle values. Hard equalities apply only at finite collocation points.
Dense ODE residual is a separate empirical evaluation. Numerical failures are retained.
Condition-number warnings are reported separately from solve failures.

| Method | Runs | Valid | Warnings | Unstable | Severe | Infeasible | Failed | Mean RMSE | SD RMSE | Median RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fourier | 1 | 0 | 0 | 1 | 0 | 0 | 0 | n/a | n/a | n/a |
| fourier (rectangular) | 2 | 1 | 0 | 1 | 0 | 0 | 0 | 0.175721 | n/a | 0.175721 |

The rectangular extension uses K=33 basis functions. At M=15, the 17×33
system used a minimum-norm solution: maximum hard residual 1.33e-14 and
condition number 5.27e3. At M=63, the 65×33 least-squares system met the
specified 1e-8 hard-feasibility tolerance (residual 9.09e-10), but its
condition number 4.77e10 is in the unstable band. The M=31 square case is
also unstable. These are one-curve feasibility diagnostics, not formal
multi-seed accuracy statistics.

## Condition-number warnings (retained, not failures)

- none

## Failed or numerically unstable runs

- fourier: numerical_failure [unstable] at results/p1_rectangular_fourier/p1_rectangular_001_sample000/fourier; condition=56070894010.950195; 
- fourier (rectangular): numerical_failure [unstable] at results/p1_rectangular_fourier/p1_rectangular_002_sample000/fourier; condition=47698942139.06643; 
