# CNF-DampSine experiment report

Physics parameters are oracle values. Hard equalities apply only at finite collocation points.
Dense ODE residual is a separate empirical evaluation. Numerical failures are retained.
Condition-number warnings are reported separately from solve failures.

| Method | Runs | Valid | Warnings | Unstable | Severe | Infeasible | Failed | Mean RMSE | SD RMSE | Median RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fourier | 3 | 1 | 0 | 1 | 1 | 0 | 0 | 8.49031e-06 | n/a | 8.49031e-06 |

## Condition-number warnings (retained, not failures)

- none

## Failed or numerically unstable runs

- fourier: numerical_failure [unstable] at results/p1_capacity_fourier/p1_capacity_001_sample000/fourier; condition=56070894010.950195; 
- fourier: numerical_failure [severe] at results/p1_capacity_fourier/p1_capacity_002_sample000/fourier; condition=6.064741624933961e+17; 
