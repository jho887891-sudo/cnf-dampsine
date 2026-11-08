# CNF-DampSine experiment report

Physics parameters are oracle values. Hard equalities apply only at finite collocation points.
Dense ODE residual is a separate empirical evaluation. Numerical failures are retained.
Condition-number warnings are reported separately from solve failures.

| Method | Runs | Valid | Warnings | Unstable | Severe | Infeasible | Failed | Mean RMSE | SD RMSE | Median RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fourier | 3 | 0 | 0 | 1 | 2 | 0 | 0 | n/a | n/a | n/a |

## Condition-number warnings (retained, not failures)

- none

## Failed or numerically unstable runs

- fourier: numerical_failure [severe] at results/p1_layout_fourier/p1_layout_000_sample000/fourier; condition=327806321481585.0; 
- fourier: numerical_failure [severe] at results/p1_layout_fourier/p1_layout_001_sample000/fourier; condition=9.392092743609133e+17; 
- fourier: numerical_failure [unstable] at results/p1_layout_fourier/p1_layout_002_sample000/fourier; condition=56070894010.950195; 
