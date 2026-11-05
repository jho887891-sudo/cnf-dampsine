# CNF-DampSine experiment report

Physics parameters are oracle values. Hard equalities apply only at finite collocation points.
Dense ODE residual is a separate empirical evaluation. Numerical failures are retained.
Condition-number warnings are reported separately from solve failures.

| Method | Runs | Valid | Warnings | Unstable | Severe | Infeasible | Failed | Mean RMSE | SD RMSE | Median RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fourier | 1 | 0 | 0 | 1 | 0 | 0 | 0 | n/a | n/a | n/a |
| fourier (rectangular) | 2 | 0 | 0 | 0 | 0 | 0 | 2 | n/a | n/a | n/a |

## Condition-number warnings (retained, not failures)

- none

## Failed or numerically unstable runs

- fourier: numerical_failure [unstable] at results/p1_rectangular_fourier/p1_rectangular_001_sample000/fourier; condition=56070894010.950195; 
- fourier (rectangular): failed [unknown] at results/p1_rectangular_fourier/p1_rectangular_000_sample000/fourier; condition=n/a; TypeError: dict.update() got multiple values for keyword argument 'hard_residual_max'
- fourier (rectangular): failed [unknown] at results/p1_rectangular_fourier/p1_rectangular_002_sample000/fourier; condition=n/a; TypeError: dict.update() got multiple values for keyword argument 'hard_residual_max'
