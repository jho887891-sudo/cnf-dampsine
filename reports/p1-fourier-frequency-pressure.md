# CNF-DampSine experiment report

Physics parameters are oracle values. Hard equalities apply only at finite collocation points.
Dense ODE residual is a separate empirical evaluation. Numerical failures are retained.
Condition-number warnings are reported separately from solve failures.

| Method | Runs | Valid | Warnings | Unstable | Severe | Infeasible | Failed | Mean RMSE | SD RMSE | Median RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fourier | 2 | 0 | 0 | 2 | 0 | 0 | 0 | n/a | n/a | n/a |

## Condition-number warnings (retained, not failures)

- none

## Failed or numerically unstable runs

- fourier: numerical_failure [unstable] at results/p1_fourier_frequency_pressure/p1_fourier_frequency_pressure_000_sample000/fourier; condition=23824548012.392384; 
- fourier: numerical_failure [unstable] at results/p1_fourier_frequency_pressure/p1_fourier_frequency_pressure_001_sample000/fourier; condition=60939254146.490524; 
