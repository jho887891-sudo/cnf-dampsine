# CNF-DampSine experiment report

Physics parameters are oracle values. Hard equalities apply only at finite collocation points.
Dense ODE residual is a separate empirical evaluation. Numerical failures are retained.
Condition-number warnings are reported separately from solve failures.

| Method | Runs | Valid | Warnings | Unstable | Severe | Infeasible | Failed | Mean RMSE | SD RMSE | Median RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cnf_siren | 3 | 3 | 1 | 0 | 0 | 0 | 0 | 0.0128823 | 0.0206418 | 0.00189918 |

## Condition-number warnings (retained, not failures)

- cnf_siren at results/p1_layout_cnf_siren/p1_layout_001_sample000/cnf_siren; condition=2.667e+09; rmse=0.0018991804221237662

## Failed or numerically unstable runs

