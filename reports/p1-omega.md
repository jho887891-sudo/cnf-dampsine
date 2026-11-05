# CNF-DampSine experiment report

Physics parameters are oracle values. Hard equalities apply only at finite collocation points.
Dense ODE residual is a separate empirical evaluation. Numerical failures are retained.
Condition-number warnings are reported separately from solve failures.

| Method | Runs | Valid | Warnings | Unstable | Severe | Infeasible | Failed | Mean RMSE | SD RMSE | Median RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cnf_siren | 4 | 4 | 0 | 0 | 0 | 0 | 0 | 0.0164002 | 0.0191874 | 0.0144513 |
| siren_pinn | 4 | 4 | 0 | 0 | 0 | 0 | 0 | 0.146173 | 0.038871 | 0.160718 |

## Condition-number warnings (retained, not failures)

- none

## Failed or numerically unstable runs

