# CNF-DampSine experiment report

Physics parameters are oracle values. Hard equalities apply only at finite collocation points.
Dense ODE residual is a separate empirical evaluation. Numerical failures are retained.
Condition-number warnings are reported separately from solve failures.

| Method | Runs | Valid | Warnings | Unstable | Severe | Infeasible | Failed | Mean RMSE | SD RMSE | Median RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cnf_gaussian | 4 | 4 | 1 | 0 | 0 | 0 | 0 | 0.134281 | 0.0890962 | 0.178817 |

## Condition-number warnings (retained, not failures)

- cnf_gaussian at results/p1_encoder_gaussian/p1_encoder_000_sample000/cnf_gaussian; condition=2.475e+08; rmse=0.0006362122406032111

## Failed or numerically unstable runs

