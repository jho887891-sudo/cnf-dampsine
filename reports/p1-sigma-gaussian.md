# CNF-DampSine experiment report

Physics parameters are oracle values. Hard equalities apply only at finite collocation points.
Dense ODE residual is a separate empirical evaluation. Numerical failures are retained.
Condition-number warnings are reported separately from solve failures.

| Method | Runs | Valid | Warnings | Unstable | Severe | Infeasible | Failed | Mean RMSE | SD RMSE | Median RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cnf_gaussian | 5 | 5 | 2 | 0 | 0 | 0 | 0 | 0.145058 | 0.0758333 | 0.178873 |

## Condition-number warnings (retained, not failures)

- cnf_gaussian at results/p1_sigma_gaussian/p1_sigma_003_sample000/cnf_gaussian; condition=1.113e+07; rmse=0.17923308832459717
- cnf_gaussian at results/p1_sigma_gaussian/p1_sigma_004_sample000/cnf_gaussian; condition=5.527e+06; rmse=0.009403873297866927

## Failed or numerically unstable runs

