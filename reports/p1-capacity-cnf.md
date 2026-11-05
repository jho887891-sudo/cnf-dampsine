# CNF-DampSine experiment report

Physics parameters are oracle values. Hard equalities apply only at finite collocation points.
Dense ODE residual is a separate empirical evaluation. Numerical failures are retained.
Condition-number warnings are reported separately from solve failures.

| Method | Runs | Valid | Warnings | Unstable | Severe | Infeasible | Failed | Mean RMSE | SD RMSE | Median RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cnf_gaussian | 3 | 3 | 1 | 0 | 0 | 0 | 0 | 0.17457 | 0.00828936 | 0.178852 |
| cnf_siren | 3 | 3 | 2 | 0 | 0 | 0 | 0 | 32.5894 | 56.4367 | 0.0112227 |

## Condition-number warnings (retained, not failures)

- cnf_gaussian at results/p1_capacity_cnf/p1_capacity_002_sample000/cnf_gaussian; condition=3.181e+07; rmse=0.1650154875480788
- cnf_siren at results/p1_capacity_cnf/p1_capacity_000_sample000/cnf_siren; condition=3.683e+07; rmse=97.75687881658627
- cnf_siren at results/p1_capacity_cnf/p1_capacity_002_sample000/cnf_siren; condition=1.447e+06; rmse=1.7972774971956674e-05

## Failed or numerically unstable runs

