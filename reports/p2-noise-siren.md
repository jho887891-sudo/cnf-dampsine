# CNF-DampSine experiment report

Physics parameters are oracle values. Hard equalities apply only at finite collocation points.
Dense ODE residual is a separate empirical evaluation. Numerical failures are retained.
Condition-number warnings are reported separately from solve failures.

| Method | Runs | Valid | Warnings | Unstable | Severe | Infeasible | Failed | Mean RMSE | SD RMSE | Median RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cnf_siren | 40 | 40 | 10 | 0 | 0 | 0 | 0 | 0.0541779 | 0.174415 | 0.00427568 |

## Condition-number warnings (retained, not failures)

- cnf_siren at results/p2_noise_siren_full20/p2_noise_siren_full20_000_sample001/cnf_siren; condition=2.795e+07; rmse=1.0286694221442854
- cnf_siren at results/p2_noise_siren_full20/p2_noise_siren_full20_000_sample004/cnf_siren; condition=1.837e+06; rmse=0.13421199131971598
- cnf_siren at results/p2_noise_siren_full20/p2_noise_siren_full20_000_sample005/cnf_siren; condition=6.726e+06; rmse=0.014867831006407305
- cnf_siren at results/p2_noise_siren_full20/p2_noise_siren_full20_000_sample013/cnf_siren; condition=5.536e+06; rmse=0.017637857799286883
- cnf_siren at results/p2_noise_siren_full20/p2_noise_siren_full20_001_sample002/cnf_siren; condition=2.722e+06; rmse=0.006957411124933831
- cnf_siren at results/p2_noise_siren_full20/p2_noise_siren_full20_001_sample004/cnf_siren; condition=2.407e+06; rmse=0.007615366730355613
- cnf_siren at results/p2_noise_siren_full20/p2_noise_siren_full20_001_sample006/cnf_siren; condition=1.522e+06; rmse=0.049422746578313526
- cnf_siren at results/p2_noise_siren_full20/p2_noise_siren_full20_001_sample012/cnf_siren; condition=1.072e+06; rmse=0.0008422355914305132
- cnf_siren at results/p2_noise_siren_full20/p2_noise_siren_full20_001_sample014/cnf_siren; condition=1.185e+06; rmse=0.004390106344943249
- cnf_siren at results/p2_noise_siren_full20/p2_noise_siren_full20_001_sample017/cnf_siren; condition=3.026e+06; rmse=0.0025772156397136157

## Failed or numerically unstable runs

