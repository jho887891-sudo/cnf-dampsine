# CNF-DampSine experiment report

Physics parameters are oracle values. Hard equalities apply only at finite collocation points.
Dense ODE residual is a separate empirical evaluation. Numerical failures are retained.
Condition-number warnings are reported separately from solve failures.

| Method | Runs | Valid | Warnings | Unstable | Severe | Infeasible | Failed | Mean RMSE | SD RMSE | Median RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cnf_gaussian | 40 | 40 | 5 | 0 | 0 | 0 | 0 | 0.236469 | 0.0536876 | 0.225349 |

## Condition-number warnings (retained, not failures)

- cnf_gaussian at results/p2_noise_gaussian_full20/p2_noise_gaussian_full20_000_sample015/cnf_gaussian; condition=1.172e+06; rmse=0.3590619115636303
- cnf_gaussian at results/p2_noise_gaussian_full20/p2_noise_gaussian_full20_000_sample017/cnf_gaussian; condition=2.826e+06; rmse=0.2343120409496771
- cnf_gaussian at results/p2_noise_gaussian_full20/p2_noise_gaussian_full20_001_sample013/cnf_gaussian; condition=1.839e+07; rmse=0.29267161706973044
- cnf_gaussian at results/p2_noise_gaussian_full20/p2_noise_gaussian_full20_001_sample015/cnf_gaussian; condition=1.518e+06; rmse=0.37431636654477357
- cnf_gaussian at results/p2_noise_gaussian_full20/p2_noise_gaussian_full20_001_sample016/cnf_gaussian; condition=1.281e+06; rmse=0.29475405662966464

## Failed or numerically unstable runs

