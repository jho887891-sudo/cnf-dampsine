# CNF-DampSine experiment report

Physics parameters are oracle values. Hard equalities apply only at finite collocation points.
Dense ODE residual is a separate empirical evaluation. Numerical failures are retained.
Condition-number warnings are reported separately from solve failures.

| Method | Runs | Valid | Warnings | Unstable | Severe | Infeasible | Failed | Mean RMSE | SD RMSE | Median RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cnf_gaussian | 100 | 100 | 10 | 0 | 0 | 0 | 0 | 0.225007 | 0.0575044 | 0.225709 |
| cnf_siren | 100 | 100 | 24 | 0 | 0 | 0 | 0 | 0.738157 | 6.43235 | 0.00354156 |

## Condition-number warnings (retained, not failures)

- cnf_gaussian at results/p2_noise_formal/seed42/noise01/sample005/cnf_gaussian; condition=7.597e+06; rmse=0.16748574098959965
- cnf_gaussian at results/p2_noise_formal/seed42/noise01/sample007/cnf_gaussian; condition=1.108e+07; rmse=0.37912008500274835
- cnf_gaussian at results/p2_noise_formal/seed42/noise01/sample014/cnf_gaussian; condition=2.901e+06; rmse=0.23543247348372887
- cnf_gaussian at results/p2_noise_formal/seed42/noise01/sample031/cnf_gaussian; condition=4.010e+06; rmse=0.13302763032663412
- cnf_gaussian at results/p2_noise_formal/seed42/noise01/sample047/cnf_gaussian; condition=1.900e+06; rmse=0.24389133192566978
- cnf_gaussian at results/p2_noise_formal/seed42/noise01/sample061/cnf_gaussian; condition=2.775e+06; rmse=0.16025601973865605
- cnf_gaussian at results/p2_noise_formal/seed42/noise01/sample066/cnf_gaussian; condition=1.319e+06; rmse=0.18843726680646603
- cnf_gaussian at results/p2_noise_formal/seed42/noise01/sample068/cnf_gaussian; condition=1.220e+06; rmse=0.2484131878886264
- cnf_gaussian at results/p2_noise_formal/seed42/noise01/sample077/cnf_gaussian; condition=2.414e+06; rmse=0.27896388796977756
- cnf_gaussian at results/p2_noise_formal/seed42/noise01/sample088/cnf_gaussian; condition=2.904e+06; rmse=0.5298033116592727
- cnf_siren at results/p2_noise_formal/seed42/noise01/sample001/cnf_siren; condition=1.280e+06; rmse=0.009376974186840777
- cnf_siren at results/p2_noise_formal/seed42/noise01/sample005/cnf_siren; condition=2.886e+06; rmse=0.00951668400779384
- cnf_siren at results/p2_noise_formal/seed42/noise01/sample011/cnf_siren; condition=1.232e+06; rmse=0.0032231895295178454
- cnf_siren at results/p2_noise_formal/seed42/noise01/sample014/cnf_siren; condition=1.343e+06; rmse=0.0628637952646066
- cnf_siren at results/p2_noise_formal/seed42/noise01/sample019/cnf_siren; condition=4.076e+06; rmse=0.06567973393803014
- cnf_siren at results/p2_noise_formal/seed42/noise01/sample020/cnf_siren; condition=3.774e+06; rmse=0.001750657794696405
- cnf_siren at results/p2_noise_formal/seed42/noise01/sample025/cnf_siren; condition=1.206e+06; rmse=0.03414271105167509
- cnf_siren at results/p2_noise_formal/seed42/noise01/sample026/cnf_siren; condition=1.851e+07; rmse=0.19477769541751613
- cnf_siren at results/p2_noise_formal/seed42/noise01/sample027/cnf_siren; condition=1.083e+07; rmse=0.07358546365333125
- cnf_siren at results/p2_noise_formal/seed42/noise01/sample031/cnf_siren; condition=6.654e+06; rmse=0.0011535742914838781
- cnf_siren at results/p2_noise_formal/seed42/noise01/sample034/cnf_siren; condition=2.941e+06; rmse=0.41201583980530254
- cnf_siren at results/p2_noise_formal/seed42/noise01/sample035/cnf_siren; condition=5.604e+06; rmse=0.003590100664930825
- cnf_siren at results/p2_noise_formal/seed42/noise01/sample045/cnf_siren; condition=1.325e+06; rmse=0.17320535855781063
- cnf_siren at results/p2_noise_formal/seed42/noise01/sample048/cnf_siren; condition=4.320e+07; rmse=3.924151719926308
- cnf_siren at results/p2_noise_formal/seed42/noise01/sample049/cnf_siren; condition=4.809e+07; rmse=64.2760601717754
- cnf_siren at results/p2_noise_formal/seed42/noise01/sample050/cnf_siren; condition=1.090e+07; rmse=0.6002423137826044
- cnf_siren at results/p2_noise_formal/seed42/noise01/sample061/cnf_siren; condition=4.217e+06; rmse=0.007048849538030884
- cnf_siren at results/p2_noise_formal/seed42/noise01/sample063/cnf_siren; condition=1.052e+06; rmse=0.17742741335083623
- cnf_siren at results/p2_noise_formal/seed42/noise01/sample066/cnf_siren; condition=4.336e+06; rmse=0.29107813111558367
- cnf_siren at results/p2_noise_formal/seed42/noise01/sample068/cnf_siren; condition=1.007e+06; rmse=0.03692825696561012
- cnf_siren at results/p2_noise_formal/seed42/noise01/sample080/cnf_siren; condition=1.306e+06; rmse=0.16307366853884106
- cnf_siren at results/p2_noise_formal/seed42/noise01/sample082/cnf_siren; condition=4.265e+06; rmse=0.0041724203095364724
- cnf_siren at results/p2_noise_formal/seed42/noise01/sample087/cnf_siren; condition=9.180e+06; rmse=1.7015472695474658
- cnf_siren at results/p2_noise_formal/seed42/noise01/sample095/cnf_siren; condition=1.316e+07; rmse=0.09489870925132567

## Failed or numerically unstable runs
