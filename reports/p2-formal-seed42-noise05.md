# CNF-DampSine experiment report

Physics parameters are oracle values. Hard equalities apply only at finite collocation points.
Dense ODE residual is a separate empirical evaluation. Numerical failures are retained.
Condition-number warnings are reported separately from solve failures.

| Method | Runs | Valid | Warnings | Unstable | Severe | Infeasible | Failed | Mean RMSE | SD RMSE | Median RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cnf_gaussian | 100 | 100 | 9 | 0 | 0 | 0 | 0 | 0.227662 | 0.0647915 | 0.227557 |
| cnf_siren | 100 | 100 | 27 | 0 | 0 | 0 | 0 | 0.175479 | 0.860776 | 0.00873064 |

## Condition-number warnings (retained, not failures)

- cnf_gaussian at results/p2_noise_formal/seed42/noise05/sample005/cnf_gaussian; condition=1.351e+06; rmse=0.17697232705794103
- cnf_gaussian at results/p2_noise_formal/seed42/noise05/sample010/cnf_gaussian; condition=1.077e+06; rmse=0.18598876709039913
- cnf_gaussian at results/p2_noise_formal/seed42/noise05/sample027/cnf_gaussian; condition=1.076e+07; rmse=0.33785940630894873
- cnf_gaussian at results/p2_noise_formal/seed42/noise05/sample028/cnf_gaussian; condition=1.669e+06; rmse=0.19635887211317635
- cnf_gaussian at results/p2_noise_formal/seed42/noise05/sample031/cnf_gaussian; condition=2.218e+06; rmse=0.13155458693703723
- cnf_gaussian at results/p2_noise_formal/seed42/noise05/sample039/cnf_gaussian; condition=1.406e+06; rmse=0.2848806836771304
- cnf_gaussian at results/p2_noise_formal/seed42/noise05/sample041/cnf_gaussian; condition=1.112e+06; rmse=0.19180785596001113
- cnf_gaussian at results/p2_noise_formal/seed42/noise05/sample061/cnf_gaussian; condition=2.015e+07; rmse=0.6692491906616326
- cnf_gaussian at results/p2_noise_formal/seed42/noise05/sample094/cnf_gaussian; condition=9.065e+07; rmse=0.2980539104631177
- cnf_siren at results/p2_noise_formal/seed42/noise05/sample000/cnf_siren; condition=4.327e+06; rmse=1.3595712527839503
- cnf_siren at results/p2_noise_formal/seed42/noise05/sample001/cnf_siren; condition=1.370e+06; rmse=0.00945242977660376
- cnf_siren at results/p2_noise_formal/seed42/noise05/sample002/cnf_siren; condition=8.995e+06; rmse=0.22404087745089088
- cnf_siren at results/p2_noise_formal/seed42/noise05/sample005/cnf_siren; condition=6.308e+06; rmse=0.05641209115407781
- cnf_siren at results/p2_noise_formal/seed42/noise05/sample011/cnf_siren; condition=1.046e+06; rmse=0.008559594347287976
- cnf_siren at results/p2_noise_formal/seed42/noise05/sample014/cnf_siren; condition=5.051e+06; rmse=0.15329586051183894
- cnf_siren at results/p2_noise_formal/seed42/noise05/sample020/cnf_siren; condition=5.182e+06; rmse=0.008256976631439159
- cnf_siren at results/p2_noise_formal/seed42/noise05/sample026/cnf_siren; condition=1.708e+08; rmse=0.14457104515650965
- cnf_siren at results/p2_noise_formal/seed42/noise05/sample027/cnf_siren; condition=7.776e+08; rmse=4.16764131826732
- cnf_siren at results/p2_noise_formal/seed42/noise05/sample028/cnf_siren; condition=4.646e+08; rmse=7.507933543249093
- cnf_siren at results/p2_noise_formal/seed42/noise05/sample034/cnf_siren; condition=1.064e+06; rmse=0.013771435726892448
- cnf_siren at results/p2_noise_formal/seed42/noise05/sample045/cnf_siren; condition=1.077e+06; rmse=0.16142994919317488
- cnf_siren at results/p2_noise_formal/seed42/noise05/sample047/cnf_siren; condition=2.088e+06; rmse=0.002644655859474668
- cnf_siren at results/p2_noise_formal/seed42/noise05/sample052/cnf_siren; condition=1.548e+06; rmse=0.11465781210527559
- cnf_siren at results/p2_noise_formal/seed42/noise05/sample054/cnf_siren; condition=3.040e+06; rmse=0.00437425797965606
- cnf_siren at results/p2_noise_formal/seed42/noise05/sample058/cnf_siren; condition=1.297e+06; rmse=0.14030384095386608
- cnf_siren at results/p2_noise_formal/seed42/noise05/sample061/cnf_siren; condition=4.389e+06; rmse=0.00301742491368
- cnf_siren at results/p2_noise_formal/seed42/noise05/sample066/cnf_siren; condition=1.381e+06; rmse=0.29139984473930564
- cnf_siren at results/p2_noise_formal/seed42/noise05/sample069/cnf_siren; condition=3.713e+06; rmse=0.003552351110341189
- cnf_siren at results/p2_noise_formal/seed42/noise05/sample072/cnf_siren; condition=2.384e+06; rmse=0.0031968236682630986
- cnf_siren at results/p2_noise_formal/seed42/noise05/sample079/cnf_siren; condition=1.177e+06; rmse=0.020113535805931688
- cnf_siren at results/p2_noise_formal/seed42/noise05/sample081/cnf_siren; condition=1.948e+06; rmse=0.004397613205442927
- cnf_siren at results/p2_noise_formal/seed42/noise05/sample082/cnf_siren; condition=1.189e+06; rmse=0.006920011357617866
- cnf_siren at results/p2_noise_formal/seed42/noise05/sample090/cnf_siren; condition=2.168e+06; rmse=0.0052276202638647665
- cnf_siren at results/p2_noise_formal/seed42/noise05/sample093/cnf_siren; condition=3.609e+06; rmse=0.00377997260884838
- cnf_siren at results/p2_noise_formal/seed42/noise05/sample095/cnf_siren; condition=2.822e+07; rmse=0.013592371716092472
- cnf_siren at results/p2_noise_formal/seed42/noise05/sample098/cnf_siren; condition=7.569e+06; rmse=0.005378349228683443

## Failed or numerically unstable runs
