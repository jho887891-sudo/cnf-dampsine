# P1 formal clean evaluation

5 independent split seeds; 100 clean curves per method in each split.
Every saved run is counted. High-error fits remain in the mean and maximum.

| Noise | Method | Splits complete | Runs | Valid | Mean of split means | SD of split means | Pooled median RMSE | Max RMSE | Warnings | Numerical failures |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0% | cnf_gaussian | 5/5 | 500 | 500 | 0.219449 | 0.00398409 | 0.216385 | 0.416349 | 29 | 0 |
| 0% | cnf_siren | 5/5 | 500 | 500 | 0.156219 | 0.139889 | 0.00264859 | 20.6975 | 132 | 0 |
| 0% | siren_pinn | 5/5 | 500 | 500 | 0.208331 | 0.00405389 | 0.206979 | 0.333366 | 0 | 0 |

A small median with a large mean or maximum indicates severe fit outliers.
Numerical failure counts use the saved run status. Condition warnings are separate.

## Per-split results

| Split seed | Noise | Method | Runs | Valid | Mean RMSE | Median RMSE | Max RMSE | Warnings | Numerical failures |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 0% | cnf_gaussian | 100 | 100 | 0.222579 | 0.223552 | 0.401527 | 4 | 0 |
| 123 | 0% | cnf_gaussian | 100 | 100 | 0.224515 | 0.224377 | 0.342772 | 7 | 0 |
| 2026 | 0% | cnf_gaussian | 100 | 100 | 0.218395 | 0.212847 | 0.416349 | 6 | 0 |
| 3407 | 0% | cnf_gaussian | 100 | 100 | 0.216727 | 0.215058 | 0.376801 | 7 | 0 |
| 7777 | 0% | cnf_gaussian | 100 | 100 | 0.215031 | 0.206416 | 0.325826 | 5 | 0 |
| 42 | 0% | cnf_siren | 100 | 100 | 0.24863 | 0.00303068 | 19.7789 | 31 | 0 |
| 123 | 0% | cnf_siren | 100 | 100 | 0.0864176 | 0.00296639 | 4.5173 | 26 | 0 |
| 2026 | 0% | cnf_siren | 100 | 100 | 0.064749 | 0.00124121 | 2.71411 | 25 | 0 |
| 3407 | 0% | cnf_siren | 100 | 100 | 0.0261392 | 0.00314866 | 0.379024 | 21 | 0 |
| 7777 | 0% | cnf_siren | 100 | 100 | 0.355157 | 0.00254026 | 20.6975 | 29 | 0 |
| 42 | 0% | siren_pinn | 100 | 100 | 0.210807 | 0.212634 | 0.311697 | 0 | 0 |
| 123 | 0% | siren_pinn | 100 | 100 | 0.214167 | 0.214413 | 0.333366 | 0 | 0 |
| 2026 | 0% | siren_pinn | 100 | 100 | 0.206284 | 0.203597 | 0.313229 | 0 | 0 |
| 3407 | 0% | siren_pinn | 100 | 100 | 0.204269 | 0.204225 | 0.323774 | 0 | 0 |
| 7777 | 0% | siren_pinn | 100 | 100 | 0.206127 | 0.199908 | 0.31681 | 0 | 0 |
