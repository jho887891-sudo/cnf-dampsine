# P2 formal noise evaluation

Five independent split seeds; 100 curves per noise level and method in each split.
The same curve seeds are used across noise levels within a split.
Every saved run is counted. High-error fits remain in the mean and maximum.

| Noise | Method | Splits complete | Runs | Valid | Mean of split means | SD of split means | Pooled median RMSE | Max RMSE | Warnings | Numerical failures |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1% | cnf_gaussian | 5/5 | 500 | 500 | 0.223524 | 0.00814952 | 0.216374 | 1.3716 | 32 | 0 |
| 1% | cnf_siren | 5/5 | 500 | 500 | 0.212977 | 0.296036 | 0.00221617 | 64.2761 | 114 | 0 |
| 5% | cnf_gaussian | 5/5 | 500 | 500 | 0.236173 | 0.0347288 | 0.216793 | 6.87125 | 42 | 0 |
| 5% | cnf_siren | 5/5 | 500 | 500 | 0.151573 | 0.120603 | 0.00745943 | 27.527 | 171 | 0 |

A small median with a large mean or maximum indicates severe fit outliers.
Numerical failure counts use the saved run status. Condition warnings are separate.

## Per-split results

| Split seed | Noise | Method | Runs | Valid | Mean RMSE | Median RMSE | Max RMSE | Warnings | Numerical failures |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 1% | cnf_gaussian | 100 | 100 | 0.225007 | 0.225709 | 0.529803 | 10 | 0 |
| 123 | 1% | cnf_gaussian | 100 | 100 | 0.236624 | 0.22559 | 1.3716 | 6 | 0 |
| 2026 | 1% | cnf_gaussian | 100 | 100 | 0.219388 | 0.213259 | 0.589758 | 6 | 0 |
| 3407 | 1% | cnf_gaussian | 100 | 100 | 0.221481 | 0.213916 | 1.0299 | 6 | 0 |
| 7777 | 1% | cnf_gaussian | 100 | 100 | 0.215123 | 0.206416 | 0.327736 | 4 | 0 |
| 42 | 1% | cnf_siren | 100 | 100 | 0.738157 | 0.00354156 | 64.2761 | 24 | 0 |
| 123 | 1% | cnf_siren | 100 | 100 | 0.0497407 | 0.00160983 | 1.7713 | 18 | 0 |
| 2026 | 1% | cnf_siren | 100 | 100 | 0.146187 | 0.00184235 | 10.0438 | 22 | 0 |
| 3407 | 1% | cnf_siren | 100 | 100 | 0.0713021 | 0.00436559 | 1.43755 | 25 | 0 |
| 7777 | 1% | cnf_siren | 100 | 100 | 0.0594967 | 0.00175195 | 1.33121 | 25 | 0 |
| 42 | 5% | cnf_gaussian | 100 | 100 | 0.227662 | 0.227557 | 0.669249 | 9 | 0 |
| 123 | 5% | cnf_gaussian | 100 | 100 | 0.297664 | 0.228474 | 6.87125 | 9 | 0 |
| 2026 | 5% | cnf_gaussian | 100 | 100 | 0.216589 | 0.212469 | 0.358839 | 8 | 0 |
| 3407 | 5% | cnf_gaussian | 100 | 100 | 0.223322 | 0.215047 | 1.11641 | 7 | 0 |
| 7777 | 5% | cnf_gaussian | 100 | 100 | 0.215627 | 0.206416 | 0.346127 | 9 | 0 |
| 42 | 5% | cnf_siren | 100 | 100 | 0.175479 | 0.00873064 | 7.50793 | 27 | 0 |
| 123 | 5% | cnf_siren | 100 | 100 | 0.345561 | 0.00650591 | 27.527 | 28 | 0 |
| 2026 | 5% | cnf_siren | 100 | 100 | 0.0335582 | 0.00725276 | 0.491734 | 29 | 0 |
| 3407 | 5% | cnf_siren | 100 | 100 | 0.12468 | 0.00820266 | 5.33539 | 38 | 0 |
| 7777 | 5% | cnf_siren | 100 | 100 | 0.0785872 | 0.00720768 | 1.38936 | 49 | 0 |
