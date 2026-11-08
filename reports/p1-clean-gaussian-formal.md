# P1 formal clean evaluation

5 independent split seeds; 100 clean curves per method in each split.
Every saved run is counted. High-error fits remain in the mean and maximum.

| Noise | Method | Splits complete | Runs | Valid | Mean of split means | SD of split means | Pooled median RMSE | Max RMSE | Warnings | Numerical failures |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0% | cnf_gaussian | 5/5 | 500 | 500 | 0.219449 | 0.00398409 | 0.216385 | 0.416349 | 29 | 0 |

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
