# P2 formal noise evaluation

Five independent split seeds; 100 curves per noise level and method in each split.
The same curve seeds are used across noise levels within a split.
Every saved run is counted. High-error fits remain in the mean and maximum.

| Noise | Method | Splits complete | Runs | Valid | Mean of split means | SD of split means | Pooled median RMSE | Max RMSE | Warnings | Numerical failures |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1% | siren_pinn | 5/5 | 500 | 500 | 0.208296 | 0.00401353 | 0.206933 | 0.333025 | 0 | 0 |
| 5% | siren_pinn | 5/5 | 500 | 500 | 0.208326 | 0.00404049 | 0.20709 | 0.332322 | 0 | 0 |

A small median with a large mean or maximum indicates severe fit outliers.
Numerical failure counts use the saved run status. Condition warnings are separate.

## Per-split results

| Split seed | Noise | Method | Runs | Valid | Mean RMSE | Median RMSE | Max RMSE | Warnings | Numerical failures |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 1% | siren_pinn | 100 | 100 | 0.210712 | 0.212621 | 0.3132 | 0 | 0 |
| 123 | 1% | siren_pinn | 100 | 100 | 0.214104 | 0.214628 | 0.333025 | 0 | 0 |
| 2026 | 1% | siren_pinn | 100 | 100 | 0.206286 | 0.204319 | 0.313773 | 0 | 0 |
| 3407 | 1% | siren_pinn | 100 | 100 | 0.204321 | 0.204545 | 0.323987 | 0 | 0 |
| 7777 | 1% | siren_pinn | 100 | 100 | 0.206056 | 0.199978 | 0.316993 | 0 | 0 |
| 42 | 5% | siren_pinn | 100 | 100 | 0.210804 | 0.213485 | 0.314074 | 0 | 0 |
| 123 | 5% | siren_pinn | 100 | 100 | 0.214157 | 0.214355 | 0.332322 | 0 | 0 |
| 2026 | 5% | siren_pinn | 100 | 100 | 0.206291 | 0.203456 | 0.313864 | 0 | 0 |
| 3407 | 5% | siren_pinn | 100 | 100 | 0.20437 | 0.204266 | 0.323655 | 0 | 0 |
| 7777 | 5% | siren_pinn | 100 | 100 | 0.206008 | 0.199468 | 0.316119 | 0 | 0 |
