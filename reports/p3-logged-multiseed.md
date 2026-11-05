# P3 multi-seed shared-basis transfer

Each split trains on 800 curves and evaluates 100 independent test curves with a frozen basis.
Mean and SD below summarize the per-split test means. Incomplete methods remain labeled as such.

| Method | Completed seeds | Mean of test means | SD across seeds | Pooled median RMSE | Solver failures | Unstable conditions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cnf_gaussian | 5/5 | 0.212987 | 0.00454337 | 0.207627 | 0 | 0 |
| cnf_siren | 5/5 | 0.117374 | 0.107253 | 0.141224 | 0 | 0 |
