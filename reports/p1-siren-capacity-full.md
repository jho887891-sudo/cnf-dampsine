# P1 CNF-SIREN capacity scan

Single clean curve (seed 42), K=33, M=31, 32 observations and 2048 dense
evaluation points. Only one model setting changes per row. These are
full-step P1 runs, not five-seed statistics.

| Width | Hidden layers | Status | Dense RMSE | Condition number | Training steps |
| ---: | ---: | --- | ---: | ---: | ---: |
| 21 | 1 | ok | 0.070192 | 6.88e5 | 501 |
| 32 | 1 | ok, condition warning | 46.128569 | 3.65e7 | 514 |
| 64 | 1 | ok | 0.119936 | 1.59e5 | 535 |
| 21 | 2 | ok, condition warning | 0.283963 | 6.98e6 | 1587 |
| 21 | 3 | ok, condition warning | 0.185047 | 4.61e7 | 565 |

The width-32 fit has a severe accuracy error despite a numerically valid
solve; its RMSE remains in the table. The other two warning rows also remain.
Predictions, diagnostics, training logs and all five loss plots are archived.
