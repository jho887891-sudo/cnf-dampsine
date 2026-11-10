# P1 Gaussian condition-weight ablation

One clean seed-42 curve per setting; K=33, M=31, 32 observations, 2048 dense
evaluation points. These are full-step runs with early stopping, not five-seed
statistics. All four solves are numerically valid.

| Condition weight | Dense RMSE | Condition number | Training steps |
| ---: | ---: | ---: | ---: |
| 0 | 0.178853218 | 633683 | 502 |
| 1e-8 | 0.178853218 | 551725 | 502 |
| 1e-6 | 0.178852268 | 266758 | 502 |
| 1e-4 | 0.178838221 | 87574 | 666 |

The highest weight reduces the saved condition number on this curve, while
the dense RMSE changes little. The four run directories contain predictions,
diagnostics, training logs and loss plots. The condition-weight values are
engineering settings, not source-paper constants.
