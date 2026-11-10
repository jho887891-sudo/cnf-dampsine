# P1 CNF collocation density, fixed K=33

One clean curve (seed 42) per setting, 32 observations and 2048 dense test
points. Gaussian anchors remain at the 31 Chebyshev interior locations plus
two boundary locations as M changes. Rectangular fits use a minimum-norm
solution at M=15 and least squares at M=63. Hard feasibility requires a
maximum constraint residual no greater than 1e-8.

| Method | M | Matrix | Solver | Status | Hard feasible | Max hard residual | Condition number | Dense RMSE |
| --- | ---: | --- | --- | --- | --- | ---: | ---: | ---: |
| CNF-Gaussian | 15 | 17×33 | minimum norm | ok | yes | 1.74e-14 | 1.39e4 | 0.179168 |
| CNF-Gaussian | 31 | 33×33 | square solve | ok | yes | 3.08e-13 | 2.67e5 | 0.178852 |
| CNF-Gaussian | 63 | 65×33 | least squares | infeasible | no | 0.044247 | 1.27 | 0.178895 |
| CNF-SIREN | 15 | 17×33 | minimum norm | ok | yes | 2.53e-14 | 5.54e3 | 0.001348 |
| CNF-SIREN | 31 | 33×33 | square solve | ok | yes | 2.09e-12 | 6.88e5 | 0.070192 |
| CNF-SIREN | 63 | 65×33 | least squares | infeasible | no | 0.044212 | 7.33e3 | 0.178881 |

The M=63 fits have finite RMSE and ordinary condition numbers, but they do
not satisfy the specified hard equations. Their predictions remain archived
as infeasible diagnostics and are excluded from valid hard-CNF accuracy
claims. These are single-curve P1 scan results, not five-seed statistics.
All six training logs, predictions, and loss plots are in the artifact ZIP.
