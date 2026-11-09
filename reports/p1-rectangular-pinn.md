# P1 SIREN-PINN collocation density

Single clean curve (seed 42), 32 observations, 2048 dense evaluation points.
SIREN-PINN applies a soft ODE loss at M interior collocation points; K=33 is
the fixed basis count for the separate CNF/Fourier comparisons. PINN has no
rectangular hard-constraint matrix or hard-feasibility claim.

| Interior points M | Status | Dense RMSE | Training steps |
| ---: | --- | ---: | ---: |
| 15 | ok | 0.173901 | 7275 |
| 31 | ok | 0.170322 | 8780 |
| 63 | ok | 0.167388 | 5305 |

All three runs completed without numerical failure. These are single-curve
scan results; the raw predictions, logs, and three loss plots are archived.
