# P1 SIREN-PINN capacity and ODE-weight scans

Single clean curve (seed 42), 32 observations, 31 soft ODE collocation
points and 2048 dense evaluation points. Width and depth scans keep the
other network settings at the MVP values (width 64, three hidden layers).
The ODE-weight scan changes only lambda_ode. These are full-step single-curve
runs, not five-seed statistics.

| Scan | Setting | Dense RMSE | Training steps | Status |
| --- | --- | ---: | ---: | --- |
| Width | 32 | 0.174729 | 10000 | ok |
| Width | 64 | 0.170322 | 8780 | ok |
| Width | 128 | 0.170859 | 4732 | ok |
| Hidden layers | 2 | 0.157654 | 10000 | ok |
| Hidden layers | 3 | 0.170322 | 8780 | ok |
| Hidden layers | 4 | 0.182725 | 2476 | ok |
| ODE weight | 0.1 | 0.136856 | 7785 | ok |
| ODE weight | 1 | 0.170322 | 8780 | ok |
| ODE weight | 10 | 0.180875 | 3641 | ok |
| ODE weight | 100 | 0.181609 | 4116 | ok |

The width-64/layer-3 baseline appears twice in the raw runs (once for the
width scan, once for lambda_ode=1) and yields the same RMSE. Width 32 and
two hidden layers reached the 10000-step cap; their `ok` status means the
saved run was numerically valid, not that an early-stopping criterion fired.
All nine predictions, training logs and loss plots are archived.
