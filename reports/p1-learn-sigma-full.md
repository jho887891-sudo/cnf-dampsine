# P1 Gaussian learned-sigma ablation

Single clean curve (seed 42), K=33, M=31, 32 observations and 2048 dense
evaluation points. The Gaussian basis starts at sigma=0.05 with
`learn_sigma=true`. The full-step run finished with status `ok`, dense RMSE
0.178858, and 502 training steps. Its initial-sigma-fixed counterpart in
the P1 sigma scan has RMSE 0.178852 on the same curve. This one-curve
comparison shows no material accuracy change; it is not a five-seed result.

The prediction, diagnostics, training log and loss plot are archived.
