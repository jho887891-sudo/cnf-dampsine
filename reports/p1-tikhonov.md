# P1 CNF-Tikhonov diagnostic ablation

Four full-step runs on the same clean seed-42 curve. This is a Gaussian neural
basis with 33 columns and 31 interior collocation points. Coefficients use
`(AᵀA + λI)⁻¹Aᵀg` during training and evaluation. They are **regularized**
solutions, not direct hard-constraint solves. This one-curve diagnostic is
separate from every square hard-CNF headline result.

| λ | Dense RMSE | Original-matrix condition | Max hard-equation residual | Steps | Label |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1e-10 | 0.178852268 | 266759 | 2.53e-8 | 502 | regularized |
| 1e-8 | 0.178852268 | 266758 | 4.92e-9 | 502 | regularized |
| 1e-6 | 0.178852260 | 266741 | 2.16e-7 | 502 | regularized |
| 1e-4 | 0.178851299 | 264801 | 2.26e-5 | 502 | regularized |

The reconstruction error changes little on this sample. The largest λ
increases the residual of the *unregularized* equations. Small residuals at
other λ values do not change the method label or establish an exact equality
guarantee. All four predictions, summaries, per-step logs and loss plots are
archived with the release.
