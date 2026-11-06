# P2 noise, formal protocol: split seed 42 (1 of 5)

This completed split contains 100 independent curves at each noise level for
both CNF-Gaussian and CNF-SIREN. Curve seeds are 1,042,000–1,042,099; the same
underlying curves are used for the 1% and 5% noise settings. All four groups
have 100 saved summaries and 100 loss plots. The other four split seeds are
still running, so this is not the final five-seed estimate.

| Noise | Method | Valid/100 | Mean RMSE | SD RMSE | Median RMSE | Maximum RMSE | Condition warnings | Numerical failures |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1% | CNF-Gaussian | 100 | 0.225007 | 0.0575044 | 0.225709 | 0.529803 | 10 | 0 |
| 1% | CNF-SIREN | 100 | 0.738157 | 6.43235 | 0.00354156 | 64.2761 | 24 | 0 |
| 5% | CNF-Gaussian | 100 | 0.227662 | 0.0647915 | 0.227557 | 0.669249 | 9 | 0 |
| 5% | CNF-SIREN | 100 | 0.175479 | 0.860776 | 0.00873064 | 7.50793 | 27 | 0 |

The CNF-SIREN mean is sensitive to high-error fits. The 1% sample 049 has
RMSE 64.2761 with condition number 4.81×10⁷; the 5% sample 028 has RMSE
7.50793 with condition number 4.65×10⁸. Both are retained. A condition
warning is a numerical diagnostic, while high RMSE indicates a poor fit;
neither is silently excluded from the statistics.

Detailed per-run warnings: [1%](p2-formal-seed42-noise01.md),
[5%](p2-formal-seed42-noise05.md).

Representative figures: [Gaussian 1% loss](p2-formal-seed42-gaussian-001-loss.png),
[Gaussian 5% loss](p2-formal-seed42-gaussian-005-loss.png),
[SIREN 1% loss](p2-formal-seed42-siren-001-loss.png),
[SIREN 5% loss](p2-formal-seed42-siren-005-loss.png).
High-error SIREN reconstructions: [1% sample 049](p2-formal-seed42-outlier-001-reconstruction.png),
[5% sample 028](p2-formal-seed42-outlier-005-reconstruction.png).

The [all-in-one artifact ZIP](https://github.com/jho887891-sudo/cnf-dampsine/releases/download/experiment-artifacts-2026-11-05/cnf-dampsine-all-completed-artifacts.zip)
contains the complete split data, predictions, configurations, logs, and 400
loss plots. Fourier's earlier 40 noise smoke runs had unstable condition
numbers and were not expanded under the SPEC gate; SIREN-PINN has not entered
this formal P2 batch.
