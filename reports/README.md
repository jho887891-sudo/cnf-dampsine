# CNF-DampSine experiment reports

These are compact, versioned snapshots of completed experiment groups. Raw run
directories, predictions, and model checkpoints remain in `results/` on the
experiment host and are excluded from Git by `.gitignore`. Download the
[experiment artifact release](https://github.com/jho887891-sudo/cnf-dampsine/releases/tag/experiment-artifacts-2026-11-05)
for the archived raw data, predictions, logs, figures, and frozen checkpoints.
For a single download, use the [all-in-one ZIP](https://github.com/jho887891-sudo/cnf-dampsine/releases/download/experiment-artifacts-2026-11-05/cnf-dampsine-all-completed-artifacts.zip).

## P0

- [Four-method report](p0.md) and [reconstruction comparison](p0-comparison.png)
- Loss: [Gaussian](p0-gaussian-loss.png), [CNF-SIREN](p0-siren-loss.png), [SIREN-PINN](p0-pinn-loss.png)

## P1 single-seed scans

- Capacity: [CNF](p1-capacity-cnf.md), [Fourier](p1-capacity-fourier.md), [RMSE plot](p1-capacity-rmse.png)
- Example loss curves: [K=65 CNF-SIREN](p1-k65-siren-loss.png), [Gaussian σ=0.2](p1-sigma-020-loss.png)
- Observation count: [Gaussian](p1-observations-gaussian.md), [SIREN](p1-observations-siren.md),
  [SIREN-PINN](p1-observations-pinn.md), [Fourier](p1-observations-fourier.md)
- Collocation layout: [Gaussian](p1-layout-gaussian.md), [SIREN](p1-layout-siren.md),
  [Fourier](p1-layout-fourier.md)
- [Gaussian width](p1-sigma-gaussian.md), [Gaussian encoder](p1-encoder-gaussian.md)
- [SIREN basis family](p1-basis-family.md), [SIREN frequency scale](p1-omega.md)
- [Fourier rectangular systems](p1-rectangular-fourier.md)
- [SIREN-PINN collocation density](p1-rectangular-pinn.md): M=15/31/63
  on one clean curve, using soft ODE loss.

The three Fourier observation-count scans and three layout scans are all
numerically unstable by the specified condition threshold; their reports
retain each run. The three SIREN-PINN observation-count scans are valid.

## P1 clean main configuration

- [CNF-Gaussian five-seed formal report](p1-clean-gaussian-formal.md) and
  [seed-level RMSE figure](p1-clean-gaussian-seed-means.png): 500/500 valid
  curves with 29 condition warnings. Its raw data and 500 loss plots are in
  the artifact ZIP.
- [CNF-Gaussian and CNF-SIREN five-seed report](p1-clean-cnf-formal.md) and
  [seed-level RMSE figure](p1-clean-cnf-seed-means.png), plus
  [two severe fit outliers](p1-clean-siren-outliers.png): CNF-SIREN has 500/500
  numerically valid curves, 132 condition warnings, and nine fits with RMSE
  above 1. Its full loss plots and predictions are in the artifact ZIP.
- [All three methods, five-seed formal report](p1-clean-formal-all.md) and
  [seed-level RMSE figure](p1-clean-formal-all.png): SIREN-PINN finished
  500/500 numerically valid curves with no condition warnings. Its 500 loss
  plots and predictions are in the artifact ZIP.

## P2 noise

- [Gaussian 20 samples per noise level](p2-noise-gaussian.md)
- [Fourier 20 samples per noise level](p2-noise-fourier.md)
- [CNF-SIREN 20 samples per noise level](p2-noise-siren.md)
- CNF-SIREN noise 0.01: [loss](p2-siren-noise-001-loss.png), [reconstruction](p2-siren-noise-001-reconstruction.png)
- CNF-SIREN noise 0.05: [loss](p2-siren-noise-005-loss.png), [reconstruction](p2-siren-noise-005-reconstruction.png)

The SIREN full-step noise run finished all 40 samples. Its pooled mean RMSE
includes a large outlier; the report retains the per-run conditioning warnings.
The OOD analysis remains deferred.

- [Formal five-seed P2 report](p2-formal-multiseed.md) and [seed-level RMSE figure](p2-formal-seed-means.png):
  100 curves per split and noise level for Gaussian and CNF-SIREN. All 2,000
  runs completed. High-error fits remain in the statistics.
- [Seed 42 detail](p2-formal-seed42.md), including representative loss curves
  and high-error reconstructions. The release ZIP contains all 2,000 loss logs
  and figures.
- [SIREN-PINN 20-curve smoke at each noise level](p2-pinn-noise-smoke.md):
  [1% loss](p2-pinn-noise-001-loss.png), [5% loss](p2-pinn-noise-005-loss.png).
- [SIREN-PINN five-seed formal report](p2-pinn-formal-multiseed.md) and
  [seed-level RMSE figure](p2-pinn-formal-seed-means.png): 100 curves per
  split and noise level, 1,000 runs total. Its 1,000 loss logs and figures
  are in the artifact ZIP.

## P3 frozen shared-basis transfer

- [Reproducible five-seed report](p3-repro-multiseed.md) and [seed-level RMSE plot](p3-seed-means.png)
- [Five-seed rerun with loss logs](p3-logged-multiseed.md), [seed-level RMSE plot](p3-logged-seed-means.png)
- Representative loss: [Gaussian seed 42](p3-gaussian-seed42-loss.png), [CNF-SIREN seed 42](p3-siren-seed42-loss.png). All ten loss plots and logs are in the artifact ZIP.

Each seed uses 800 training, 100 validation, and 100 test curves, with 5,000
global training steps. Earlier exploratory P3 runs omitted an explicit model
initialization seed; this report uses only the corrected reproducible runs.
The first P3 global-step runs did not record a per-step loss log. The logged
five-seed rerun records loss every 10 steps. Its validation and test summaries
match the first reproducible runs for all ten method/seed pairs.
