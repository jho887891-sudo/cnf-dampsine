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
- Observation count: [Gaussian](p1-observations-gaussian.md), [SIREN](p1-observations-siren.md)
- Collocation layout: [Gaussian](p1-layout-gaussian.md), [SIREN](p1-layout-siren.md)
- [Gaussian width](p1-sigma-gaussian.md), [Gaussian encoder](p1-encoder-gaussian.md)
- [SIREN basis family](p1-basis-family.md), [SIREN frequency scale](p1-omega.md)
- [Fourier rectangular systems](p1-rectangular-fourier.md)

## P2 noise

- [Gaussian 20 samples per noise level](p2-noise-gaussian.md)
- [Fourier 20 samples per noise level](p2-noise-fourier.md)
- [CNF-SIREN 20 samples per noise level](p2-noise-siren.md)
- CNF-SIREN noise 0.01: [loss](p2-siren-noise-001-loss.png), [reconstruction](p2-siren-noise-001-reconstruction.png)
- CNF-SIREN noise 0.05: [loss](p2-siren-noise-005-loss.png), [reconstruction](p2-siren-noise-005-reconstruction.png)

The SIREN full-step noise run finished all 40 samples. Its pooled mean RMSE
includes a large outlier; the report retains the per-run conditioning warnings.
The OOD analysis remains deferred.

## P3 frozen shared-basis transfer

- [Reproducible five-seed report](p3-repro-multiseed.md) and [seed-level RMSE plot](p3-seed-means.png)

Each seed uses 800 training, 100 validation, and 100 test curves, with 5,000
global training steps. Earlier exploratory P3 runs omitted an explicit model
initialization seed; this report uses only the corrected reproducible runs.
The P3 global-step runs did not record a per-step loss log; the release assets
contain frozen checkpoints, split definitions, and per-curve validation/test
metrics, but no P3 loss curve is claimed.
