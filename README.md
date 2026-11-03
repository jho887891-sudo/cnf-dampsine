# CNF-DampSine

Independent PyTorch experiment suite for oracle-parameter reconstruction of FunDiff-style damped sine curves. Four methods are available: CNF-Gaussian, CNF-SIREN, Fourier collocation, and SIREN-PINN. The first three solve finite hard constraints; SIREN-PINN uses soft ODE and boundary losses.

Run from the project root:

```sh
python3 scripts/run_mvp.py --config configs/mvp.yaml --methods all --output-root results/mvp
python3 scripts/run_sweep.py --suite p1obs --config configs/sweeps/p1_observations.yaml --output-root results/p1obs
python3 scripts/run_sweep.py --suite p2noise --config configs/sweeps/p2_noise.yaml --output-root results/p2noise
python3 -m evaluation.plots results/mvp
python3 -m evaluation.report results/mvp --output results/mvp/report.md
python3 -m pytest -q
```

For a quick integration check, add `--steps-override 1` to the run command. Such a run does not produce meaningful trained-model comparisons. Numerical failures and inapplicable Gaussian fallback cases are retained in each `summary.json`.

The supplied SPEC's full statistical experiments require 100 test curves across five seeds. The P2 gamma OOD range must be supplied explicitly. Current remote environment differs from the reference versions listed in `environment.md`.
