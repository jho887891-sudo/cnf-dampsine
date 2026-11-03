# CNF-DampSine Implementation Plan

> **For agentic workers:** Read `../SPEC/CNF_DampSine_SPEC.md` first.
> Execute Tasks 1--6 to freeze shared interfaces. Tasks 7--10 are four
> independent method tracks and should be developed in parallel after
> the shared interfaces are stable. Each task is test-driven and
> independently reviewable.

**Goal:** Implement the four-method CNF-DampSine experiment suite
defined by the SPEC, including hard ODE/Dirichlet constraints for CNF
and Fourier, and soft ODE/BC constraints for SIREN-PINN.

**Architecture:** Build shared data, physics, basis-derivative,
constraint-system, solver, evaluation, and experiment-runner interfaces.
After those contracts are stable, implement CNF-Gaussian, CNF-SIREN,
Fourier Collocation, and SIREN-PINN as four parallel method tracks using
the same sample metadata and evaluation API.

**Tech Stack:** Python 3.9; PyTorch 1.13.0; PyTorch Lightning 1.8.1;
SciPy 1.9.3; NumPy; pytest; YAML configuration.

**Spec:** `../SPEC/CNF_DampSine_SPEC.md`

## Global Constraints

-   Use an independent PyTorch environment; do not import the FunDiff
    JAX/Flax package.
-   Reproduce the FunDiff damp_sine formula and parameter distributions;
    save A, gamma, omega, b for every sample.
-   MVP: 128 sensors, 32 observations, 2048 dense evaluation points,
    seed 42, zero noise.
-   MVP hard system: M=31 Chebyshev interior points, two Dirichlet rows,
    K=33.
-   CNF square main path uses `torch.linalg.solve(A, g)`; never form an
    explicit matrix inverse.
-   CNF beta is solved on every iteration and is not an Adam parameter.
-   Higher derivatives must retain the autograd graph
    (`create_graph=True`).
-   CNF-Gaussian MVP: sigma=0.05, PositionalEncoder n_freqs=3, Adam
    lr=1e-3, 5000 steps, patience 500, condition weight 1e-6.
-   CNF-SIREN MVP: official `hypernet_siren`, width 21, one hidden
    layer, omega0=30, Adam lr=3e-4, 5000 initial steps (10k scan cap),
    patience 500.
-   Fourier MVP: 33 columns, constant + 16 sin/cos pairs, max frequency
    16π, direct solve, no optimizer.
-   SIREN-PINN MVP: width 64, three hidden layers, omega0=30, Adam
    lr=1e-4, lambda_data=1, lambda_ODE=1, lambda_BC=100, 10k initial
    steps (20k cap), patience 1000.
-   Noisy observations are regression targets only; ODE and boundary
    hard constraints remain clean.
-   Do not claim finite collocation constraints imply exact
    continuous-domain ODE satisfaction.
-   Mark all numerical failures; do not silently replace the strict
    solver with Ridge/Tikhonov.
-   Distinguish report-derived choices from engineering decisions in
    configs and final results.

## Review Focus

1.  Right boundary near zero causes an initial-condition fallback with
    two operators at the same x=0; test basis routing and
    duplicate-anchor rejection.
2.  P1 fixed-K collocation scans produce rectangular matrices; test
    underdetermined/overdetermined handling and feasibility labels.
3.  Basis derivative code can accidentally sum across K columns; test
    each column's first/second derivative independently.
4.  Noise can leak into hard RHS values; test that noisy observations do
    not alter ODE or clean boundary constraints.
5.  Ill-conditioned or rank-deficient A can produce unstable beta; test
    diagnostics, failure flags, and no silent Ridge fallback.

------------------------------------------------------------------------

## Repository Layout

``` text
CNF_DampSine/
├── configs/
│   ├── mvp.yaml
│   ├── gaussian.yaml
│   ├── cnf_siren.yaml
│   ├── fourier.yaml
│   └── siren_pinn.yaml
├── data/
│   ├── __init__.py
│   ├── generator.py
│   ├── dataset.py
│   └── splits/
├── models/
│   ├── __init__.py
│   ├── bases/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── gaussian_kernel.py
│   │   ├── hypernet_siren.py
│   │   └── fourier_basis.py
│   ├── cnf.py
│   └── siren_pinn.py
├── physics/
│   ├── __init__.py
│   ├── damped_oscillator.py
│   ├── constraints.py
│   └── collocation.py
├── solvers/
│   ├── __init__.py
│   ├── differentiable_solve.py
│   └── diagnostics.py
├── training/
│   ├── __init__.py
│   ├── train_cnf.py
│   ├── train_pinn.py
│   └── early_stop.py
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py
│   ├── evaluate.py
│   └── plots.py
├── scripts/
│   ├── run_mvp.py
│   ├── run_sweep.py
│   └── aggregate.py
├── tests/
├── results/
├── pyproject.toml
└── README.md
```

------------------------------------------------------------------------

## Task 1: Verify reference implementations and establish the environment

**Files:** - Create: `requirements-cnf-ref.txt` - Create:
`environment.md` - Create: `scripts/inspect_cnf_reference.py` - Test:
`tests/test_environment.py`

**Interfaces:** - Produces a compatibility table for official CNF
classes and constructor arguments. - Confirms actual names/limits for
`KernelBasis`, `HypernetBasis`, `SeparateBasis`, `hypernet_siren`,
`sepbasis_siren`, Gaussian encoder, SIREN initialization, derivative
utilities, and solve path.

-   [ ] **Step 1: Write failing source-contract tests**
    -   Add tests for required package versions and reference-adapter
        import.
    -   Add a small inspection test that records actual basis factory
        names and signatures; do not hard-code guessed signatures before
        inspecting source.
-   [ ] **Step 2: Verify the tests fail**
    -   Run: `python -m pytest tests/test_environment.py -q`
    -   Expected: FAIL until environment metadata and adapter exist.
-   [ ] **Step 3: Inspect the official CNF repository**
    -   Read `network.py`, `nbf.py`, `diff_utils.py`, and
        `requirements.txt`.
    -   Record actual class names, constructor arguments, basis
        registry/factory keys, shape conventions, and initialization
        rules in `environment.md`.
    -   Confirm the PyTorch 1.13.0 path supports differentiable
        `torch.linalg.solve` and required derivative operations.
    -   Do not copy a method name from the report without checking
        source.
-   [ ] **Step 4: Pin the reference environment**
    -   Set Python 3.9, torch 1.13.0, pytorch-lightning 1.8.1, scipy
        1.9.3.
    -   Keep FunDiff/JAX in a separate environment.
    -   Record platform, CUDA availability, package versions, and git
        commit.
-   [ ] **Step 5: Run the tests**
    -   Run: `python -m pytest tests/test_environment.py -q`
    -   Expected: PASS.

------------------------------------------------------------------------

## Task 2: Build reproducible damp_sine data generation

**Files:** - Create: `data/generator.py` - Create: `data/dataset.py` -
Create: `configs/mvp.yaml` - Test: `tests/test_data_generator.py`

**Interfaces:** -
`sample_damp_sine(n_functions: int, n_sensors: int, seed: int) -> dict` -
`make_splits(train_count: int, val_count: int, test_count: int, seed: int) -> dict` -
Returned arrays include `x`, `y_clean`, `y_noisy`, `params` with columns
`[A, gamma, omega, b]`.

-   [ ] **Step 1: Write failing data tests**
    -   `test_sample_shapes`: expect x/y `[F,128,1]` for the MVP sensor
        count.
    -   `test_parameter_ranges`: assert A∈\[0.5,1\], gamma∈\[2,4\],
        omega∈\[6π,8π\], b∈\[-0.5,0.5\].
    -   `test_seed_reproducibility`: same seed yields identical
        parameters and samples.
    -   `test_split_parameter_disjointness`: train/val/test parameter
        rows do not duplicate.
    -   `test_noise_scale`: noise std follows `eta * std(y_clean)` and
        eta=0 is exactly clean.
-   [ ] **Step 2: Verify tests fail**
    -   Run: `python -m pytest tests/test_data_generator.py -q`
    -   Expected: FAIL because the generator is not implemented.
-   [ ] **Step 3: Implement generator and split logic**
    -   Reproduce the FunDiff formula without importing the FunDiff
        package.
    -   Generate 128 uniform sensors.
    -   Save the four true parameters.
    -   Add deterministic observation selection for 16/32/64 points.
    -   Add noise only to observation values; preserve clean full sensor
        signal.
-   [ ] **Step 4: Run tests**
    -   Run: `python -m pytest tests/test_data_generator.py -q`
    -   Expected: PASS.

------------------------------------------------------------------------

## Task 3: Implement the analytic physics operator and collocation points

**Files:** - Create: `physics/damped_oscillator.py` - Create:
`physics/collocation.py` - Test: `tests/test_physics.py`

**Interfaces:** - `damped_sine(x, A, gamma, omega, b) -> f` -
`damped_sine_q(x, A, gamma, omega) -> q` -
`analytic_q_derivatives(x, A, gamma, omega) -> (q, q1, q2)` -
`apply_damped_operator(q, q1, q2, gamma, omega) -> residual` -
`make_collocation_points(M, kind, seed) -> Tensor[M,1]`

-   [ ] **Step 1: Write failing analytic tests**
    -   `test_analytic_derivatives_match_closed_form`.
    -   `test_true_function_satisfies_ode`.
    -   `test_chebyshev_points_are_interior_and_sorted`.
    -   `test_uniform_random_chebyshev_shapes`.
    -   Use float64; require analytic ODE max residual \< 1e-9 on the
        dense test grid.
-   [ ] **Step 2: Run tests**
    -   Run: `python -m pytest tests/test_physics.py -q`
    -   Expected: FAIL before implementation.
-   [ ] **Step 3: Implement formulas**
    -   Implement exact q, q′, q″ expressions from SPEC.
    -   Implement the linear operator.
    -   Implement Chebyshev interior formula with m=1..M.
    -   Ensure no collocation point equals either boundary.
-   [ ] **Step 4: Run tests**
    -   Run: `python -m pytest tests/test_physics.py -q`
    -   Expected: PASS.

------------------------------------------------------------------------

## Task 4: Define the basis contract and column-wise automatic differentiation

**Files:** - Create: `models/bases/base.py` - Create:
`solvers/derivatives.py` - Test: `tests/test_basis_derivatives.py`

**Interfaces:** - `Basis.forward(x: Tensor[N,1]) -> Tensor[N,K]` -
`evaluate_basis_derivatives(basis, x) -> (phi0, phi1, phi2)`, each
`[N,K]`.

-   [ ] **Step 1: Write failing shape and derivative tests**
    -   `test_basis_contract_shape`.
    -   `test_columnwise_derivatives_do_not_sum_basis_columns`.
    -   `test_second_derivative_retains_graph`.
    -   Compare derivative outputs against an analytic toy basis with
        multiple columns having different derivatives.
-   [ ] **Step 2: Run tests**
    -   Run: `python -m pytest tests/test_basis_derivatives.py -q`
    -   Expected: FAIL.
-   [ ] **Step 3: Implement the basis protocol and derivative utility**
    -   Use per-column autograd or a correct batched Jacobian/vmap
        implementation.
    -   Preserve `create_graph=True` for first and second derivatives.
    -   Return tensors with identical `[N,K]` shape.
    -   Do not sum over the basis dimension.
-   [ ] **Step 4: Run tests**
    -   Run: `python -m pytest tests/test_basis_derivatives.py -q`
    -   Expected: PASS.

------------------------------------------------------------------------

## Task 5: Implement hard-constraint assembly, solver, and diagnostics

**Files:** - Create: `physics/constraints.py` - Create:
`solvers/differentiable_solve.py` - Create: `solvers/diagnostics.py` -
Test: `tests/test_constraints.py`

**Interfaces:** -
`build_hard_constraint_system(basis, x_interior, gamma, omega, boundary_spec) -> (A, g)` -
`solve_coefficients(A, g, method="direct") -> (beta, diagnostics)` -
`constraint_residual(A, beta, g) -> scalar` -
`matrix_diagnostics(A) -> {rank, cond_2, singular_values, solver_status}`

-   [ ] **Step 1: Write failing constraint tests**
    -   `test_mvp_matrix_shape_is_33_by_33`.
    -   `test_ode_rows_precede_two_dirichlet_rows`.
    -   `test_rhs_contains_clean_boundary_values`.
    -   `test_direct_solver_satisfies_constraints`.
    -   `test_no_explicit_inverse_used`.
    -   `test_near_zero_right_boundary_selects_initial_condition_fallback`.
    -   `test_standard_gaussian_rejects_duplicate_anchor_operator_pair`.
-   [ ] **Step 2: Run tests**
    -   Run: `python -m pytest tests/test_constraints.py -q`
    -   Expected: FAIL.
-   [ ] **Step 3: Implement matrix assembly**
    -   Build M ODE rows and two boundary rows in SPEC order.
    -   Return A `[M+2,K]` and g `[M+2,1]`.
    -   Keep boundary RHS clean and based on saved true parameters.
    -   Implement configurable near-zero boundary branch; route
        repeated-anchor initial-value constraints away from ordinary
        Gaussian KernelBasis.
-   [ ] **Step 4: Implement the strict solver**
    -   Square main path uses `torch.linalg.solve`.
    -   Do not compute `torch.inverse`.
    -   Record solve residual, rank, condition number, and singular
        values.
    -   Preserve autograd through solve.
-   [ ] **Step 5: Implement diagnostics and failure states**
    -   Apply SPEC condition thresholds.
    -   QR/SVD may diagnose but must not silently replace the primary
        solver.
    -   Numerical failure must be explicit in returned diagnostics.
-   [ ] **Step 6: Run tests**
    -   Run: `python -m pytest tests/test_constraints.py -q`
    -   Expected: PASS.

------------------------------------------------------------------------

## Task 6: Freeze shared contracts and configs

**Files:** - Create: `configs/gaussian.yaml` - Create:
`configs/cnf_siren.yaml` - Create: `configs/fourier.yaml` - Create:
`configs/siren_pinn.yaml` - Create: `configs/mvp.yaml` - Test:
`tests/test_config_contracts.py`

**Interfaces:** - All methods consume the same sample dictionary and
evaluation API. - All method configs expose method name, seed, K/M/Nobs,
noise, boundary mode, solver, and evaluation grid.

-   [ ] **Step 1: Write failing config tests**
    -   `test_mvp_common_data_protocol`.
    -   `test_method_hyperparameters_match_spec`.
    -   `test_all_method_configs_have_same_physics_protocol`.
-   [ ] **Step 2: Run tests**
    -   Run: `python -m pytest tests/test_config_contracts.py -q`
    -   Expected: FAIL.
-   [ ] **Step 3: Write configs**
    -   Set exact MVP values from SPEC.
    -   Separate report-derived values from engineering flags in YAML
        comments/metadata.
    -   Ensure all four methods share seed=42, 128 sensors, 32
        observations, M=31, dense grid=2048, zero noise.
-   [ ] **Step 4: Run tests**
    -   Run: `python -m pytest tests/test_config_contracts.py -q`
    -   Expected: PASS.

------------------------------------------------------------------------

## Parallel Method Tracks

After Tasks 1--6 freeze the shared contracts, implement Tasks 7--10 as
four independent tracks. These are not a sequential "finish one method
before starting the next" roadmap. Develop them in parallel where the
Codex execution environment supports independent agents/worktrees;
otherwise keep the task boundaries independent and integrate only
against the frozen interfaces.

## Task 7: CNF-Gaussian method track

**Files:** - Create: `models/bases/gaussian_kernel.py` - Create:
`models/cnf.py` - Test: `tests/test_gaussian.py`

**Interfaces:** - `GaussianKernelBasis.forward(x) -> Phi[N,33]` -
`CNFModel.forward(x_query) -> q_hat[N,1]` -
`CNFModel.solve_beta(sample, x_interior) -> beta[33,1]`

-   [ ] **Step 1: Write failing tests**
    -   `test_gaussian_basis_shape_and_finite_values`.
    -   `test_gaussian_anchor_count_and_order`.
    -   `test_sigma_initialization_is_005`.
    -   `test_encoder_uses_positional_n_freqs_3`.
    -   `test_cnf_gaussian_beta_is_solved_not_optimized`.
    -   `test_cnf_gaussian_gradient_reaches_encoder`.
-   [ ] **Step 2: Run tests**
    -   Run: `python -m pytest tests/test_gaussian.py -q`
    -   Expected: FAIL.
-   [ ] **Step 3: Adapt the official CNF Gaussian implementation**
    -   Inspect and reuse/adapt the actual official `KernelBasis`,
        Gaussian kernel and encoder behavior.
    -   Keep 33 anchors: 31 Chebyshev interior locations + two boundary
        locations.
    -   Use sigma init 0.05 and PositionalEncoder n_freqs=3.
    -   Expose fixed/learnable sigma as explicit config; MVP uses fixed
        sigma.
    -   Keep basis encoder parameters trainable; beta is solved every
        iteration.
-   [ ] **Step 4: Implement CNF Gaussian forward/solve path**
    -   Call shared matrix builder and differentiable solver.
    -   Use regression + condition-number objective from SPEC.
    -   Ensure prediction is `Phi_query @ beta`.
-   [ ] **Step 5: Run tests**
    -   Run: `python -m pytest tests/test_gaussian.py -q`
    -   Expected: PASS.

------------------------------------------------------------------------

## Task 8: CNF-SIREN method track

**Files:** - Create: `models/bases/hypernet_siren.py` - Modify:
`models/cnf.py` - Test: `tests/test_cnf_siren.py`

**Interfaces:** - `HypernetSirenBasis.forward(x) -> Phi[N,33]` - Basis
factory records official basis identifier and constructor metadata.

-   [ ] **Step 1: Write failing tests**
    -   `test_hypernet_siren_basis_shape`.
    -   `test_hypernet_siren_uses_width_21_one_hidden_layer`.
    -   `test_hypernet_siren_uses_omega0_30`.
    -   `test_basis_id_generates_distinct_basis_columns`.
    -   `test_cnf_siren_uses_shared_hard_solver`.
    -   `test_cnf_siren_gradient_reaches_hypernetwork`.
-   [ ] **Step 2: Run tests**
    -   Run: `python -m pytest tests/test_cnf_siren.py -q`
    -   Expected: FAIL.
-   [ ] **Step 3: Adapt official `hypernet_siren`**
    -   Verify exact factory key and constructor in official
        `network.py`.
    -   Use width 21, one hidden layer, first/hidden omega0=30.
    -   Preserve SIREN-specific initialization.
    -   Ensure K=33 basis IDs produce K basis functions through the
        official hypernetwork path.
    -   Keep `sepbasis_siren` as optional P1 ablation, not MVP basis.
-   [ ] **Step 4: Integrate with shared CNF solver**
    -   Reuse `build_hard_constraint_system` and `solve_coefficients`.
    -   Do not implement a second constraint solver.
-   [ ] **Step 5: Run tests**
    -   Run: `python -m pytest tests/test_cnf_siren.py -q`
    -   Expected: PASS.

------------------------------------------------------------------------

## Task 9: Fourier Collocation method track

**Files:** - Create: `models/bases/fourier_basis.py` - Create:
`models/fourier_collocation.py` - Test: `tests/test_fourier.py`

**Interfaces:** - `FourierBasis.forward(x) -> Phi[N,33]` -
`FourierBasis.derivatives(x) -> (Phi0, Phi1, Phi2)` -
`FourierCollocation.solve(sample, x_interior) -> beta[33,1]`

-   [ ] **Step 1: Write failing tests**
    -   `test_fourier_has_33_columns`.
    -   `test_fourier_frequency_set_is_fixed_public_library`.
    -   `test_fourier_does_not_use_sample_true_omega_as_basis`.
    -   `test_fourier_analytic_derivatives`.
    -   `test_fourier_collocation_direct_solve`.
    -   `test_fourier_has_no_optimizer_or_training_steps`.
-   [ ] **Step 2: Run tests**
    -   Run: `python -m pytest tests/test_fourier.py -q`
    -   Expected: FAIL.
-   [ ] **Step 3: Implement the fixed spectral basis**
    -   Main basis: constant + sin/cos pairs for nπ, n=1..16.
    -   Total columns K=33.
    -   Implement analytic first and second derivatives.
    -   Store selected frequency vector in config and output metadata.
-   [ ] **Step 4: Implement collocation solve**
    -   Use the same ODE/Dirichlet row ordering and RHS as CNF.
    -   Solve once; do not create optimizer or neural checkpoint.
    -   Record A, beta and frequency set.
-   [ ] **Step 5: Run tests**
    -   Run: `python -m pytest tests/test_fourier.py -q`
    -   Expected: PASS.

------------------------------------------------------------------------

## Task 10: SIREN-PINN method track

**Files:** - Create: `models/siren_pinn.py` - Create:
`training/train_pinn.py` - Test: `tests/test_siren_pinn.py`

**Interfaces:** - `SirenPINN.forward(x) -> q_hat[N,1]` -
`siren_pinn_loss(model, batch, physics_batch, weights) -> loss_dict` -
`train_siren_pinn(config, sample) -> trained_model, history`

-   [ ] **Step 1: Write failing tests**
    -   `test_siren_pinn_output_shape`.
    -   `test_siren_pinn_uses_siren_initialization`.
    -   `test_loss_returns_data_ode_bc_components`.
    -   `test_boundary_loss_is_soft_penalty`.
    -   `test_pinn_optimizer_updates_network_weights`.
    -   `test_pinn_uses_31_ode_collocation_points`.
-   [ ] **Step 2: Run tests**
    -   Run: `python -m pytest tests/test_siren_pinn.py -q`
    -   Expected: FAIL.
-   [ ] **Step 3: Implement SIREN network**
    -   Width 64, three hidden layers, omega0=30.
    -   Use dedicated SIREN initialization.
    -   Output q, not f; add b only for final f prediction/evaluation.
-   [ ] **Step 4: Implement soft-constraint training**
    -   Adam lr=1e-4.
    -   lambda_data=1, lambda_ODE=1, lambda_BC=100.
    -   10000 initial steps, 20000 maximum, patience 1000.
    -   Log each loss component independently.
-   [ ] **Step 5: Run tests**
    -   Run: `python -m pytest tests/test_siren_pinn.py -q`
    -   Expected: PASS.

------------------------------------------------------------------------

## Task 11: Implement the shared CNF training loop

**Files:** - Create: `training/train_cnf.py` - Create:
`training/early_stop.py` - Modify: `models/cnf.py` - Test:
`tests/test_training.py`

**Interfaces:** -
`train_cnf_sample(model, sample, config) -> model, history, best_step` -
`condition_regularizer(A) -> scalar` -
`EarlyStopping.step(metric) -> bool`

-   [ ] **Step 1: Write failing training-loop tests**
    -   `test_beta_recomputed_each_iteration`.
    -   `test_beta_not_in_optimizer_parameter_groups`.
    -   `test_backward_updates_basis_parameters`.
    -   `test_condition_regularizer_matches_log_cond`.
    -   `test_early_stopping_patience_500`.
-   [ ] **Step 2: Run tests**
    -   Run: `python -m pytest tests/test_training.py -q`
    -   Expected: FAIL.
-   [ ] **Step 3: Implement the differentiable CNF loop**
    -   Rebuild A/g and solve beta every step.
    -   Predict q at observation points.
    -   Compute data MSE + condition regularizer.
    -   Backpropagate through solve to basis parameters.
    -   Adam updates basis parameters only.
    -   Gaussian: lr 1e-3, 5000 steps; CNF-SIREN: lr 3e-4, 5000 initial
        / 10000 scan cap; patience 500.
-   [ ] **Step 4: Run tests**
    -   Run: `python -m pytest tests/test_training.py -q`
    -   Expected: PASS.

------------------------------------------------------------------------

## Task 12: Implement unified evaluation and artifact logging

**Files:** - Create: `evaluation/metrics.py` - Create:
`evaluation/evaluate.py` - Create: `evaluation/plots.py` - Create:
`scripts/aggregate.py` - Test: `tests/test_evaluation.py`

**Interfaces:** -
`evaluate_sample(method, sample, dense_grid) -> metrics, predictions` -
`write_run_artifacts(run_dir, config, metrics, predictions, diagnostics) -> None` -
`aggregate_runs(result_dirs) -> summary_table`

-   [ ] **Step 1: Write failing metric tests**
    -   `test_rmse_relative_l2_max_error`.
    -   `test_dense_ode_rms_and_max`.
    -   `test_collocation_residual_separate_from_dense_residual`.
    -   `test_boundary_errors`.
    -   `test_metrics_include_condition_and_solver_time`.
    -   `test_predictions_npz_contains_required_arrays`.
-   [ ] **Step 2: Run tests**
    -   Run: `python -m pytest tests/test_evaluation.py -q`
    -   Expected: FAIL.
-   [ ] **Step 3: Implement metrics and output schema**
    -   Save required metrics columns from SPEC.
    -   Save `summary.json`, `config.yaml`, `environment.txt`,
        `git_commit.txt`, `train.log`, predictions, beta, and diagnostic
        arrays.
    -   Produce the 15 specified figures.
-   [ ] **Step 4: Run tests**
    -   Run: `python -m pytest tests/test_evaluation.py -q`
    -   Expected: PASS.

------------------------------------------------------------------------

## Task 13: Integrate the four-method P0 runner

**Files:** - Create: `scripts/run_mvp.py` - Modify: `configs/mvp.yaml` -
Test: `tests/test_end_to_end.py`

**Interfaces:** - `run_mvp(config_path, methods="all") -> run_summary` -
Supported methods: `cnf_gaussian`, `cnf_siren`, `fourier`, `siren_pinn`,
`all`.

-   [ ] **Step 1: Write failing end-to-end tests**
    -   `test_all_four_methods_run_from_same_sample`.
    -   `test_all_methods_use_same_seed_and_observation_indices`.
    -   `test_all_methods_write_metrics_and_predictions`.
    -   `test_failed_method_does_not_corrupt_other_method_outputs`.
-   [ ] **Step 2: Run tests**
    -   Run: `python -m pytest tests/test_end_to_end.py -q`
    -   Expected: FAIL.
-   [ ] **Step 3: Implement the unified runner**
    -   Load one seed=42 sample with 128 sensors, 32 observations and
        zero noise.
    -   Execute all four methods with common ODE/BC and evaluation grid.
    -   Save each method to a separate run directory under a common
        experiment id.
    -   A failure in one method must be recorded and must not erase
        other method outputs.
-   [ ] **Step 4: Run the full test suite**
    -   Run: `python -m pytest -q`
    -   Expected: PASS.
-   [ ] **Step 5: Run P0**
    -   Run:
        `python scripts/run_mvp.py --config configs/mvp.yaml --methods all`
    -   Expected: four method run records, metrics, predictions, logs
        and figures. Numerical failure is reported explicitly; do not
        mark P0 successful if any method is silently skipped.

------------------------------------------------------------------------

## Task 14: Implement P1 structure/capacity sweeps

**Files:** - Create: `scripts/run_sweep.py` - Create:
`configs/sweeps/p1_capacity.yaml` - Create:
`configs/sweeps/p1_observations.yaml` - Create:
`configs/sweeps/p1_collocation.yaml` - Create:
`configs/sweeps/p1_basis.yaml` - Test: `tests/test_sweeps.py`

**Interfaces:** -
`run_sweep(suite: str, config_path: str) -> list[run_summary]`

-   [ ] **Step 1: Write failing sweep tests**
    -   `test_capacity_sweep_uses_K_17_33_65_and_M_K_minus_2`.
    -   `test_observation_sweep_uses_16_32_64`.
    -   `test_point_layout_sweep_uses_uniform_random_chebyshev`.
    -   `test_gaussian_sigma_sweep_values`.
    -   `test_siren_omega0_sweep_values`.
    -   `test_rectangular_collocation_branch_marks_underdetermined_and_overdetermined`.
-   [ ] **Step 2: Run tests**
    -   Run: `python -m pytest tests/test_sweeps.py -q`
    -   Expected: FAIL.
-   [ ] **Step 3: Implement square capacity and observation sweeps**
    -   Capacity: K=17/33/65, M=K-2.
    -   Observations: Nobs=16/32/64, K=33, M=31.
    -   Point layout: uniform/random/Chebyshev.
    -   Gaussian sigma: 0.01/0.025/0.05/0.1/0.2.
    -   SIREN omega0: 10/20/30/40.
-   [ ] **Step 4: Implement rectangular collocation extension**
    -   Preserve report's fixed K=33, M=15/31/63 sweep as a separately
        labeled extension.
    -   M=15: underdetermined minimum-norm equality solve; record rank
        and residual.
    -   M=63: overdetermined least-squares feasibility diagnostic; if
        residual exceeds hard-constraint tolerance, mark
        `hard_feasible=false`.
    -   Do not use square `torch.linalg.solve` on rectangular matrices.
    -   Do not merge rectangular results into strict square-CNF headline
        results.
-   [ ] **Step 5: Run tests**
    -   Run: `python -m pytest tests/test_sweeps.py -q`
    -   Expected: PASS.

------------------------------------------------------------------------

## Task 15: Implement P2 noise and OOD experiments

**Files:** - Create: `configs/sweeps/p2_noise.yaml` - Create:
`configs/sweeps/p2_ood_frequency.yaml` - Create:
`configs/sweeps/p2_ood_decay.yaml` - Modify: `scripts/run_sweep.py` -
Test: `tests/test_noise_ood.py`

-   [ ] **Step 1: Write failing tests**
    -   `test_noise_only_changes_observation_targets`.
    -   `test_clean_ode_and_boundary_rhs_unchanged_under_noise`.
    -   `test_frequency_ood_uses_report_ranges`.
    -   `test_decay_ood_requires_explicit_range`.
    -   `test_smoke_20_before_full_100`.
-   [ ] **Step 2: Run tests**
    -   Run: `python -m pytest tests/test_noise_ood.py -q`
    -   Expected: FAIL.
-   [ ] **Step 3: Implement noise sweeps**
    -   Use 1% and 5% relative noise.
    -   Keep clean ODE and clean boundary RHS.
    -   Use noisy observations only in regression loss.
    -   Run 20-function smoke test before full 100-function test.
-   [ ] **Step 4: Implement OOD sweeps**
    -   Frequency: use report's 5π--6π and 8π--9π ranges.
    -   Decay: accept explicit configured gamma OOD ranges; do not
        invent a default because the report does not provide one.
    -   Save true physical parameters used by the ODE operator.
-   [ ] **Step 5: Run tests**
    -   Run: `python -m pytest tests/test_noise_ood.py -q`
    -   Expected: PASS.

------------------------------------------------------------------------

## Task 16: Implement P3 shared-basis transfer

**Files:** - Create: `training/train_shared_basis.py` - Create:
`configs/sweeps/p3_transfer.yaml` - Test: `tests/test_transfer.py`

-   [ ] **Step 1: Write failing transfer tests**
    -   `test_shared_basis_trains_only_on_train_split`.
    -   `test_test_theta_is_frozen`.
    -   `test_test_sample_recomputes_A_and_beta`.
    -   `test_no_test_observations_leak_into_basis_training`.
-   [ ] **Step 2: Run tests**
    -   Run: `python -m pytest tests/test_transfer.py -q`
    -   Expected: FAIL.
-   [ ] **Step 3: Implement transfer protocol**
    -   Train shared Gaussian/SIREN basis inductive bias on 800 train
        functions.
    -   Freeze theta for validation/test.
    -   For each unseen test function, rebuild its own ODE/BC matrix and
        solve only beta.
    -   Save training split hash, frozen basis checkpoint, per-test A
        diagnostics and beta.
-   [ ] **Step 4: Run tests**
    -   Run: `python -m pytest tests/test_transfer.py -q`
    -   Expected: PASS.

------------------------------------------------------------------------

## Task 17: Aggregate results and produce the experiment report

**Files:** - Create: `evaluation/report.py` - Modify:
`scripts/aggregate.py` - Test: `tests/test_report_aggregation.py`

-   [ ] **Step 1: Write failing aggregation tests**
    -   `test_mean_std_by_method_and_seed`.
    -   `test_numerical_failures_are_counted_not_dropped`.
    -   `test_report_separates_hard_and_soft_methods`.
    -   `test_report_labels_oracle_parameters`.
    -   `test_report_does_not_claim_dense_ode_exactness`.
-   [ ] **Step 2: Run tests**
    -   Run: `python -m pytest tests/test_report_aggregation.py -q`
    -   Expected: FAIL.
-   [ ] **Step 3: Implement aggregation**
    -   Produce mean/std/median and failure counts.
    -   Produce method comparison tables and required figures.
    -   Include environment/config hashes and parameter-known protocol.
    -   Explain that CNF/Fourier hard equalities apply at finite
        constraints; dense ODE residual is empirical continuous-domain
        evaluation.
-   [ ] **Step 4: Run tests**
    -   Run: `python -m pytest tests/test_report_aggregation.py -q`
    -   Expected: PASS.
-   [ ] **Step 5: Run final verification**
    -   Run: `python -m pytest -q`
    -   Run:
        `python scripts/run_mvp.py --config configs/mvp.yaml --methods all`
    -   Run: `python scripts/run_sweep.py --suite p1`
    -   Expected: tests pass; all generated runs contain config, logs,
        metrics, predictions and diagnostics. P2/P3 may be launched
        after P1 review.

------------------------------------------------------------------------

## Completion Checklist

-   [ ] Official CNF source APIs inspected and documented.
-   [ ] Data generator and splits reproducible.
-   [ ] Analytic ODE and derivative tests pass.
-   [ ] Column-wise basis derivatives pass.
-   [ ] Hard matrix and direct solver pass.
-   [ ] CNF-Gaussian passes shape, solve, and gradient tests.
-   [ ] CNF-SIREN passes shape, solve, and gradient tests.
-   [ ] Fourier passes basis, derivative, and direct solve tests.
-   [ ] SIREN-PINN passes loss and optimizer tests.
-   [ ] Unified P0 runs all four methods.
-   [ ] P1 capacity/observation/layout/sigma/omega sweeps run.
-   [ ] Rectangular collocation is separately labeled and
    feasibility-checked.
-   [ ] P2 noise/OOD tests pass.
-   [ ] P3 shared-basis transfer tests pass.
-   [ ] Metrics, figures, configurations, logs, and numerical failures
    are retained.
-   [ ] Final report distinguishes source facts, experiment proposals,
    engineering thresholds, oracle parameters, hard constraints, and
    soft constraints.
