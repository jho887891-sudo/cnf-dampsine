import math

import torch

from data.generator import generate_sample
from models.bases.fourier_basis import FourierBasis
from models.fourier_collocation import FourierCollocation
from physics.collocation import chebyshev_interior
from solvers.diagnostics import condition_status, linear_system_diagnostics


def test_fourier_basis_and_derivatives():
    basis = FourierBasis()
    x = torch.tensor([[0.0], [0.25], [1.0]], dtype=torch.float64)
    phi, d1, d2 = basis.derivatives(x)
    assert phi.shape == d1.shape == d2.shape == (3, 33)
    assert torch.allclose(phi[:, 0], torch.ones(3, dtype=x.dtype))
    assert torch.allclose(d1[:, 1], math.pi * torch.cos(math.pi * x[:, 0]))
    assert torch.allclose(d2[:, 2], -(math.pi**2) * torch.cos(math.pi * x[:, 0]))
    assert torch.allclose(basis(x), phi)
    assert basis.frequencies == tuple(range(1, 17))


def test_fourier_collocation_solves_same_physics_without_optimizer():
    sample = generate_sample(seed=42)
    model = FourierCollocation()
    beta, diagnostics = model.solve(sample, chebyshev_interior(31))
    assert beta.shape == (33, 1)
    assert diagnostics['solver_residual'] < 1e-7
    assert not list(model.parameters())
    prediction = model.predict(sample['x_sensor'], beta, sample['b'])
    assert prediction.shape == sample['y_clean'].shape


def test_condition_status_bands_follow_spec():
    assert condition_status(1e3) == 'normal'
    assert condition_status(1e5) == 'normal'
    assert condition_status(1e8) == 'warning'
    assert condition_status(1e10) == 'warning'
    assert condition_status(1e11) == 'unstable'
    assert condition_status(1e12) == 'unstable'
    assert condition_status(1e13) == 'severe'
    assert condition_status(float('nan')) == 'severe'
    assert condition_status(float('inf')) == 'severe'
    assert condition_status(None) == 'unknown'


def test_diagnostics_separate_solve_success_from_conditioning():
    matrix = torch.diag(torch.tensor([1.0, 1e-13], dtype=torch.float64))
    beta = torch.ones(2, 1, dtype=torch.float64)
    rhs = matrix @ beta
    diagnostics = linear_system_diagnostics(matrix, rhs, beta)
    assert diagnostics['solver_ok'] is True
    assert diagnostics['full_rank'] is True
    assert diagnostics['solver_residual'] == 0.0
    assert diagnostics['backward_error'] < 1e-12
    assert diagnostics['condition_status'] == 'severe'
    assert diagnostics['condition_number'] > 1e12


def test_backward_error_matches_definition():
    torch.manual_seed(0)
    matrix = torch.randn(6, 6, dtype=torch.float64)
    beta_true = torch.randn(6, 1, dtype=torch.float64)
    rhs = matrix @ beta_true
    beta = torch.linalg.solve(matrix, rhs)
    diagnostics = linear_system_diagnostics(matrix, rhs, beta)
    residual = float(torch.linalg.vector_norm(matrix @ beta - rhs))
    denominator = (float(torch.linalg.norm(matrix, 2)) * float(torch.linalg.vector_norm(beta))
                   + float(torch.linalg.vector_norm(rhs)) + torch.finfo(matrix.dtype).eps)
    assert math.isclose(diagnostics['backward_error'], residual / denominator, rel_tol=1e-12)
    assert diagnostics['relative_residual'] == residual / float(torch.linalg.vector_norm(rhs))
    assert diagnostics['backward_error'] <= diagnostics['relative_residual'] + 1e-16


def test_non_finite_beta_is_reported_as_solve_failure():
    matrix = torch.eye(3, dtype=torch.float64)
    beta = torch.tensor([[1.0], [float('nan')], [2.0]], dtype=torch.float64)
    diagnostics = linear_system_diagnostics(matrix, matrix @ torch.tensor([[1.0], [0.0], [2.0]], dtype=torch.float64), beta)
    assert diagnostics['beta_finite'] is False
    assert diagnostics['solver_ok'] is False


def test_fourier_diagnostics_expose_full_field_set():
    sample = generate_sample(seed=42)
    model = FourierCollocation()
    beta, diagnostics = model.solve(sample, chebyshev_interior(31))
    for key in ('matrix_shape', 'matrix_finite', 'rhs_finite', 'beta_finite', 'solver_ok', 'rank', 'full_rank',
                'max_singular_value', 'min_singular_value', 'singular_values', 'condition_number',
                'condition_status', 'solver_residual', 'relative_residual', 'backward_error', 'beta_norm',
                'hard_residual_max', 'collocation_ode_rms', 'solver_time_seconds',
                'condition_number_row_normalised', 'condition_number_equilibrated'):
        assert key in diagnostics
    assert diagnostics['matrix_shape'] == [33, 33]
    assert diagnostics['rank'] == 33
    assert diagnostics['full_rank'] is True
    assert diagnostics['beta_finite'] is True
    assert diagnostics['solver_ok'] is True
    assert len(diagnostics['singular_values']) == 33
    assert diagnostics['max_singular_value'] >= diagnostics['min_singular_value'] > 0
    assert diagnostics['condition_number'] == diagnostics['max_singular_value'] / diagnostics['min_singular_value']
    assert diagnostics['condition_status'] == condition_status(diagnostics['condition_number'])
    assert diagnostics['min_singular_value'] < 1e-5


def test_order_sweep_stays_square_and_full_rank():
    sample = generate_sample(seed=42)
    for harmonics, interior in ((8, 15), (10, 19), (12, 23), (16, 31)):
        model = FourierCollocation(harmonics=harmonics)
        beta, diagnostics = model.solve(sample, chebyshev_interior(interior))
        assert diagnostics['matrix_shape'] == [interior + 2, 1 + 2 * harmonics]
        assert diagnostics['full_rank'] is True
        assert diagnostics['solver_ok'] is True
        assert beta.shape == (1 + 2 * harmonics, 1)


def test_rectangular_fourier_reports_both_solver_branches():
    sample = generate_sample(seed=42)
    model = FourierCollocation()
    for interior, solver in ((15, 'minimum_norm_underdetermined'),
                             (63, 'least_squares_overdetermined')):
        beta, diagnostics = model.solve(
            sample, chebyshev_interior(interior), rectangular_extension=True)
        assert beta.shape == (33, 1)
        assert diagnostics['matrix_shape'] == [interior + 2, 33]
        assert diagnostics['solver'] == solver
        assert isinstance(diagnostics['hard_feasible'], bool)
        assert math.isfinite(diagnostics['hard_residual_max'])
