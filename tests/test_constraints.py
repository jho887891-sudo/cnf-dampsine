import math

import torch

from physics.collocation import chebyshev_interior
from physics.constraints import build_hard_constraint_system
from solvers.differentiable_solve import solve_coefficients


class PolynomialBasis:
    def derivatives(self, x):
        powers = torch.arange(3, dtype=x.dtype)
        phi = x ** powers
        d1 = torch.stack((torch.zeros_like(x[:, 0]), torch.ones_like(x[:, 0]), 2*x[:, 0]), dim=1)
        d2 = torch.stack((torch.zeros_like(x[:, 0]), torch.zeros_like(x[:, 0]), 2*torch.ones_like(x[:, 0])), dim=1)
        return phi, d1, d2


def test_constraint_rows_and_differentiable_solve():
    basis = PolynomialBasis()
    x = chebyshev_interior(1)
    matrix, rhs = build_hard_constraint_system(basis, x, gamma=3.0, omega=7*math.pi,
                                                q_left=0.0, q_right=0.1)
    assert matrix.shape == (3, 3)
    assert rhs.shape == (3, 1)
    assert torch.allclose(matrix[-2], torch.tensor([1., 0., 0.], dtype=torch.float64))
    assert torch.allclose(matrix[-1], torch.tensor([1., 1., 1.], dtype=torch.float64))
    coefficients = solve_coefficients(matrix, rhs)
    assert torch.allclose(matrix @ coefficients, rhs, atol=1e-10)


def test_square_solver_rejects_rectangular_system():
    try:
        solve_coefficients(torch.ones(2, 3), torch.ones(2, 1))
    except ValueError:
        pass
    else:
        raise AssertionError('rectangular system must be explicit')


def test_initial_value_fallback_uses_positive_derivative():
    matrix, rhs = build_hard_constraint_system(
        PolynomialBasis(), chebyshev_interior(1), gamma=3.0, omega=7*math.pi,
        q_left=0.0, q_right=0.0, boundary_mode='initial_value',
        initial_derivative=0.8*7*math.pi)
    assert torch.allclose(matrix[-1], torch.tensor([0., 1., 0.], dtype=torch.float64))
    assert rhs[-1, 0] > 0
