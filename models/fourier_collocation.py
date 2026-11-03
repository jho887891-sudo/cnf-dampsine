"""Direct Fourier collocation baseline without training."""

import time

import torch
from torch import nn

from models.bases.fourier_basis import FourierBasis
from physics.constraints import build_hard_constraint_system
from solvers.differentiable_solve import equilibrated_matrix, solve_coefficients
from solvers.rectangular import solve_rectangular


class FourierCollocation(nn.Module):
    def __init__(self, harmonics=16, frequency_step=None, equilibrate=False):
        super().__init__()
        self.basis = FourierBasis(harmonics, frequency_step) if frequency_step is not None else FourierBasis(harmonics)
        self.equilibrate = equilibrate

    def solve(self, sample, x_interior, rectangular_extension=False, boundary_mode='dirichlet',
              equilibrate=None):
        matrix, rhs = build_hard_constraint_system(
            self.basis, x_interior, sample['gamma'], sample['omega'],
            sample['q_left'], sample['q_right'], boundary_mode,
            sample['A'] * sample['omega'])
        if equilibrate is None:
            equilibrate = self.equilibrate
        start = time.perf_counter()
        if matrix.shape[0] != matrix.shape[1]:
            if not rectangular_extension:
                raise ValueError('rectangular system requires explicit extension flag')
            beta, rectangle = solve_rectangular(matrix, rhs)
        else:
            beta, rectangle = solve_coefficients(matrix, rhs, equilibrate=equilibrate), {}
        elapsed = time.perf_counter() - start
        residual = matrix @ beta - rhs
        condition_raw = torch.linalg.cond(matrix).item()
        condition_equilibrated = torch.linalg.cond(equilibrated_matrix(matrix)).item()
        eps = torch.finfo(matrix.dtype).eps
        return beta, {'solver_residual': torch.linalg.vector_norm(residual).item(),
                      'condition_number': condition_raw,
                      'condition_number_equilibrated': condition_equilibrated,
                      'condition_error_bound': condition_raw * eps,
                      'condition_error_bound_equilibrated': condition_equilibrated * eps,
                      'equilibrated_solve': bool(equilibrate),
                      'rank': int(torch.linalg.matrix_rank(matrix).item()),
                      'singular_values': torch.linalg.svdvals(matrix).detach().tolist(),
                      'hard_residual_max': float(residual.abs().max()),
                      'collocation_ode_rms': float(residual[:-2].square().mean().sqrt()),
                      'solver_time_seconds': elapsed, **rectangle}

    def predict(self, x, beta, offset=0.0):
        return self.basis(x) @ beta + offset
