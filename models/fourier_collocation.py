"""Direct Fourier collocation baseline without training."""

import torch
import time
from torch import nn

from models.bases.fourier_basis import FourierBasis
from physics.constraints import build_hard_constraint_system
from solvers.differentiable_solve import solve_coefficients
from solvers.rectangular import solve_rectangular


class FourierCollocation(nn.Module):
    def __init__(self, harmonics=16, frequency_step=None):
        super().__init__()
        self.basis = FourierBasis(harmonics, frequency_step) if frequency_step is not None else FourierBasis(harmonics)

    def solve(self, sample, x_interior, rectangular_extension=False, boundary_mode='dirichlet'):
        matrix, rhs = build_hard_constraint_system(
            self.basis, x_interior, sample['gamma'], sample['omega'],
            sample['q_left'], sample['q_right'], boundary_mode,
            sample['A'] * sample['omega'])
        start = time.perf_counter()
        if matrix.shape[0] != matrix.shape[1]:
            if not rectangular_extension:
                raise ValueError('rectangular system requires explicit extension flag')
            beta, rectangle = solve_rectangular(matrix, rhs)
        else:
            beta, rectangle = solve_coefficients(matrix, rhs), {}
        elapsed = time.perf_counter() - start
        residual = torch.linalg.vector_norm(matrix @ beta - rhs).item()
        row_residual = matrix @ beta - rhs
        return beta, {'solver_residual': residual, 'condition_number': torch.linalg.cond(matrix).item(),
                      'rank': int(torch.linalg.matrix_rank(matrix).item()),
                      'singular_values': torch.linalg.svdvals(matrix).detach().tolist(),
                      'hard_residual_max': float((matrix @ beta - rhs).abs().max()),
                      'collocation_ode_rms': float(row_residual[:-2].square().mean().sqrt()),
                      'solver_time_seconds': elapsed, **rectangle}

    def predict(self, x, beta, offset=0.0):
        return self.basis(x) @ beta + offset
