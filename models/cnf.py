"""CNF hard-constrained linear combination of learned bases."""

import torch
import time
from torch import nn

from physics.constraints import build_hard_constraint_system
from solvers.differentiable_solve import solve_coefficients
from solvers.rectangular import solve_rectangular


class CNFModel(nn.Module):
    def __init__(self, basis):
        super().__init__()
        self.basis = basis

    def solve_beta(self, sample, x_interior, boundary_mode='dirichlet',
                   rectangular_extension=False):
        matrix, rhs = build_hard_constraint_system(
            self.basis, x_interior, sample['gamma'], sample['omega'],
            sample['q_left'], sample['q_right'], boundary_mode,
            sample['A'] * sample['omega'])
        start = time.perf_counter()
        if matrix.shape[0] != matrix.shape[1]:
            if not rectangular_extension:
                raise ValueError('rectangular CNF requires explicit extension flag')
            beta, rectangle = solve_rectangular(matrix, rhs)
        else:
            beta, rectangle = solve_coefficients(matrix, rhs), {}
        elapsed = time.perf_counter() - start
        residual = matrix @ beta - rhs
        diagnostics = {'solver_residual': torch.linalg.vector_norm(matrix @ beta - rhs).item(),
                      'condition_number': torch.linalg.cond(matrix).item(),
                      'rank': int(torch.linalg.matrix_rank(matrix).item()),
                      'singular_values': torch.linalg.svdvals(matrix).detach().tolist(),
                      'hard_residual_max': float((matrix @ beta - rhs).abs().max()),
                      'collocation_ode_rms': float(residual[:-2].square().mean().sqrt()),
                      'solver_time_seconds': elapsed}
        diagnostics.update(rectangle)
        return beta, diagnostics

    def forward(self, x, beta):
        return self.basis(x) @ beta
