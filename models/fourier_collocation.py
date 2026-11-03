"""Direct Fourier collocation baseline without training."""

import math
import time

import torch
from torch import nn

from models.bases.fourier_basis import FourierBasis
from physics.constraints import build_hard_constraint_system
from solvers.differentiable_solve import equilibrated_matrix, solve_coefficients
from solvers.diagnostics import linear_system_diagnostics
from solvers.rectangular import solve_rectangular

# Row-only scaling used for the SPEC diagnostics experiment, never for the main solve.
from solvers.differentiable_solve import equilibration_factors


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
        diagnostics = linear_system_diagnostics(matrix, rhs, beta)
        scale_info = self._conditioning_variants(matrix)
        finite_residual = diagnostics['residual_finite']
        diagnostics.update(
            solver_time_seconds=elapsed,
            equilibrated_solve=bool(equilibrate),
            hard_residual_max=float(residual.abs().max()) if finite_residual else float('nan'),
            collocation_ode_rms=float(residual[:-2].square().mean().sqrt()) if finite_residual else float('nan'),
            **scale_info, **rectangle)
        return beta, diagnostics

    @staticmethod
    def _conditioning_variants(matrix):
        """Row-only and row+column scaled conditions, diagnostics only."""
        if not bool(torch.isfinite(matrix).all()):
            return {'condition_number_row_normalised': float('nan'),
                    'condition_number_equilibrated': float('nan')}
        row, _ = equilibration_factors(matrix)
        return {'condition_number_row_normalised': float(torch.linalg.cond(matrix / row).item()),
                'condition_number_equilibrated': float(torch.linalg.cond(equilibrated_matrix(matrix)).item())}

    def predict(self, x, beta, offset=0.0):
        return self.basis(x) @ beta + offset
