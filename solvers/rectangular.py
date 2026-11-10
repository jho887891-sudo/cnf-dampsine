"""Explicit rectangular constraint diagnostics, separate from strict square CNF."""

import torch


def rectangular_coefficients(matrix, rhs):
    rows, cols = matrix.shape
    if rows == cols:
        raise ValueError('square constraints belong to the direct solver')
    if rhs.shape != (rows, 1):
        raise ValueError('rhs must have shape [M,1]')
    if rows < cols:
        beta = torch.linalg.pinv(matrix) @ rhs
        branch = 'minimum_norm_underdetermined'
    else:
        beta = torch.linalg.lstsq(matrix, rhs).solution
        branch = 'least_squares_overdetermined'
    return beta, branch


def solve_rectangular(matrix, rhs, hard_tolerance=1e-8):
    beta, branch = rectangular_coefficients(matrix, rhs)
    residual = matrix @ beta - rhs
    error = float(residual.abs().max())
    return beta, {'solver': branch, 'hard_residual_max': error,
                  'hard_feasible': error <= hard_tolerance,
                  'matrix_shape': list(matrix.shape),
                  'rank': int(torch.linalg.matrix_rank(matrix).item()),
                  'condition_number': float(torch.linalg.cond(matrix))}
