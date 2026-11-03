"""Strict square hard-constraint solve with optional exact equilibration."""

import torch


def equilibration_factors(matrix):
    """Row then column 2-norm scalars. Mathematically a diagonal preconditioner only."""
    if matrix.ndim != 2:
        raise ValueError('matrix must be 2-D')
    tiny = torch.finfo(matrix.dtype).tiny
    row = torch.linalg.vector_norm(matrix, dim=1, keepdim=True).clamp_min(tiny)
    col = torch.linalg.vector_norm(matrix / row, dim=0, keepdim=True).clamp_min(tiny)
    return row, col


def equilibrated_matrix(matrix):
    """Scaled copy of the same constraint system, used for conditioning diagnostics."""
    row, col = equilibration_factors(matrix)
    return (matrix / row) / col


def solve_coefficients(matrix, rhs, equilibrate=False):
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError('direct hard solve requires a square matrix')
    if rhs.shape != (matrix.shape[0], 1):
        raise ValueError('rhs must have shape [K,1]')
    if not equilibrate:
        return torch.linalg.solve(matrix, rhs)
    # A = diag(row) * (A / row);  A / row = equilibrated * diag(col).
    # Solving the diagonally scaled system and rescaling beta is an exact
    # reparameterisation: no ridge, no truncation, no change of solution.
    row, col = equilibration_factors(matrix)
    gamma = torch.linalg.solve((matrix / row) / col, rhs / row)
    return gamma / col.transpose(0, 1)
