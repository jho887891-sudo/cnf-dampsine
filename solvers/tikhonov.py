"""Explicit regularized diagnostic solve; never used by hard CNF defaults."""

import torch


def solve_tikhonov(matrix, rhs, ridge_lambda):
    if ridge_lambda <= 0:
        raise ValueError('Tikhonov lambda must be positive')
    if matrix.ndim != 2 or rhs.shape != (matrix.shape[0], 1):
        raise ValueError('invalid matrix or rhs shape')
    gram = matrix.T @ matrix
    identity = torch.eye(matrix.shape[1], dtype=matrix.dtype, device=matrix.device)
    return torch.linalg.solve(gram + ridge_lambda * identity, matrix.T @ rhs)
