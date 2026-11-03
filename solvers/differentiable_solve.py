"""Strict square hard-constraint solve."""

import torch


def solve_coefficients(matrix, rhs):
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError('direct hard solve requires a square matrix')
    if rhs.shape != (matrix.shape[0], 1):
        raise ValueError('rhs must have shape [K,1]')
    return torch.linalg.solve(matrix, rhs)
