"""Numerical diagnostics that keep solve validity separate from conditioning."""

import math

import torch


def condition_status(condition_number):
    """SPEC 4.5 bands: normal / warning / unstable / severe."""
    if condition_number is None:
        return 'unknown'
    value = float(condition_number)
    if not math.isfinite(value):
        return 'severe'
    if value < 1e6:
        return 'normal'
    if value <= 1e10:
        return 'warning'
    if value <= 1e12:
        return 'unstable'
    return 'severe'


def linear_system_diagnostics(matrix, rhs, beta):
    """Report solve validity, residuals and conditioning as independent facts."""
    eps = torch.finfo(matrix.dtype).eps
    matrix_finite = bool(torch.isfinite(matrix).all())
    rhs_finite = bool(torch.isfinite(rhs).all())
    beta_finite = bool(torch.isfinite(beta).all())
    residual = matrix @ beta - rhs
    residual_finite = bool(torch.isfinite(residual).all())
    absolute = float(torch.linalg.vector_norm(residual)) if residual_finite else float('nan')
    rhs_norm = float(torch.linalg.vector_norm(rhs)) if rhs_finite else float('nan')
    beta_norm = float(torch.linalg.vector_norm(beta)) if beta_finite else float('nan')
    relative = absolute / rhs_norm if rhs_norm and math.isfinite(rhs_norm) and rhs_norm > 0 else float('nan')
    if matrix_finite:
        singular = torch.linalg.svdvals(matrix)
        maximum = float(singular.max())
        minimum = float(singular.min())
        rank = int(torch.linalg.matrix_rank(matrix).item())
        condition = maximum / minimum if minimum > 0 else float('inf')
        spectral = [float(value) for value in singular]
    else:
        maximum = minimum = condition = float('nan')
        rank = 0
        spectral = []
    denominator = maximum * beta_norm + rhs_norm + eps
    backward = absolute / denominator if math.isfinite(denominator) and denominator > 0 else float('nan')
    return {'matrix_shape': list(matrix.shape),
            'matrix_finite': matrix_finite,
            'rhs_finite': rhs_finite,
            'beta_finite': beta_finite,
            'residual_finite': residual_finite,
            'solver_ok': bool(matrix_finite and rhs_finite and beta_finite and residual_finite),
            'rank': rank,
            'full_rank': bool(rank == min(matrix.shape)),
            'max_singular_value': maximum,
            'min_singular_value': minimum,
            'singular_values': spectral,
            'condition_number': condition,
            'condition_status': condition_status(condition),
            'solver_residual': absolute,
            'relative_residual': relative,
            'backward_error': backward,
            'beta_norm': beta_norm}
