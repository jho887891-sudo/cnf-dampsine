"""Interior collocation grids."""

import math
import torch


def chebyshev_interior(count, *, dtype=torch.float64):
    if count < 1:
        raise ValueError('count must be positive')
    m = torch.arange(1, count + 1, dtype=dtype)
    return (0.5 * (1 - torch.cos(m * math.pi / (count + 1))))[:, None]
