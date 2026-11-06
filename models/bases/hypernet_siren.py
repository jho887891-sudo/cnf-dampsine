"""Basis-ID-conditioned SIREN weights, following the official hypernet_siren route."""

import math
import torch
from torch import nn
from torch.nn import functional as F

from models.bases.base import column_derivatives


class HypernetSirenBasis(nn.Module):
    def __init__(self, num_basis=33, width=21, hidden_layers=1, omega0=30):
        super().__init__()
        self.num_basis = num_basis
        self.width = width
        self.hidden_layers = hidden_layers
        self.omega0 = omega0
        shapes = [(width, 1), (width,)]
        for _ in range(hidden_layers):
            shapes.extend(((width, width), (width,)))
        shapes.extend(((1, width), (1,)))
        self.shapes = shapes
        count = sum(math.prod(shape) for shape in shapes)
        self.hypernetwork = nn.Sequential(nn.Linear(num_basis, 100), nn.ReLU(), nn.Linear(100, count))
        self.register_buffer('basis_ids', torch.eye(num_basis))

    def _one_basis(self, x, flat):
        params = []
        cursor = 0
        for shape in self.shapes:
            size = math.prod(shape)
            params.append(flat[cursor:cursor + size].reshape(shape))
            cursor += size
        state = x
        for layer in range(self.hidden_layers + 1):
            state = torch.sin(self.omega0 * F.linear(state, params[2*layer], params[2*layer + 1]))
        return F.linear(state, params[-2], params[-1])

    def forward(self, x):
        generated = self.hypernetwork(self.basis_ids.to(dtype=x.dtype))
        params = []
        cursor = 0
        for shape in self.shapes:
            size = math.prod(shape)
            params.append(generated[:, cursor:cursor + size].reshape((self.num_basis,) + shape))
            cursor += size
        state = x.unsqueeze(0).expand(self.num_basis, -1, -1)
        for layer in range(self.hidden_layers + 1):
            weight, bias = params[2 * layer:2 * layer + 2]
            state = torch.sin(self.omega0 * (torch.bmm(state, weight.transpose(1, 2)) + bias[:, None, :]))
        weight, bias = params[-2:]
        return (torch.bmm(state, weight.transpose(1, 2)) + bias[:, None, :]).squeeze(-1).transpose(0, 1)

    def derivatives(self, x):
        return column_derivatives(self, x)
