"""Independent SIREN per basis column for the P1 sepbasis_siren ablation."""

import torch
from torch import nn

from models.bases.base import column_derivatives
from models.siren_pinn import SirenPINN


class SeparateSirenBasis(nn.Module):
    def __init__(self, num_basis=33, width=21, hidden_layers=1, omega0=30):
        super().__init__()
        self.networks = nn.ModuleList(SirenPINN(width, hidden_layers, omega0)
                                      for _ in range(num_basis))

    def forward(self, x):
        return torch.cat([network(x) for network in self.networks], dim=1)

    def derivatives(self, x):
        return column_derivatives(self, x)
