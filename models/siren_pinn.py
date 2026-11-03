"""SIREN direct representation with soft ODE and boundary penalties."""

import math
import torch
from torch import nn


class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, omega0=30, is_first=False):
        super().__init__()
        self.omega0 = omega0
        self.linear = nn.Linear(in_features, out_features)
        bound = (1 / in_features if is_first else math.sqrt(6 / in_features) / omega0)
        with torch.no_grad():
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega0 * self.linear(x))


class SirenPINN(nn.Module):
    def __init__(self, width=64, hidden_layers=3, omega0=30):
        super().__init__()
        layers = [SineLayer(1, width, omega0, is_first=True)]
        layers.extend(SineLayer(width, width, omega0) for _ in range(hidden_layers))
        final = nn.Linear(width, 1)
        with torch.no_grad():
            final.weight.uniform_(-math.sqrt(6 / width) / omega0, math.sqrt(6 / width) / omega0)
        self.net = nn.Sequential(*layers, final)

    def forward(self, x):
        return self.net(x)


def siren_pinn_loss(model, sample, x_interior, weights):
    x = x_interior.detach().clone().requires_grad_(True)
    q = model(x)
    dq = torch.autograd.grad(q.sum(), x, create_graph=True)[0]
    ddq = torch.autograd.grad(dq.sum(), x, create_graph=True)[0]
    gamma, omega = sample['gamma'], sample['omega']
    residual = ddq + 2*gamma*dq + (gamma*gamma + omega*omega)*q
    data = (model(sample['x_obs']) - (sample['y_obs'] - sample['b'])).square().mean()
    ode = residual.square().mean()
    ends = torch.tensor([[0.0], [1.0]], dtype=x.dtype, device=x.device)
    boundary = model(ends)
    if sample.get('boundary_mode') == 'initial_value':
        x0 = ends[:1].detach().clone().requires_grad_(True)
        initial_dq = torch.autograd.grad(model(x0).sum(), x0, create_graph=True)[0]
        second = (initial_dq - sample['A']*sample['omega']).square().sum()
    else:
        second = (boundary[1] - sample['q_right']).square().sum()
    bc = (boundary[0] - sample['q_left']).square().sum() + second
    return {'data': data, 'ode': ode, 'bc': bc,
            'total': weights['data']*data + weights['ode']*ode + weights['bc']*bc}
