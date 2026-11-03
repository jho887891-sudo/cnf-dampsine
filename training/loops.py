"""Per-sample training for hard CNF and soft PINN methods."""

import math
import torch

from models.siren_pinn import siren_pinn_loss
from physics.constraints import build_hard_constraint_system
from solvers.differentiable_solve import solve_coefficients


def train_cnf_sample(model, sample, x_interior, *, steps=5000, lr=1e-3,
                     condition_weight=1e-6, patience=500, boundary_mode='dirichlet'):
    optimizer = torch.optim.Adam((p for p in model.basis.parameters() if p.requires_grad), lr=lr)
    best, stale, history, best_state = math.inf, 0, [], None
    for step in range(steps):
        optimizer.zero_grad()
        matrix, rhs = build_hard_constraint_system(model.basis, x_interior,
            sample['gamma'], sample['omega'], sample['q_left'], sample['q_right'],
            boundary_mode, sample['A'] * sample['omega'])
        beta = solve_coefficients(matrix, rhs)
        diagnostics = {'solver_residual': float(torch.linalg.vector_norm(matrix @ beta - rhs).detach()),
                       'condition_number': float(torch.linalg.cond(matrix).detach())}
        data = (model(sample['x_obs'], beta) - (sample['y_obs'] - sample['b'])).square().mean()
        loss = data + condition_weight * torch.log(torch.linalg.cond(matrix))
        loss.backward()
        optimizer.step()
        value = float(loss.detach())
        history.append({'step': step + 1, 'total': value, 'data': float(data.detach()), **diagnostics})
        if value < best:
            best, stale = value, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    return model, history


def train_siren_pinn(model, sample, x_interior, *, steps=10000, lr=1e-4,
                     patience=1000, weights=None):
    weights = weights or {'data': 1.0, 'ode': 1.0, 'bc': 100.0}
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best, stale, history, best_state = math.inf, 0, [], None
    for step in range(steps):
        optimizer.zero_grad()
        losses = siren_pinn_loss(model, sample, x_interior, weights)
        losses['total'].backward()
        optimizer.step()
        record = {'step': step + 1, **{k: float(v.detach()) for k, v in losses.items()}}
        history.append(record)
        if record['total'] < best:
            best, stale = record['total'], 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    return model, history
