"""Dense and constraint-point diagnostics for all four methods."""

import math
import torch


def evaluate_prediction(model, sample, grid, prediction, beta=None, chunk_size=128):
    truth = sample['A'] * torch.exp(-sample['gamma']*grid) * torch.sin(sample['omega']*grid) + sample['b']
    error = prediction - truth
    metrics = {
        'rmse': float(error.square().mean().sqrt()),
        'relative_l2': float(torch.linalg.vector_norm(error) / torch.linalg.vector_norm(truth)),
        'max_abs_error': float(error.abs().max()),
    }
    residuals = []
    for part in grid.split(chunk_size):
        x = part.detach().clone().requires_grad_(True)
        if beta is not None:
            q = model.basis(x) @ beta.detach()
        else:
            q = model(x)
        dq = torch.autograd.grad(q.sum(), x, create_graph=True)[0]
        ddq = torch.autograd.grad(dq.sum(), x, create_graph=False)[0]
        gamma, omega = sample['gamma'], sample['omega']
        residuals.append((ddq + 2*gamma*dq + (gamma*gamma + omega*omega)*q).detach())
    residual = torch.cat(residuals)
    metrics.update(dense_ode_rms=float(residual.square().mean().sqrt()),
                   dense_ode_max=float(residual.abs().max()))
    with torch.no_grad():
        ends = torch.tensor([[0.0], [1.0]], dtype=grid.dtype, device=grid.device)
        boundary = model.predict(ends, beta, 0.0) if hasattr(model, 'predict') else (model(ends, beta) if beta is not None else model(ends))
        metrics['left_bc_abs'] = float((boundary[0] - sample['q_left']).abs())
        metrics['right_bc_abs'] = float((boundary[1] - sample['q_right']).abs())
    return metrics, residual
