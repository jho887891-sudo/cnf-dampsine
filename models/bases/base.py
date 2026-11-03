"""Per-column basis derivatives, preserving the parameter graph."""

import torch


def column_derivatives(basis, x):
    if hasattr(torch, 'func') and hasattr(torch.func, 'jvp'):
        tangent = torch.ones_like(x)
        phi, first = torch.func.jvp(basis, (x,), (tangent,))
        _, second = torch.func.jvp(
            lambda z: torch.func.jvp(basis, (z,), (torch.ones_like(z),))[1],
            (x,), (tangent,))
        return phi, first, second
    x = x.detach().clone().requires_grad_(True)
    phi = basis(x)
    first, second = [], []
    for k in range(phi.shape[1]):
        d1 = torch.autograd.grad(phi[:, k].sum(), x, create_graph=True, retain_graph=True)[0]
        d2 = torch.autograd.grad(d1.sum(), x, create_graph=True, retain_graph=True)[0]
        first.append(d1)
        second.append(d2)
    return phi, torch.cat(first, dim=1), torch.cat(second, dim=1)
