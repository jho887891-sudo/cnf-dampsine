"""Assemble ODE rows followed by the two Dirichlet rows."""

import torch


def build_hard_constraint_system(basis, x_interior, gamma, omega, q_left, q_right,
                                 boundary_mode='dirichlet', initial_derivative=None):
    if x_interior.ndim != 2 or x_interior.shape[1] != 1:
        raise ValueError('x_interior must have shape [M,1]')
    if not torch.all((x_interior > 0) & (x_interior < 1)):
        raise ValueError('collocation points must be interior')
    phi, dphi, ddphi = basis.derivatives(x_interior)
    ode = ddphi + 2 * gamma * dphi + (gamma * gamma + omega * omega) * phi
    ends = torch.tensor([[0.0], [1.0]], dtype=x_interior.dtype, device=x_interior.device)
    phi_ends, dphi_ends, _ = basis.derivatives(ends)
    if boundary_mode == 'dirichlet':
        boundary = phi_ends
        second_rhs = q_right
    elif boundary_mode == 'initial_value':
        if initial_derivative is None:
            raise ValueError('initial derivative is required')
        boundary = torch.cat((phi_ends[:1], dphi_ends[:1]), dim=0)
        second_rhs = initial_derivative
    else:
        raise ValueError(f'unknown boundary mode: {boundary_mode}')
    matrix = torch.cat((ode, boundary), dim=0)
    rhs = torch.zeros((matrix.shape[0], 1), dtype=matrix.dtype, device=matrix.device)
    rhs[-2, 0] = q_left
    rhs[-1, 0] = second_rhs
    return matrix, rhs
