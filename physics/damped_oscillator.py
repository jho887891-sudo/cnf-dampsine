"""Analytic damped sine and its homogeneous ODE."""

import torch


def damped_sine(x, A, gamma, omega, b=0.0):
    return A * torch.exp(-gamma * x) * torch.sin(omega * x) + b


def ode_residual(x, A, gamma, omega):
    envelope = A * torch.exp(-gamma * x)
    sine, cosine = torch.sin(omega * x), torch.cos(omega * x)
    q = envelope * sine
    dq = envelope * (-gamma * sine + omega * cosine)
    ddq = envelope * ((gamma * gamma - omega * omega) * sine - 2 * gamma * omega * cosine)
    return ddq + 2 * gamma * dq + (gamma * gamma + omega * omega) * q
