"""Reproducible FunDiff-style damped sine samples with retained parameters."""

import math
import torch


def generate_sample(seed=42, n_sensors=128, n_obs=32, noise=0.0,
                    amplitude_range=(0.5, 1.0), gamma_range=(2.0, 4.0),
                    omega_range=(6 * math.pi, 8 * math.pi), offset_range=(-0.5, 0.5)):
    if not 1 <= n_obs <= n_sensors:
        raise ValueError('n_obs must be between 1 and n_sensors')
    if noise < 0:
        raise ValueError('noise must be nonnegative')
    rng = torch.Generator().manual_seed(seed)
    uniform = lambda lo, hi: lo + (hi - lo) * torch.rand((), generator=rng, dtype=torch.float64).item()
    amplitude = uniform(*amplitude_range)
    gamma = uniform(*gamma_range)
    omega = uniform(*omega_range)
    offset = uniform(*offset_range)
    x = torch.linspace(0, 1, n_sensors, dtype=torch.float64)[:, None]
    clean = amplitude * torch.exp(-gamma * x) * torch.sin(omega * x) + offset
    indices = torch.randperm(n_sensors, generator=rng)[:n_obs].sort().values
    observed = clean[indices].clone()
    if noise:
        observed += noise * clean.std(unbiased=False) * torch.randn(observed.shape, generator=rng, dtype=observed.dtype)
    return dict(x_sensor=x, y_clean=clean, x_obs=x[indices], y_obs=observed,
                obs_indices=indices, A=amplitude, gamma=gamma, omega=omega,
                b=offset, q_left=torch.zeros(1, dtype=torch.float64),
                q_right=clean[-1] - offset, seed=seed, noise=noise)
