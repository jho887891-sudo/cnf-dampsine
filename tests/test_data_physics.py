import math

import torch

from data.generator import generate_sample
from physics.damped_oscillator import damped_sine, ode_residual
from physics.collocation import chebyshev_interior


def test_mvp_sample_is_reproducible_and_keeps_clean_physics():
    a = generate_sample(seed=42, n_sensors=128, n_obs=32, noise=0.05)
    b = generate_sample(seed=42, n_sensors=128, n_obs=32, noise=0.05)
    assert a['x_sensor'].shape == (128, 1)
    assert a['x_obs'].shape == (32, 1)
    assert torch.equal(a['obs_indices'], b['obs_indices'])
    assert torch.equal(a['y_obs'], b['y_obs'])
    assert not torch.equal(a['y_obs'], a['y_clean'][a['obs_indices']])
    assert torch.allclose(a['q_right'], a['y_clean'][-1] - a['b'])
    assert 0.5 <= a['A'] <= 1.0
    assert 2 <= a['gamma'] <= 4
    assert 6 * math.pi <= a['omega'] <= 8 * math.pi


def test_analytic_ode_residual_is_zero():
    x = torch.linspace(0, 1, 41, dtype=torch.float64)[:, None]
    residual = ode_residual(x, 0.8, 3.0, 7 * math.pi)
    assert residual.abs().max() < 1e-10
    assert torch.allclose(damped_sine(x, 0.8, 3.0, 7 * math.pi, 0.2)[0], torch.tensor([0.2], dtype=torch.float64))


def test_chebyshev_points_are_strictly_interior():
    points = chebyshev_interior(31)
    assert points.shape == (31, 1)
    assert torch.all(points > 0) and torch.all(points < 1)
    assert torch.all(points[1:] > points[:-1])
