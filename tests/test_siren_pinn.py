import torch

from data.generator import generate_sample
from models.siren_pinn import SirenPINN, siren_pinn_loss
from physics.collocation import chebyshev_interior


def test_siren_pinn_output_and_soft_losses():
    sample = generate_sample(seed=42)
    model = SirenPINN(width=64, hidden_layers=3, omega0=30).double()
    output = model(sample['x_obs'])
    assert output.shape == (32, 1)
    losses = siren_pinn_loss(model, sample, chebyshev_interior(31),
                             dict(data=1.0, ode=1.0, bc=100.0))
    assert set(losses) == {'total', 'data', 'ode', 'bc'}
    assert all(torch.isfinite(item) for item in losses.values())
    losses['total'].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
