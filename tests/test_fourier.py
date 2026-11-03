import math

import torch

from data.generator import generate_sample
from models.bases.fourier_basis import FourierBasis
from models.fourier_collocation import FourierCollocation
from physics.collocation import chebyshev_interior


def test_fourier_basis_and_derivatives():
    basis = FourierBasis()
    x = torch.tensor([[0.0], [0.25], [1.0]], dtype=torch.float64)
    phi, d1, d2 = basis.derivatives(x)
    assert phi.shape == d1.shape == d2.shape == (3, 33)
    assert torch.allclose(phi[:, 0], torch.ones(3, dtype=x.dtype))
    assert torch.allclose(d1[:, 1], math.pi * torch.cos(math.pi * x[:, 0]))
    assert torch.allclose(d2[:, 2], -(math.pi**2) * torch.cos(math.pi * x[:, 0]))
    assert torch.allclose(basis(x), phi)
    assert basis.frequencies == tuple(range(1, 17))


def test_fourier_collocation_solves_same_physics_without_optimizer():
    sample = generate_sample(seed=42)
    model = FourierCollocation()
    beta, diagnostics = model.solve(sample, chebyshev_interior(31))
    assert beta.shape == (33, 1)
    assert diagnostics['solver_residual'] < 1e-7
    assert not list(model.parameters())
    prediction = model.predict(sample['x_sensor'], beta, sample['b'])
    assert prediction.shape == sample['y_clean'].shape
