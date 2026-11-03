import torch

from data.generator import generate_sample
from models.bases.gaussian_kernel import GaussianKernelBasis
from models.cnf import CNFModel
from physics.collocation import chebyshev_interior


def test_gaussian_columns_have_separate_derivatives():
    anchors = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.float64)
    basis = GaussianKernelBasis(anchors, sigma=0.2, n_freqs=3)
    x = torch.tensor([[0.2], [0.7]], dtype=torch.float64)
    phi, d1, d2 = basis.derivatives(x)
    assert phi.shape == d1.shape == d2.shape == (2, 3)
    assert not torch.allclose(d1[:, 0], d1[:, 1])
    assert not basis.sigma.requires_grad
    assert any(p.requires_grad for p in basis.encoder.parameters())


def test_cnf_gaussian_gradient_reaches_encoder_through_solve():
    sample = generate_sample(seed=42)
    x = chebyshev_interior(31)
    anchors = torch.cat((x, torch.tensor([[0.0], [1.0]], dtype=x.dtype)))
    model = CNFModel(GaussianKernelBasis(anchors, sigma=0.05, n_freqs=3))
    beta, diagnostics = model.solve_beta(sample, x)
    assert beta.shape == (33, 1)
    assert diagnostics['solver_residual'] < 1e-5
    loss = (model(sample['x_obs'], beta) - (sample['y_obs'] - sample['b'])).square().mean()
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
               for p in model.basis.encoder.parameters())
