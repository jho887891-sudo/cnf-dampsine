import torch

from models.bases.hypernet_siren import HypernetSirenBasis
from models.bases.base import column_derivatives


def test_hypernet_siren_produces_distinct_columns_and_derivatives():
    basis = HypernetSirenBasis(num_basis=33, width=21, hidden_layers=1, omega0=30).double()
    x = torch.tensor([[0.1], [0.4]], dtype=torch.float64)
    phi, d1, d2 = basis.derivatives(x)
    assert phi.shape == d1.shape == d2.shape == (2, 33)
    assert torch.isfinite(phi).all() and torch.isfinite(d2).all()
    assert not torch.allclose(phi[:, 0], phi[:, 1])
    assert any(p.requires_grad for p in basis.hypernetwork.parameters())


def test_hypernet_batched_forward_matches_individual_basis_and_gradient():
    basis = HypernetSirenBasis(num_basis=5, width=7, hidden_layers=2, omega0=20).double()
    x = torch.tensor([[0.1], [0.3], [0.7]], dtype=torch.float64)
    generated = basis.hypernetwork(basis.basis_ids.to(dtype=x.dtype))
    reference = torch.cat([basis._one_basis(x, flat) for flat in generated], dim=1)
    actual = basis(x)
    torch.testing.assert_close(actual, reference, atol=1e-10, rtol=1e-10)
    parameter = basis.hypernetwork[-1].weight
    actual_grad = torch.autograd.grad(actual.square().sum(), parameter, retain_graph=True)[0]
    reference_grad = torch.autograd.grad(reference.square().sum(), parameter)[0]
    torch.testing.assert_close(actual_grad, reference_grad, atol=1e-8, rtol=1e-8)
    def individual_forward(z):
        weights = basis.hypernetwork(basis.basis_ids.to(dtype=z.dtype))
        return torch.cat([basis._one_basis(z, flat) for flat in weights], dim=1)

    actual_derivatives = basis.derivatives(x)
    reference_derivatives = column_derivatives(individual_forward, x)
    for actual_column, reference_column in zip(actual_derivatives, reference_derivatives):
        torch.testing.assert_close(actual_column, reference_column, atol=1e-7, rtol=1e-7)
    actual_loss = sum(t.square().sum() for t in actual_derivatives)
    reference_loss = sum(t.square().sum() for t in reference_derivatives)
    actual_grad = torch.autograd.grad(actual_loss, parameter, retain_graph=True)[0]
    reference_grad = torch.autograd.grad(reference_loss, parameter)[0]
    torch.testing.assert_close(actual_grad, reference_grad, atol=1e-4, rtol=1e-7)
