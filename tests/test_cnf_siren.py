import torch

from models.bases.hypernet_siren import HypernetSirenBasis


def test_hypernet_siren_produces_distinct_columns_and_derivatives():
    basis = HypernetSirenBasis(num_basis=33, width=21, hidden_layers=1, omega0=30).double()
    x = torch.tensor([[0.1], [0.4]], dtype=torch.float64)
    phi, d1, d2 = basis.derivatives(x)
    assert phi.shape == d1.shape == d2.shape == (2, 33)
    assert torch.isfinite(phi).all() and torch.isfinite(d2).all()
    assert not torch.allclose(phi[:, 0], phi[:, 1])
    assert any(p.requires_grad for p in basis.hypernetwork.parameters())
