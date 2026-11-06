import torch

from data.generator import generate_sample
from models.bases.gaussian_kernel import GaussianKernelBasis
from models.cnf import CNFModel
from models.siren_pinn import SirenPINN
from physics.collocation import chebyshev_interior
from training.loops import train_cnf_sample, train_siren_pinn


def test_pinn_training_updates_weights_and_logs_components():
    sample = generate_sample(seed=42)
    model = SirenPINN(width=8, hidden_layers=1).double()
    before = next(model.parameters()).detach().clone()
    model, history = train_siren_pinn(model, sample, chebyshev_interior(3), steps=2)
    assert len(history) == 2
    assert {'data', 'ode', 'bc', 'total'} <= history[0].keys()
    assert not torch.equal(before, next(model.parameters()))


def test_cnf_training_reuses_condition_number_for_loss_and_log(monkeypatch):
    sample = generate_sample(seed=42)
    interior = chebyshev_interior(3)
    anchors = torch.cat((interior, torch.tensor([[0.0], [1.0]], dtype=torch.float64)))
    model = CNFModel(GaussianKernelBasis(anchors, sigma=0.05, n_freqs=3))
    actual_cond = torch.linalg.cond
    calls = []

    def counted_cond(matrix):
        calls.append(1)
        return actual_cond(matrix)

    monkeypatch.setattr(torch.linalg, 'cond', counted_cond)
    _, history = train_cnf_sample(model, sample, interior, steps=2)
    assert len(history) == 2
    assert len(calls) == 2
    assert all('condition_number' in row and 'total' in row for row in history)
