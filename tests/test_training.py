import torch

from data.generator import generate_sample
from models.siren_pinn import SirenPINN
from physics.collocation import chebyshev_interior
from training.loops import train_siren_pinn


def test_pinn_training_updates_weights_and_logs_components():
    sample = generate_sample(seed=42)
    model = SirenPINN(width=8, hidden_layers=1).double()
    before = next(model.parameters()).detach().clone()
    model, history = train_siren_pinn(model, sample, chebyshev_interior(3), steps=2)
    assert len(history) == 2
    assert {'data', 'ode', 'bc', 'total'} <= history[0].keys()
    assert not torch.equal(before, next(model.parameters()))
