import torch
import yaml

from solvers.tikhonov import solve_tikhonov
from scripts.run_mvp import run_mvp


def test_tikhonov_matches_normal_equations_and_keeps_gradient():
    matrix = torch.tensor([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]],
                          dtype=torch.float64, requires_grad=True)
    rhs = torch.tensor([[1.0], [0.0], [1.0]], dtype=torch.float64)
    beta = solve_tikhonov(matrix, rhs, 1e-4)
    expected = torch.linalg.solve(matrix.T @ matrix + 1e-4 * torch.eye(2), matrix.T @ rhs)
    assert torch.allclose(beta, expected, atol=1e-12)
    beta.sum().backward()
    assert matrix.grad is not None and torch.isfinite(matrix.grad).all()


def test_tikhonov_rejects_nonpositive_lambda():
    matrix = torch.eye(2, dtype=torch.float64)
    rhs = torch.ones((2, 1), dtype=torch.float64)
    for value in (0, -1e-6):
        try:
            solve_tikhonov(matrix, rhs, value)
        except ValueError:
            pass
        else:
            raise AssertionError('lambda must be positive')


def test_tikhonov_run_is_labeled_separately_from_hard_cnf(tmp_path):
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump({'seed': 42, 'device': 'cpu',
        'methods': {'cnf_tikhonov': {'ridge_lambda': 1e-4}}}))
    record = run_mvp(methods='cnf_tikhonov', output_root=tmp_path / 'run',
                     steps_override=1, config_path=config)['cnf_tikhonov']
    assert record['status'] == 'regularized'
    assert record['constraints'] == 'regularized'
    assert record['constraint_system'] == 'tikhonov_ablation'
    assert record['regularization_lambda'] == 1e-4
    assert record['hard_residual_max'] > 1e-8
