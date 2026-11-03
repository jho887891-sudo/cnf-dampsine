"""Run selected MVP methods on one reproducible damped-sine sample."""

import argparse
import json
import math
import time
import platform
import subprocess
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import yaml

from data.generator import generate_sample
from evaluation.metrics import evaluate_prediction
from models.bases.fourier_basis import FourierBasis
from models.bases.gaussian_kernel import GaussianKernelBasis
from models.bases.hypernet_siren import HypernetSirenBasis
from models.bases.separate_siren import SeparateSirenBasis
from models.cnf import CNFModel
from models.fourier_collocation import FourierCollocation
from models.siren_pinn import SirenPINN
from physics.collocation import chebyshev_interior
from training.loops import train_cnf_sample, train_siren_pinn


class InapplicableBasis(Exception):
    pass


def run_mvp(methods='all', output_root='results', steps_override=None, config_path=None):
    config = yaml.safe_load(Path(config_path).read_text(encoding='utf-8')) if config_path else {}
    names = ['cnf_gaussian', 'cnf_siren', 'fourier', 'siren_pinn'] if methods == 'all' else methods.split(',')
    root = Path(output_root)
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    sample = generate_sample(seed=seed, n_sensors=config.get('sensors', 128),
        n_obs=config.get('observations', 32), noise=config.get('noise', 0.0),
        gamma_range=config.get('gamma_range', (2.0, 4.0)),
        omega_range=[math.pi*v for v in config.get('omega_range_pi', (6.0, 8.0))])
    tolerance = config.get('right_bc_degenerate_rel_tol', 1e-8)
    boundary_mode = ('initial_value' if abs(float(sample['q_right'])) <= tolerance * max(abs(sample['A']), 1e-12)
                     else 'dirichlet')
    sample['boundary_mode'] = boundary_mode
    m = config.get('collocation_interior', 31)
    layout = config.get('point_layout', 'chebyshev')
    if layout == 'chebyshev':
        collocation = chebyshev_interior(m)
    elif layout == 'uniform':
        collocation = torch.linspace(0, 1, m + 2, dtype=torch.float64)[1:-1, None]
    elif layout == 'random':
        collocation = torch.rand(m, 1, generator=torch.Generator().manual_seed(seed), dtype=torch.float64).sort(dim=0).values
    else:
        raise ValueError(f'unknown point layout: {layout}')
    k = config.get('basis_count', m + 2)
    if k != m + 2 and not config.get('rectangular_extension'):
        raise ValueError('square MVP requires basis_count == collocation_interior + 2; rectangular sweeps need a separate solver')
    grid = torch.linspace(0, 1, config.get('dense_points', 2048), dtype=torch.float64)[:, None]
    results = {}
    for name in names:
        run_dir = root / name
        run_dir.mkdir(parents=True, exist_ok=True)
        method_config = config.get('methods', {}).get(name, {})
        record = {'method': name, 'run_dir': str(run_dir), 'seed': seed, 'K': k,
                  'M': len(collocation), 'Nobs': len(sample['x_obs']),
                  'sensors': len(sample['x_sensor']), 'dense_points': len(grid), 'noise': sample['noise'],
                  'point_layout': layout,
                  'boundary_mode': boundary_mode,
                  'constraint_system': 'rectangular_extension' if k != m + 2 else 'square_main',
                  'parameters_known': True, 'constraints': 'hard' if name != 'siren_pinn' else 'soft'}
        record['physics_parameters'] = {key: sample[key] for key in ('A', 'gamma', 'omega', 'b')}
        try:
            started = time.perf_counter()
            beta = None
            if name == 'cnf_gaussian' and boundary_mode == 'initial_value':
                raise InapplicableBasis('ordinary Gaussian kernel cannot use duplicate x=0 anchors for value and derivative')
            if name == 'fourier':
                frequency_step = method_config.get('frequency_step', math.pi)
                if frequency_step == 'pi':
                    frequency_step = math.pi
                model = FourierCollocation(harmonics=(k-1)//2,
                    frequency_step=frequency_step)
                beta, diagnostics = model.solve(sample, collocation,
                    rectangular_extension=config.get('rectangular_extension', False), boundary_mode=boundary_mode)
                prediction = model.predict(grid, beta, sample['b'])
            elif name in ('cnf_gaussian', 'cnf_siren'):
                if k != m + 2:
                    raise ValueError('rectangular CNF basis training is not implemented; use separate extension')
                anchors = torch.cat((collocation, torch.tensor([[0.0], [1.0]], dtype=torch.float64)))
                if name == 'cnf_gaussian':
                    basis = GaussianKernelBasis(anchors, sigma=method_config.get('sigma', 0.05),
                        n_freqs=method_config.get('n_freqs', 3), learn_sigma=method_config.get('learn_sigma', False))
                else:
                    basis_type = method_config.get('basis', 'hypernet_siren')
                    basis_class = {'hypernet_siren': HypernetSirenBasis,
                                   'sepbasis_siren': SeparateSirenBasis}.get(basis_type)
                    if basis_class is None:
                        raise ValueError(f'unknown CNF-SIREN basis: {basis_type}')
                    basis = basis_class(num_basis=k, width=method_config.get('width', 21),
                        hidden_layers=method_config.get('hidden_layers', 1), omega0=method_config.get('omega0', 30))
                model = CNFModel(basis).double()
                model, history = train_cnf_sample(model, sample, collocation,
                    steps=steps_override or method_config.get('steps', 5000),
                    lr=method_config.get('lr', 1e-3 if name == 'cnf_gaussian' else 3e-4),
                    patience=method_config.get('patience', 500),
                    condition_weight=method_config.get('condition_weight', 1e-6), boundary_mode=boundary_mode)
                beta, diagnostics = model.solve_beta(sample, collocation, boundary_mode=boundary_mode)
                prediction = model(grid, beta) + sample['b']
                record['history_steps'] = len(history)
            elif name == 'siren_pinn':
                model = SirenPINN(width=method_config.get('width', 64),
                    hidden_layers=method_config.get('hidden_layers', 3), omega0=method_config.get('omega0', 30)).double()
                model, history = train_siren_pinn(model, sample, collocation,
                    steps=steps_override or method_config.get('steps', 10000),
                    lr=method_config.get('lr', 1e-4), patience=method_config.get('patience', 1000),
                    weights={'data': method_config.get('lambda_data', 1),
                             'ode': method_config.get('lambda_ode', 1),
                             'bc': method_config.get('lambda_bc', 100)})
                prediction = model(grid) + sample['b']
                diagnostics = {'solver_residual': None}
                record['history_steps'] = len(history)
            else:
                raise ValueError(f'unknown method: {name}')
            metrics, dense_residual = evaluate_prediction(model, sample, grid, prediction, beta)
            truth = sample['A'] * torch.exp(-sample['gamma']*grid) * torch.sin(sample['omega']*grid) + sample['b']
            condition = diagnostics.get('condition_number')
            status = ('infeasible' if diagnostics.get('hard_feasible') is False else
                      'numerical_failure' if condition is not None and (not np.isfinite(condition) or condition > 1e10) else 'ok')
            record.update(status=status, **metrics, **diagnostics,
                          total_time_seconds=time.perf_counter() - started,
                          parameter_count=sum(p.numel() for p in model.parameters()))
            np.savez_compressed(run_dir / 'predictions.npz', x=grid.numpy(), y_true=truth.detach().numpy(),
                                y_pred=prediction.detach().numpy(), dense_ode_residual=dense_residual.numpy(),
                                x_obs=sample['x_obs'].numpy(), y_obs=sample['y_obs'].numpy())
            if beta is not None:
                np.save(run_dir / 'beta.npy', beta.detach().numpy())
            if name != 'fourier':
                (run_dir / 'train.log').write_text('\n'.join(json.dumps(row) for row in history) + '\n', encoding='utf-8')
            else:
                (run_dir / 'train.log').write_text('Fourier collocation: direct solve, no optimizer.\n', encoding='utf-8')
        except InapplicableBasis as exc:
            record.update(status='inapplicable', error=str(exc))
        except Exception as exc:
            record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        (run_dir / 'config.yaml').write_text(yaml.safe_dump({'run': record, 'source': config}, sort_keys=True), encoding='utf-8')
        (run_dir / 'summary.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
        environment = {'python': platform.python_version(), 'platform': platform.platform(),
                       'torch': torch.__version__, 'numpy': np.__version__,
                       'cuda_available': torch.cuda.is_available()}
        (run_dir / 'environment.txt').write_text(json.dumps(environment, indent=2), encoding='utf-8')
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, text=True, check=False)
        (run_dir / 'git_commit.txt').write_text(commit.stdout.strip() if commit.returncode == 0 else 'unversioned', encoding='utf-8')
        results[name] = record
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods', default='all')
    parser.add_argument('--output-root', default='results/mvp')
    parser.add_argument('--steps-override', type=int)
    parser.add_argument('--config')
    args = parser.parse_args()
    print(json.dumps(run_mvp(args.methods, args.output_root, args.steps_override, args.config), indent=2))
