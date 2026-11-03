"""Held-out OOD evaluation of a frozen CNF basis checkpoint."""

import argparse
import json
import math
from pathlib import Path
import sys
import statistics

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml

from data.generator import generate_sample
from evaluation.metrics import evaluate_prediction
from models.bases.gaussian_kernel import GaussianKernelBasis
from models.bases.hypernet_siren import HypernetSirenBasis
from models.cnf import CNFModel
from physics.collocation import chebyshev_interior


def run_frozen_ood(method, checkpoint_path, suite_path, output_root, device='auto'):
    if method not in ('cnf_gaussian', 'cnf_siren'):
        raise ValueError('frozen OOD supports CNF-Gaussian and CNF-SIREN')
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    spec = yaml.safe_load(Path(suite_path).read_text(encoding='utf-8'))
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    interior = chebyshev_interior(spec['base'].get('collocation_interior', 31)).to(device)
    anchors = torch.cat((interior, torch.tensor([[0.0], [1.0]], dtype=torch.float64, device=device)))
    basis = GaussianKernelBasis(anchors).double() if method == 'cnf_gaussian' else HypernetSirenBasis().double()
    basis.load_state_dict(checkpoint['basis'])
    model = CNFModel(basis).to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    grid = torch.linspace(0, 1, spec['base'].get('dense_points', 2048), dtype=torch.float64, device=device)[:, None]
    records = []
    for cohort in spec['runs']:
        for index in range(spec.get('sample_count', 1)):
            seed = cohort['seed'] + index
            gamma_range = cohort.get('gamma_range', spec['base'].get('gamma_range', (2, 4)))
            omega_range = cohort.get('omega_range_pi', spec['base'].get('omega_range_pi', (6, 8)))
            sample = generate_sample(seed=seed, gamma_range=gamma_range,
                omega_range=(omega_range[0]*math.pi, omega_range[1]*math.pi))
            sample = {key: value.to(device) if isinstance(value, torch.Tensor) else value
                      for key, value in sample.items()}
            run = {'method': method, 'cohort': cohort.get('cohort', f'cohort_{len(records)}'),
                   'seed': seed, 'train_split_hash': checkpoint['train_split_hash'],
                   'gamma': sample['gamma'], 'omega': sample['omega'], 'frozen_basis': True}
            try:
                beta, diagnostics = model.solve_beta(sample, interior)
                prediction = model(grid, beta) + sample['b']
                metrics, _ = evaluate_prediction(model, sample, grid, prediction, beta)
                condition = diagnostics['condition_number']
                run.update(status='numerical_failure' if condition > 1e10 else 'ok', **diagnostics, **metrics)
            except Exception as exc:
                run.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            records.append(run)
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'frozen_ood.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    lines = ['# Frozen-basis OOD evaluation', '',
             f'Method: {method}; train split hash: `{checkpoint["train_split_hash"]}`.',
             'The test curves did not update basis parameters. Physics parameters were supplied as oracle values.', '',
             '| Cohort | Runs | Valid | Failures | Mean RMSE | SD RMSE |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name in sorted({row['cohort'] for row in records}):
        rows = [row for row in records if row['cohort'] == name]
        valid = [row['rmse'] for row in rows if row['status'] == 'ok']
        mean = f'{statistics.mean(valid):.6g}' if valid else 'n/a'
        sd = f'{statistics.stdev(valid):.6g}' if len(valid) > 1 else 'n/a'
        lines.append(f'| {name} | {len(rows)} | {len(valid)} | {len(rows)-len(valid)} | {mean} | {sd} |')
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return records


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--suite', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    result = run_frozen_ood(args.method, args.checkpoint, args.suite, args.output_root, args.device)
    print({'runs': len(result), 'failures': sum(row['status'] != 'ok' for row in result)})
