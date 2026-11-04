"""Inductive shared-basis protocol with frozen validation and test evaluation."""

import hashlib
import json
from pathlib import Path
import statistics

import torch

from data.generator import generate_sample
from evaluation.metrics import evaluate_prediction
from models.cnf import CNFModel
from models.bases.gaussian_kernel import GaussianKernelBasis
from models.bases.hypernet_siren import HypernetSirenBasis
from physics.collocation import chebyshev_interior
from training.loops import train_cnf_sample


def train_shared_basis(method, output_dir, *, train_count=800, test_count=100,
                       val_count=100, split_seed=42, steps_per_curve=5000, device='auto'):
    if method not in ('cnf_gaussian', 'cnf_siren'):
        raise ValueError('shared basis supports the two CNF methods')
    if train_count < 1 or test_count < 1 or val_count < 1:
        raise ValueError('all splits must be nonempty')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    interior = chebyshev_interior(31).to(device)
    anchors = torch.cat((interior, torch.tensor([[0.0], [1.0]], dtype=torch.float64, device=device)))
    basis = GaussianKernelBasis(anchors).double() if method == 'cnf_gaussian' else HypernetSirenBasis().double()
    model = CNFModel(basis).to(device)
    train_seeds = list(range(split_seed, split_seed + train_count))
    val_seeds = list(range(split_seed + train_count, split_seed + train_count + val_count))
    test_seeds = list(range(split_seed + train_count + val_count,
                            split_seed + train_count + val_count + test_count))
    (output / 'split.json').write_text(json.dumps({'train_seeds': train_seeds,
        'val_seeds': val_seeds, 'test_seeds': test_seeds}), encoding='utf-8')
    digest = hashlib.sha256(json.dumps(train_seeds).encode()).hexdigest()
    for seed in train_seeds:
        sample = generate_sample(seed=seed)
        sample = {key: value.to(device) if isinstance(value, torch.Tensor) else value
                  for key, value in sample.items()}
        train_cnf_sample(model, sample, interior, steps=steps_per_curve,
                         lr=1e-3 if method == 'cnf_gaussian' else 3e-4)
    torch.save({'basis': {key: value.detach().cpu() for key, value in basis.state_dict().items()},
                'train_split_hash': digest}, output / 'frozen_basis.pt')
    for parameter in basis.parameters():
        parameter.requires_grad_(False)
    grid = torch.linspace(0, 1, 2048, dtype=torch.float64, device=device)[:, None]
    def evaluate_split(seeds, name):
        records = []
        for seed in seeds:
            sample = generate_sample(seed=seed)
            sample = {key: value.to(device) if isinstance(value, torch.Tensor) else value
                      for key, value in sample.items()}
            beta, diagnostics = model.solve_beta(sample, interior)
            with torch.enable_grad():
                prediction = model(grid, beta) + sample['b']
                metrics, _ = evaluate_prediction(model, sample, grid, prediction, beta)
            records.append({'seed': seed,
                            'parameters': {key: sample[key] for key in ('A','gamma','omega','b')},
                            'beta': beta.detach().cpu().flatten().tolist(), **diagnostics, **metrics})
        (output / f'{name}_results.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
        return records

    validation = evaluate_split(val_seeds, 'val')
    test = evaluate_split(test_seeds, 'test')
    def summarize(records):
        errors = [row['rmse'] for row in records]
        return {'count': len(records),
                'mean_rmse': statistics.mean(errors),
                'median_rmse': statistics.median(errors),
                'sd_rmse': statistics.stdev(errors) if len(errors) > 1 else None,
                'solver_failures': sum(not row.get('solver_ok', True) for row in records),
                'unstable_condition_count': sum(row.get('condition_number', 0) > 1e10
                                                for row in records)}
    (output / 'transfer_summary.json').write_text(json.dumps({
        'method': method, 'train_count': len(train_seeds), 'train_split_hash': digest,
        'steps_per_curve': steps_per_curve, 'validation': summarize(validation),
        'test': summarize(test)}, indent=2), encoding='utf-8')
    return {'train_split_hash': digest, 'val_count': len(validation),
            'test_count': len(test), 'output_dir': str(output)}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['cnf_gaussian', 'cnf_siren'], required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--train-count', type=int, default=800)
    parser.add_argument('--test-count', type=int, default=100)
    parser.add_argument('--val-count', type=int, default=100)
    parser.add_argument('--split-seed', type=int, default=42)
    parser.add_argument('--steps-per-curve', type=int, default=5000)
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    print(train_shared_basis(args.method, args.output_dir, train_count=args.train_count,
          test_count=args.test_count, val_count=args.val_count, split_seed=args.split_seed,
          steps_per_curve=args.steps_per_curve, device=args.device))
