"""Inductive shared-basis protocol: train on train curves, freeze on test."""

import hashlib
import json
from pathlib import Path

import torch

from data.generator import generate_sample
from models.cnf import CNFModel
from models.bases.gaussian_kernel import GaussianKernelBasis
from models.bases.hypernet_siren import HypernetSirenBasis
from physics.collocation import chebyshev_interior
from training.loops import train_cnf_sample


def train_shared_basis(method, output_dir, *, train_count=800, test_count=100,
                       split_seed=42, steps_per_curve=5000):
    if method not in ('cnf_gaussian', 'cnf_siren'):
        raise ValueError('shared basis supports the two CNF methods')
    if train_count < 1 or test_count < 1:
        raise ValueError('both splits must be nonempty')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    interior = chebyshev_interior(31)
    anchors = torch.cat((interior, torch.tensor([[0.0], [1.0]], dtype=torch.float64)))
    basis = GaussianKernelBasis(anchors).double() if method == 'cnf_gaussian' else HypernetSirenBasis().double()
    model = CNFModel(basis)
    train_seeds = list(range(split_seed, split_seed + train_count))
    test_seeds = list(range(split_seed + train_count, split_seed + train_count + test_count))
    (output / 'split.json').write_text(json.dumps({'train_seeds': train_seeds, 'test_seeds': test_seeds}), encoding='utf-8')
    digest = hashlib.sha256(json.dumps(train_seeds).encode()).hexdigest()
    for seed in train_seeds:
        sample = generate_sample(seed=seed)
        train_cnf_sample(model, sample, interior, steps=steps_per_curve,
                         lr=1e-3 if method == 'cnf_gaussian' else 3e-4)
    torch.save({'basis': basis.state_dict(), 'train_split_hash': digest}, output / 'frozen_basis.pt')
    for parameter in basis.parameters():
        parameter.requires_grad_(False)
    records = []
    for seed in test_seeds:
        sample = generate_sample(seed=seed)
        beta, diagnostics = model.solve_beta(sample, interior)
        records.append({'seed': seed, 'parameters': {key: sample[key] for key in ('A','gamma','omega','b')},
                        'beta': beta.detach().flatten().tolist(), **diagnostics})
    (output / 'test_results.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    return {'train_split_hash': digest, 'test_count': len(records), 'output_dir': str(output)}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['cnf_gaussian', 'cnf_siren'], required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--train-count', type=int, default=800)
    parser.add_argument('--test-count', type=int, default=100)
    parser.add_argument('--split-seed', type=int, default=42)
    parser.add_argument('--steps-per-curve', type=int, default=5000)
    args = parser.parse_args()
    print(train_shared_basis(args.method, args.output_dir, train_count=args.train_count,
          test_count=args.test_count, split_seed=args.split_seed, steps_per_curve=args.steps_per_curve))
