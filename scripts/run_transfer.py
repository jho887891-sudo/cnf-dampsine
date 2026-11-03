"""Run P3 shared-basis transfer using a YAML protocol."""

import argparse
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from training.train_shared_basis import train_shared_basis


def run_transfer(config_path, output_root, *, train_count=None, val_count=None,
                 test_count=None, steps_per_curve=None, methods=None):
    config = yaml.safe_load(Path(config_path).read_text(encoding='utf-8'))
    names = methods or config['methods']
    results = {}
    for name in names:
        results[name] = train_shared_basis(name, Path(output_root)/name,
            train_count=train_count or config['train_count'],
            val_count=val_count or config['val_count'],
            test_count=test_count or config['test_count'],
            split_seed=config['split_seed'],
            steps_per_curve=steps_per_curve or config['steps_per_curve'])
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/sweeps/p3_transfer.yaml')
    parser.add_argument('--output-root', default='results/p3_transfer')
    parser.add_argument('--train-count', type=int)
    parser.add_argument('--val-count', type=int)
    parser.add_argument('--test-count', type=int)
    parser.add_argument('--steps-per-curve', type=int)
    parser.add_argument('--methods', nargs='+')
    args = parser.parse_args()
    print(run_transfer(args.config, args.output_root, train_count=args.train_count,
        val_count=args.val_count, test_count=args.test_count,
        steps_per_curve=args.steps_per_curve, methods=args.methods))
