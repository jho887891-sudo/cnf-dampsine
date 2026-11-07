"""Run resumable P2 noise evaluation on disjoint 100-curve seed splits."""

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from scripts.run_mvp import run_mvp


DEFAULT_SPLIT_SEEDS = (42, 123, 2026, 3407, 7777)
NOISE_LEVELS = (0.01, 0.05)
METHODS = ('cnf_gaussian', 'cnf_siren', 'fourier', 'siren_pinn')


def sample_seed(split_seed, sample_index):
    """Assign each formal split a disjoint seed block away from smoke runs."""
    if split_seed < 0 or not 0 <= sample_index < 1000:
        raise ValueError('split seed must be nonnegative; sample index must be below 1000')
    return 1_000_000 + split_seed * 1000 + sample_index


def iter_cases(split_seeds=DEFAULT_SPLIT_SEEDS, sample_count=100, noise_levels=NOISE_LEVELS):
    if not 1 <= sample_count <= 1000:
        raise ValueError('sample count must be between 1 and 1000')
    if len(set(split_seeds)) != len(split_seeds):
        raise ValueError('split seeds must be unique')
    for split_seed in split_seeds:
        for noise in noise_levels:
            for sample_index in range(sample_count):
                yield split_seed, noise, sample_index, sample_seed(split_seed, sample_index)


def run(config_path, output_root, methods, split_seeds=DEFAULT_SPLIT_SEEDS,
        sample_count=100, noise_levels=NOISE_LEVELS):
    selected = tuple(methods)
    if not selected or any(name not in METHODS for name in selected):
        raise ValueError(f'methods must be selected from {METHODS}')
    if not noise_levels or len(set(noise_levels)) != len(noise_levels) or any(not 0 <= level < 1 for level in noise_levels):
        raise ValueError('noise levels must be distinct values in [0, 1)')
    base = yaml.safe_load(Path(config_path).read_text(encoding='utf-8'))['base']
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / 'protocol.json'
    protocol = {'split_seeds': list(split_seeds), 'sample_count_per_noise': sample_count,
                'noise_levels': list(noise_levels), 'seed_formula': '1000000 + split_seed*1000 + sample_index',
                'config_base': base}
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != protocol:
        raise ValueError('existing output protocol differs from requested protocol')
    manifest_path.write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    for split_seed, noise, index, seed in iter_cases(split_seeds, sample_count, noise_levels):
        case = root / f'seed{split_seed}' / f'noise{int(noise*100):02d}' / f'sample{index:03d}'
        config = dict(base, seed=seed, noise=noise)
        config.setdefault('device', 'auto')
        config_file = case / 'input.yaml'
        case.mkdir(parents=True, exist_ok=True)
        if config_file.exists() and yaml.safe_load(config_file.read_text()) != config:
            raise ValueError(f'existing case config differs: {config_file}')
        config_file.write_text(yaml.safe_dump(config, sort_keys=True), encoding='utf-8')
        for method in selected:
            summary_file = case / method / 'summary.json'
            if summary_file.exists():
                summary = json.loads(summary_file.read_text())
                if (summary.get('seed'), summary.get('noise'), summary.get('method')) != (seed, noise, method):
                    raise ValueError(f'existing result differs: {summary_file}')
                continue
            run_mvp(methods=method, output_root=case, config_path=config_file)
            print(f'completed split={split_seed} noise={noise} sample={index} method={method}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/sweeps/p2_noise.yaml')
    parser.add_argument('--output-root', default='results/p2_noise_formal')
    parser.add_argument('--methods', nargs='+', required=True, choices=METHODS)
    parser.add_argument('--split-seeds', nargs='+', type=int, default=DEFAULT_SPLIT_SEEDS)
    parser.add_argument('--sample-count', type=int, default=100)
    parser.add_argument('--noise-levels', nargs='+', type=float, default=NOISE_LEVELS)
    args = parser.parse_args()
    run(args.config, args.output_root, args.methods, tuple(args.split_seeds), args.sample_count,
        tuple(args.noise_levels))
