"""Small configurable MVP sweep runner."""
import argparse
from pathlib import Path
import sys
import yaml

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_mvp import run_mvp


def run_sweep(suite, config_path, output_root='results/sweeps', steps_override=None, methods=None):
    spec = yaml.safe_load(Path(config_path).read_text(encoding='utf-8'))
    if not spec.get('runs'):
        raise ValueError('sweep has no runs; supply explicit parameter ranges')
    results = []
    sample_count = spec.get('sample_count', 1)
    if sample_count < 1:
        raise ValueError('sample_count must be positive')
    for i, params in enumerate(spec['runs']):
        for j in range(sample_count):
            merged = dict(spec.get('base', {}), **params)
            merged['seed'] = merged.get('seed', 42) + j
            run_id = f'{suite}_{i:03d}_sample{j:03d}'
            path = Path(output_root) / f'{run_id}.yaml'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(merged), encoding='utf-8')
            results.append(run_mvp(methods=methods or merged.get('selected_methods', 'all'),
                steps_override=steps_override, output_root=Path(output_root)/run_id, config_path=path))
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output-root', default='results/sweeps')
    parser.add_argument('--steps-override', type=int)
    parser.add_argument('--methods')
    args = parser.parse_args()
    run_sweep(args.suite, args.config, args.output_root, args.steps_override, args.methods)
