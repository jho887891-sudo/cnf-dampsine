"""Aggregate summary.json records, separating solve failures from conditioning flags."""
import argparse
import json
import math
from pathlib import Path
import statistics
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.report import condition_band


def _finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def aggregate(root):
    records = [json.loads(p.read_text()) for p in Path(root).rglob('summary.json')]
    groups = {}
    for row in records:
        groups.setdefault(row['method'], []).append(row)
    output = {}
    for method, rows in groups.items():
        errors = [r['rmse'] for r in rows if _finite(r.get('rmse'))]
        band = lambda row: row.get('condition_status') or condition_band(row.get('condition_number'))
        output[method] = {
            'runs': len(rows),
            'solve_failures': sum(r.get('status') == 'failed' or r.get('solver_ok') is False
                                  for r in rows),
            'infeasible': sum(r.get('status') == 'infeasible' for r in rows),
            'condition_warning': sum(band(r) == 'warning' for r in rows),
            'condition_unstable': sum(band(r) == 'unstable' for r in rows),
            'condition_severe': sum(band(r) == 'severe' for r in rows),
            'valid_rmse': len(errors),
            'mean_rmse': statistics.mean(errors) if errors else None,
            'std_rmse': statistics.stdev(errors) if len(errors) > 1 else 0.0,
            'median_rmse': statistics.median(errors) if errors else None,
            'max_rmse': max(errors) if errors else None,
        }
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root')
    parser.add_argument('--output', default='aggregate.json')
    args = parser.parse_args()
    result = aggregate(args.root)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))