"""Aggregate summary.json records while retaining failed runs."""
import argparse
import json
from pathlib import Path
import statistics


def aggregate(root):
    records = [json.loads(p.read_text()) for p in Path(root).rglob('summary.json')]
    groups = {}
    for row in records:
        groups.setdefault(row['method'], []).append(row)
    output = {}
    for method, rows in groups.items():
        valid = [r for r in rows if r.get('status') == 'ok']
        output[method] = {'runs': len(rows), 'failures': sum(r.get('status') != 'ok' for r in rows),
                          'mean_rmse': statistics.mean(r['rmse'] for r in valid) if valid else None,
                          'std_rmse': statistics.stdev(r['rmse'] for r in valid) if len(valid) > 1 else 0.0}
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root')
    parser.add_argument('--output', default='summary.json')
    args = parser.parse_args()
    result = aggregate(args.root)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
