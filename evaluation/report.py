"""Evidence-backed summary of available experiment records."""

import json
from pathlib import Path
import statistics


def build_report(results_root, output_path):
    paths = sorted(Path(results_root).rglob('summary.json'))
    groups = {}
    for path in paths:
        row = json.loads(path.read_text(encoding='utf-8'))
        label = row['method'] + (' (rectangular)' if row.get('constraint_system') == 'rectangular_extension' else '')
        groups.setdefault(label, []).append(row)
    lines = ['# CNF-DampSine experiment report', '',
             'Physics parameters are oracle values. Hard equalities apply only at finite collocation points.',
             'Dense ODE residual is a separate empirical evaluation. Numerical failures are retained.', '',
             '| Method | Runs | Valid | Failures | Mean RMSE | SD RMSE | Median RMSE |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for method, rows in sorted(groups.items()):
        valid = [r['rmse'] for r in rows if r.get('status') == 'ok' and 'rmse' in r]
        mean = f'{statistics.mean(valid):.6g}' if valid else 'n/a'
        std = f'{statistics.stdev(valid):.6g}' if len(valid) > 1 else 'n/a'
        median = f'{statistics.median(valid):.6g}' if valid else 'n/a'
        lines.append(f'| {method} | {len(rows)} | {len(valid)} | {len(rows)-len(valid)} | {mean} | {std} | {median} |')
    lines += ['', '## Failed or numerically unstable runs', '']
    for method, rows in sorted(groups.items()):
        for row in rows:
            if row.get('status') != 'ok':
                lines.append(f'- {method}: {row.get("status")} at {row.get("run_dir")}; '
                             f'condition={row.get("condition_number", "n/a")}; {row.get("error", "")}')
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return target


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('results_root')
    parser.add_argument('--output', default='results/report.md')
    args = parser.parse_args()
    print(build_report(args.results_root, args.output))
