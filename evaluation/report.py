"""Evidence-backed summary of available experiment records."""

import json
import math
from pathlib import Path
import statistics


def condition_band(condition_number):
    """Mirror of the SPEC 4.5 bands, used when a record predates the field."""
    if condition_number is None:
        return 'unknown'
    value = float(condition_number)
    if not math.isfinite(value):
        return 'severe'
    if value < 1e6:
        return 'normal'
    if value <= 1e10:
        return 'warning'
    if value <= 1e12:
        return 'unstable'
    return 'severe'


def build_report(results_root, output_path):
    paths = sorted(Path(results_root).rglob('summary.json'))
    groups = {}
    for path in paths:
        row = json.loads(path.read_text(encoding='utf-8'))
        label = row['method'] + (' (rectangular)' if row.get('constraint_system') == 'rectangular_extension' else '')
        groups.setdefault(label, []).append(row)
    lines = ['# CNF-DampSine experiment report', '',
             'Physics parameters are oracle values. Hard equalities apply only at finite collocation points.',
             'Dense ODE residual is a separate empirical evaluation. Numerical failures are retained.',
             'Condition-number warnings are reported separately from solve failures.', '',
             '| Method | Runs | Valid | Warnings | Unstable | Severe | Infeasible | Failed | Mean RMSE | SD RMSE | Median RMSE |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for method, rows in sorted(groups.items()):
        valid = [r['rmse'] for r in rows if r.get('status') == 'ok' and 'rmse' in r]
        mean = f'{statistics.mean(valid):.6g}' if valid else 'n/a'
        std = f'{statistics.stdev(valid):.6g}' if len(valid) > 1 else 'n/a'
        median = f'{statistics.median(valid):.6g}' if valid else 'n/a'
        band = lambda row: row.get('condition_status') or condition_band(row.get('condition_number'))
        warnings = sum(1 for r in rows if r.get('status') == 'ok' and band(r) == 'warning')
        unstable = sum(1 for r in rows if band(r) == 'unstable')
        severe = sum(1 for r in rows if band(r) == 'severe')
        infeasible = sum(1 for r in rows if r.get('status') == 'infeasible')
        failed = sum(1 for r in rows if r.get('status') == 'failed')
        lines.append(f'| {method} | {len(rows)} | {len(valid)} | {warnings} | {unstable} | {severe} | '
                     f'{infeasible} | {failed} | {mean} | {std} | {median} |')
    lines += ['', '## Condition-number warnings (retained, not failures)', '']
    warned = [(method, row) for method, rows in sorted(groups.items()) for row in rows
              if row.get('status') == 'ok'
              and (row.get('condition_status') or condition_band(row.get('condition_number'))) == 'warning']
    for method, row in warned:
        lines.append(f'- {method} at {row.get("run_dir")}; condition={row.get("condition_number", "n/a"):.3e}; '
                     f'rmse={row.get("rmse", "n/a")}')
    if not warned:
        lines.append('- none')
    lines += ['', '## Failed or numerically unstable runs', '']
    for method, rows in sorted(groups.items()):
        for row in rows:
            if row.get('status') != 'ok':
                lines.append(f'- {method}: {row.get("status")} '
                             f'[{row.get("condition_status") or condition_band(row.get("condition_number"))}] '
                             f'at {row.get("run_dir")}; condition={row.get("condition_number", "n/a")}; '
                             f'{row.get("error", "")}')
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
