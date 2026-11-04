"""Aggregate P3 transfer by split seed without hiding numerical failures."""

import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def report(results_root):
    root = Path(results_root)
    dirs = [root / 'p3_global_full', *sorted(root.glob('p3_global_seed*'))]
    groups = {}
    for directory in dirs:
        for summary_path in directory.glob('*/transfer_summary.json'):
            summary = json.loads(summary_path.read_text())
            split = json.loads((summary_path.parent / 'split.json').read_text())
            seed = split['train_seeds'][0]
            records = json.loads((summary_path.parent / 'test_results.json').read_text())
            groups.setdefault(summary['method'], []).append((seed, summary, records))
    if not groups:
        raise ValueError('no completed P3 split results')

    lines = ['# P3 multi-seed shared-basis transfer', '',
             'Each split trains on 800 curves and evaluates 100 independent test curves with a frozen basis.',
             'Mean and SD below summarize the per-split test means. Incomplete methods remain labeled as such.', '',
             '| Method | Completed seeds | Mean of test means | SD across seeds | Pooled median RMSE | Solver failures | Unstable conditions |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    fig, ax = plt.subplots(figsize=(7, 4))
    for method, rows in sorted(groups.items()):
        rows.sort(key=lambda row: row[0])
        means = [row[1]['test']['mean_rmse'] for row in rows]
        all_errors = [record['rmse'] for _, _, records in rows for record in records]
        failures = sum(row[1]['test']['solver_failures'] for row in rows)
        unstable = sum(row[1]['test']['unstable_condition_count'] for row in rows)
        sd = statistics.stdev(means) if len(means) > 1 else None
        sd_text = f'{sd:.6g}' if sd is not None else 'n/a'
        lines.append(f'| {method} | {len(rows)}/5 | {statistics.mean(means):.6g} | '
                     f'{sd_text} | {statistics.median(all_errors):.6g} | '
                     f'{failures} | {unstable} |')
        ax.plot([row[0] for row in rows], means, marker='o', label=method)
    ax.set(xlabel='Split seed', ylabel='Mean test RMSE')
    ax.legend()
    ax.grid(alpha=.2)
    fig.tight_layout()
    output = root / 'p3_multiseed'
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / 'seed_means.png', dpi=160)
    plt.close(fig)
    (output / 'report.md').write_text('\n'.join(lines) + '\n')
    return output / 'report.md'


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('results_root')
    print(report(parser.parse_args().results_root))
