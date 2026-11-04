"""Report frozen shared-basis validation and test results."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def report(root):
    root = Path(root)
    methods = {}
    for path in sorted(root.glob('*/transfer_summary.json')):
        method = path.parent.name
        methods[method] = (json.loads(path.read_text()),
                           json.loads((path.parent / 'test_results.json').read_text()))
    if not methods:
        raise ValueError('no completed transfer results found')

    lines = ['# P3 frozen shared-basis transfer', '',
             'Training uses only the train split. Validation and test freeze the basis and solve a new beta per curve.',
             'This is one split seed; the specified multi-seed statistical replication remains separate.', '',
             '| Method | Train curves | Global steps | Val curves | Val mean RMSE | Test curves | Test mean RMSE | Test SD | Test median | Solver failures | Unstable conditions |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for method, (summary, records) in methods.items():
        val, test = summary['validation'], summary['test']
        lines.append(f"| {method} | {summary['train_count']} | {summary['total_steps']} | "
                     f"{val['count']} | {val['mean_rmse']:.6g} | {test['count']} | "
                     f"{test['mean_rmse']:.6g} | {test['sd_rmse']:.6g} | "
                     f"{test['median_rmse']:.6g} | {test['solver_failures']} | "
                     f"{test['unstable_condition_count']} |")
        errors = [row['rmse'] for row in records]
        conditions = [row['condition_number'] for row in records]
        axes[0].hist(errors, bins=20, alpha=.5, label=method)
        axes[1].scatter(conditions, errors, s=14, alpha=.6, label=method)
    axes[0].set(xlabel='Test RMSE', ylabel='Curves')
    axes[1].set(xlabel='Matrix condition number', ylabel='Test RMSE', xscale='log')
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(root / 'test_diagnostics.png', dpi=160)
    plt.close(fig)
    (root / 'report.md').write_text('\n'.join(lines) + '\n')
    return root / 'report.md'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root')
    print(report(parser.parse_args().root))
