"""Summarize completed P2 noise splits without hiding poor fits or failures."""

import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def report(root):
    root = Path(root)
    protocol = json.loads((root / 'protocol.json').read_text())
    lines = ['# P2 formal noise evaluation', '',
             'Five independent split seeds; 100 curves per noise level and method in each split.',
             'The same curve seeds are used across noise levels within a split.',
             'Every saved run is counted. High-error fits remain in the mean and maximum.', '',
             '| Noise | Method | Splits complete | Runs | Valid | Mean of split means | SD of split means | Pooled median RMSE | Max RMSE | Warnings | Numerical failures |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    detail = ['## Per-split results', '',
              '| Split seed | Noise | Method | Runs | Valid | Mean RMSE | Median RMSE | Max RMSE | Warnings | Numerical failures |',
              '| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for noise in protocol['noise_levels']:
        noise_name = f'noise{int(noise*100):02d}'
        for method in ('cnf_gaussian', 'cnf_siren'):
            means, all_errors, seeds = [], [], []
            runs = valid_count = warnings = failures = 0
            for seed in protocol['split_seeds']:
                paths = sorted((root / f'seed{seed}' / noise_name).glob(f'sample*/{method}/summary.json'))
                rows = [json.loads(p.read_text()) for p in paths]
                valid = [r['rmse'] for r in rows if r.get('status') == 'ok' and r.get('rmse') is not None]
                bad = sum(r.get('status') != 'ok' for r in rows)
                warning = sum(r.get('condition_status') == 'warning' for r in rows)
                runs += len(rows); valid_count += len(valid); warnings += warning; failures += bad
                all_errors.extend(valid)
                if len(rows) == protocol['sample_count_per_noise']:
                    seeds.append(seed)
                    means.append(statistics.mean(valid) if valid else float('nan'))
                detail.append(f'| {seed} | {noise:.0%} | {method} | {len(rows)} | {len(valid)} | '
                              f'{statistics.mean(valid):.6g} | {statistics.median(valid):.6g} | '
                              f'{max(valid):.6g} | {warning} | {bad} |' if valid else
                              f'| {seed} | {noise:.0%} | {method} | {len(rows)} | 0 | n/a | n/a | n/a | {warning} | {bad} |')
            if means:
                ax.plot(seeds, means, marker='o', label=f'{method}, {noise:.0%}')
            mean_text = f'{statistics.mean(means):.6g}' if means else 'n/a'
            sd_text = f'{statistics.stdev(means):.6g}' if len(means) > 1 else 'n/a'
            median_text = f'{statistics.median(all_errors):.6g}' if all_errors else 'n/a'
            maximum_text = f'{max(all_errors):.6g}' if all_errors else 'n/a'
            lines.append(f'| {noise:.0%} | {method} | {len(means)}/{len(protocol["split_seeds"])} | '
                         f'{runs} | {valid_count} | {mean_text} | {sd_text} | {median_text} | '
                         f'{maximum_text} | {warnings} | {failures} |')
    ax.set(xlabel='Split seed', ylabel='Mean test RMSE')
    ax.set_yscale('symlog', linthresh=1e-3)
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(root / 'seed_means.png', dpi=160)
    plt.close(fig)
    lines += ['', 'A small median with a large mean or maximum indicates severe fit outliers.',
              'Numerical failure counts use the saved run status. Condition warnings are separate.', '', *detail]
    output = root / 'report.md'
    output.write_text('\n'.join(lines) + '\n')
    return output


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('root')
    args = parser.parse_args()
    print(report(args.root))
