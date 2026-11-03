"""Diagnostic figures generated only from saved predictions and summaries."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml


def plot_run(run_dir):
    run_dir = Path(run_dir)
    arrays = np.load(run_dir / 'predictions.npz')
    x = arrays['x'].ravel()
    truth = arrays['y_true'].ravel()
    prediction = arrays['y_pred'].ravel()
    residual = np.abs(arrays['dense_ode_residual'].ravel())
    observations_x = arrays['x_obs'].ravel()
    observations_y = arrays['y_obs'].ravel()
    figures = {}
    def save(key, draw):
        fig, ax = plt.subplots(figsize=(7, 4))
        draw(ax)
        ax.set_xlabel('x')
        fig.tight_layout()
        path = run_dir / f'{key}.png'
        fig.savefig(path, dpi=160)
        plt.close(fig)
        figures[key] = str(path)
    save('reconstruction', lambda ax: (ax.plot(x, truth, label='truth'),
        ax.plot(x, prediction, label='prediction'), ax.scatter(observations_x, observations_y, s=8, label='observations'), ax.legend()))
    save('absolute_error', lambda ax: ax.plot(x, np.abs(prediction-truth)))
    save('dense_ode_residual', lambda ax: ax.semilogy(x, np.maximum(residual, 1e-16)))
    save('left_boundary', lambda ax: (ax.plot(x[:100], truth[:100]), ax.plot(x[:100], prediction[:100])))
    save('right_boundary', lambda ax: (ax.plot(x[-100:], truth[-100:]), ax.plot(x[-100:], prediction[-100:])))
    beta_path = run_dir / 'beta.npy'
    if beta_path.exists():
        beta = np.load(beta_path).ravel()
        save('beta_coefficients', lambda ax: ax.bar(np.arange(len(beta)), beta))
    summary = json.loads((run_dir / 'summary.json').read_text(encoding='utf-8'))
    singular = summary.get('singular_values')
    if singular:
        save('singular_values', lambda ax: ax.semilogy(np.arange(len(singular)), singular, marker='.'))
    return figures


def plot_comparison(results_root):
    root = Path(results_root)
    paths = sorted(root.glob('*/predictions.npz'))
    if not paths:
        raise ValueError('no predictions found for comparison')
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, path in enumerate(paths):
        arrays = np.load(path)
        x = arrays['x'].ravel()
        if i == 0:
            ax.plot(x, arrays['y_true'].ravel(), color='black', linewidth=2, label='truth')
            ax.scatter(arrays['x_obs'].ravel(), arrays['y_obs'].ravel(), s=7, color='black', alpha=.5, label='observations')
        ax.plot(x, arrays['y_pred'].ravel(), label=path.parent.name, alpha=.8)
    ax.set_xlabel('x')
    ax.legend(fontsize=8)
    fig.tight_layout()
    output = root / 'four_method_comparison.png'
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def plot_sweep(results_root, x_key, y_key, output_name=None):
    root = Path(results_root)
    series = {}
    for summary_path in root.rglob('summary.json'):
        row = json.loads(summary_path.read_text(encoding='utf-8'))
        if y_key not in row or row[y_key] is None:
            continue
        source = yaml.safe_load((summary_path.parent / 'config.yaml').read_text(encoding='utf-8')).get('source', {})
        if x_key in row:
            x_value = row[x_key]
        elif x_key in source:
            x_value = source[x_key]
        else:
            x_value = source.get('methods', {}).get(row['method'], {}).get(x_key)
        if not isinstance(x_value, (float, int)):
            continue
        label = row['method'] + (' failure' if row.get('status') != 'ok' else '')
        series.setdefault(label, []).append((x_value, row[y_key]))
    if not series:
        raise ValueError(f'no paired {x_key}/{y_key} values')
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, points in sorted(series.items()):
        points.sort()
        ax.plot([p[0] for p in points], [p[1] for p in points], marker='o', label=label)
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    if y_key in ('condition_number', 'dense_ode_rms'):
        ax.set_yscale('log')
    ax.legend(fontsize=8)
    fig.tight_layout()
    output = root / (output_name or f'{y_key}_vs_{x_key}.png')
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('results_root')
    parser.add_argument('--x')
    parser.add_argument('--y')
    args = parser.parse_args()
    if args.x and args.y:
        print(plot_sweep(args.results_root, args.x, args.y))
        raise SystemExit(0)
    for path in Path(args.results_root).rglob('predictions.npz'):
        print(plot_run(path.parent))
    if len(list(Path(args.results_root).glob('*/predictions.npz'))) > 1:
        print(plot_comparison(args.results_root))
