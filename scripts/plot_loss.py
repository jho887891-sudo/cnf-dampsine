"""Plot saved per-step training loss without rerunning an experiment."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_log(path):
    path = Path(path)
    rows = [json.loads(line) for line in path.read_text().splitlines()
            if line.lstrip().startswith('{')]
    if not rows:
        return None  # Direct solvers have no optimizer or loss trajectory.
    steps = [row['step'] for row in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    for key in ('total', 'data', 'ode', 'bc'):
        if all(key in row for row in rows):
            ax.plot(steps, [row[key] for row in rows], label=key, alpha=.85)
    ax.set(xlabel='Training step', ylabel='Loss', title=path.parent.name)
    ax.set_yscale('symlog', linthresh=1e-8)
    ax.legend()
    ax.grid(alpha=.2)
    fig.tight_layout()
    output = path.with_name('loss.png')
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root')
    args = parser.parse_args()
    for log in sorted(Path(args.root).rglob('train.log')):
        output = plot_log(log)
        if output is not None:
            print(output)
