"""Render the SPEC diagnostic figures from archived run data."""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


def records(root):
    names = ("p1_capacity_cnf", "p1_rectangular_gaussian", "p1_rectangular_siren", "p1_sigma_gaussian", "p1_omega")
    for name in names:
      for path in (root / "results" / name).rglob("summary.json"):
          config_path = path.with_name("config.yaml")
          if not config_path.exists():
              continue
          row = json.loads(path.read_text())
          source = yaml.safe_load(config_path.read_text()).get("source", {})
          row["_source"] = source
          yield path, row


def chart(lines, output, xlabel, ylabel, log=False):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, pairs in sorted(lines.items()):
        if not pairs:
            continue
        pairs.sort()
        ax.plot(*zip(*pairs), marker="o", label=label)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    if log:
        ax.set_yscale("log")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main(root):
    root = root.resolve()
    reports = root / "reports"
    rows = list(records(root))
    capacity = [(p, r) for p, r in rows if "p1_capacity_cnf" in p.parts]
    rectangular = [(p, r) for p, r in rows if "p1_rectangular_gaussian" in p.parts or "p1_rectangular_siren" in p.parts]
    for name, group, xkey, ykey, log in [
        ("condition-vs-k", capacity, "K", "condition_number", True),
        ("dense-residual-vs-k", capacity, "K", "dense_ode_rms", True),
        ("condition-vs-m", rectangular, "M", "condition_number", True),
    ]:
        lines = defaultdict(list)
        for _, row in group:
            if row.get(xkey) is not None and row.get(ykey) is not None:
                label = row["method"] + (" (infeasible)" if row.get("hard_feasible") is False else "")
                lines[label].append((row[xkey], row[ykey]))
        chart(lines, reports / f"{name}.png", xkey, ykey, log)

    for dirname, setting, filename in [
        ("p1_sigma_gaussian", "sigma", "gaussian-sigma-sensitivity.png"),
        ("p1_omega", "omega0", "siren-omega-sensitivity.png"),
    ]:
        lines = defaultdict(list)
        for path, row in rows:
            if dirname not in path.parts or row.get("rmse") is None:
                continue
            method = row["method"]
            value = row["_source"].get("methods", {}).get(method, {}).get(setting)
            if isinstance(value, (int, float)):
                lines[method + (" (failure)" if row.get("status") != "ok" else "")].append((value, row["rmse"]))
        chart(lines, reports / filename, setting, "dense RMSE", True)

    betas = []
    for method in ("cnf_gaussian", "cnf_siren", "fourier"):
        path = root / "results" / "p0_full" / method / "beta.npy"
        if path.exists():
            betas.append((method, np.load(path).ravel()))
    if betas:
        n = max(len(a) for _, a in betas)
        matrix = np.full((len(betas), n), np.nan)
        for i, (_, a) in enumerate(betas):
            matrix[i, :len(a)] = a
        fig, ax = plt.subplots(figsize=(8, 2.8))
        image = ax.imshow(matrix, aspect="auto", cmap="coolwarm")
        ax.set(yticks=range(len(betas)), yticklabels=[name for name, _ in betas], xlabel="basis coefficient index", title="P0 solved beta coefficients")
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        fig.savefig(reports / "p0-beta-heatmap.png", dpi=170)
        plt.close(fig)

    # The Fourier dictionary is fixed, so its archived P0 matrix can be
    # reconstructed exactly from the saved physics parameters and grid rule.
    import sys
    sys.path.insert(0, str(root))
    import torch
    from models.bases.fourier_basis import FourierBasis
    from physics.collocation import chebyshev_interior
    from physics.constraints import build_hard_constraint_system

    fourier = json.loads((root / "results/p0_full/fourier/summary.json").read_text())
    params = fourier["physics_parameters"]
    x_interior = chebyshev_interior(fourier["M"])
    q_left = 0.0
    q_right = params["A"] * np.exp(-params["gamma"]) * np.sin(params["omega"])
    matrix, _ = build_hard_constraint_system(
        FourierBasis(16), x_interior, params["gamma"], params["omega"],
        q_left, q_right)
    matrix = matrix.detach().numpy()
    fig, ax = plt.subplots(figsize=(7, 5))
    image = ax.imshow(np.log10(np.maximum(np.abs(matrix), 1e-16)),
                      aspect="auto", cmap="viridis")
    ax.set(xlabel="basis column", ylabel="constraint row",
           title="P0 Fourier log10|A| (numerically unstable solve)")
    fig.colorbar(image, ax=ax, label="log10 absolute matrix entry")
    fig.tight_layout()
    fig.savefig(reports / "p0-fourier-matrix-heatmap.png", dpi=170)
    plt.close(fig)

    lines = defaultdict(list)
    for filename in ("p1-clean-formal-all.md", "p2-formal-multiseed.md", "p2-pinn-formal-multiseed.md"):
        for line in (reports / filename).read_text().splitlines():
            fields = [field.strip() for field in line.strip().strip("|").split("|")]
            if len(fields) > 6 and fields[0] in ("0%", "1%", "5%"):
                lines[fields[1]].append((float(fields[0][:-1]), float(fields[5])))
    chart(lines, reports / "noise-degradation.png", "noise (%)", "mean dense RMSE")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method in ("cnf_gaussian", "cnf_siren", "siren_pinn"):
        path = root / "results" / "p0_full" / method / "summary.json"
        row = json.loads(path.read_text())
        if row.get("status") == "ok":
            ax.scatter(row["total_time_seconds"], row["relative_l2"], label=method)
    ax.set(xlabel="total run time (s; P0 single curve)", ylabel="relative L2", title="P0 time vs relative L2 (descriptive, not an optimized Pareto frontier)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(reports / "p0-time-error.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root", type=Path)
    main(parser.parse_args().project_root)
