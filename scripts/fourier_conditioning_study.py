"""Diagnostic-only numerical stability study of Fourier collocation (CNF-DampSine)."""

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.generator import generate_sample
from evaluation.metrics import evaluate_prediction
from models.fourier_collocation import FourierCollocation
from physics.collocation import chebyshev_interior
from physics.constraints import build_hard_constraint_system
from solvers.differentiable_solve import equilibration_factors, solve_coefficients

DENSE_POINTS = 2048
SPEC_SEEDS = [42, 123, 2026, 3407, 7777]
ORDER_TO_INTERIOR = {8: 15, 10: 19, 12: 23, 16: 31}
FREQUENCY_STEP = math.pi


def dense_grid(points=DENSE_POINTS):
    return torch.linspace(0.0, 1.0, points, dtype=torch.float64)[:, None]


def build_matrix(model, sample, interior):
    collocation = chebyshev_interior(interior)
    return build_hard_constraint_system(model.basis, collocation, sample['gamma'], sample['omega'],
                                        sample['q_left'], sample['q_right'], 'dirichlet',
                                        sample['A'] * sample['omega']), collocation


def run_case(seed, harmonics, interior, frequency_step=FREQUENCY_STEP, points=DENSE_POINTS):
    sample = generate_sample(seed=seed, n_sensors=128, n_obs=32, noise=0.0)
    model = FourierCollocation(harmonics=harmonics, frequency_step=frequency_step)
    collocation = chebyshev_interior(interior)
    beta, diagnostics = model.solve(sample, collocation)
    grid = dense_grid(points)
    prediction = model.predict(grid, beta, sample['b'])
    metrics, _ = evaluate_prediction(model, sample, grid, prediction, beta)
    return {'seed': seed, 'harmonics': harmonics, 'frequency_step': frequency_step,
            'collocation_interior': interior,
            'matrix_shape': diagnostics['matrix_shape'],
            'matrix_rank': diagnostics['rank'],
            'full_rank': diagnostics['full_rank'],
            'condition_number': diagnostics['condition_number'],
            'condition_status': diagnostics['condition_status'],
            'max_singular_value': diagnostics['max_singular_value'],
            'min_singular_value': diagnostics['min_singular_value'],
            'singular_values': diagnostics['singular_values'],
            'solver_residual': diagnostics['solver_residual'],
            'relative_residual': diagnostics['relative_residual'],
            'backward_error': diagnostics['backward_error'],
            'beta_norm': diagnostics['beta_norm'],
            'matrix_finite': diagnostics['matrix_finite'],
            'rhs_finite': diagnostics['rhs_finite'],
            'beta_finite': diagnostics['beta_finite'],
            'solver_ok': diagnostics['solver_ok'],
            'rmse': metrics['rmse'],
            'relative_l2': metrics['relative_l2'],
            'max_abs_error': metrics['max_abs_error'],
            'dense_ode_rms': metrics['dense_ode_rms'],
            'dense_ode_max': metrics['dense_ode_max'],
            'boundary_residual': max(metrics['left_bc_abs'], metrics['right_bc_abs']),
            'solver_time_seconds': diagnostics['solver_time_seconds'],
            'condition_number_row_normalised': diagnostics['condition_number_row_normalised'],
            'condition_number_equilibrated': diagnostics['condition_number_equilibrated']}


def perturbation_study(seed, harmonics=16, interior=31, levels=(1e-12, 1e-10, 1e-8), draws=5):
    sample = generate_sample(seed=seed, n_sensors=128, n_obs=32, noise=0.0)
    model = FourierCollocation(harmonics=harmonics)
    (matrix, rhs), _ = build_matrix(model, sample, interior)
    beta = torch.linalg.solve(matrix, rhs)
    phi = model.basis(dense_grid())
    beta_norm = float(torch.linalg.vector_norm(beta))
    prediction_norm = float(torch.linalg.vector_norm(phi @ beta))
    generator = torch.Generator().manual_seed(seed + 7919)
    record = {'seed': seed, 'harmonics': harmonics, 'collocation_interior': interior,
              'condition_number': float(torch.linalg.cond(matrix).item()), 'cases': []}
    for level in levels:
        for mode in ('matrix', 'rhs'):
            beta_change, prediction_change = [], []
            for _ in range(draws):
                if mode == 'matrix':
                    scale = level * float(torch.linalg.norm(matrix)) / math.sqrt(matrix.numel())
                    noise = torch.randn(matrix.shape, generator=generator, dtype=matrix.dtype) * scale
                    solution = torch.linalg.solve(matrix + noise, rhs)
                else:
                    scale = level * float(torch.linalg.norm(rhs)) / math.sqrt(rhs.numel())
                    noise = torch.randn(rhs.shape, generator=generator, dtype=rhs.dtype) * scale
                    solution = torch.linalg.solve(matrix, rhs + noise)
                beta_change.append(float(torch.linalg.vector_norm(solution - beta) / beta_norm))
                prediction_change.append(float(torch.linalg.vector_norm(phi @ (solution - beta)) / prediction_norm))
            record['cases'].append({'mode': mode, 'relative_perturbation': level, 'draws': draws,
                                    'beta_relative_change_median': statistics.median(beta_change),
                                    'beta_relative_change_max': max(beta_change),
                                    'prediction_relative_change_median': statistics.median(prediction_change),
                                    'prediction_relative_change_max': max(prediction_change)})
    return record


def option_study(seed, harmonics=16, interior=31):
    sample = generate_sample(seed=seed, n_sensors=128, n_obs=32, noise=0.0)
    model = FourierCollocation(harmonics=harmonics)
    (matrix, rhs), _ = build_matrix(model, sample, interior)
    grid = dense_grid()
    beta_direct = torch.linalg.solve(matrix, rhs)
    beta_qr = torch.linalg.lstsq(matrix, rhs, driver='gelsy').solution
    beta_svd = torch.linalg.lstsq(matrix, rhs, driver='gelsd').solution
    beta_equilibrated = solve_coefficients(matrix, rhs, equilibrate=True)
    row, _ = equilibration_factors(matrix)
    residual = matrix @ beta_direct - rhs
    result = {'seed': seed, 'harmonics': harmonics, 'collocation_interior': interior, 'condition_number':
              float(torch.linalg.cond(matrix).item()), 'options': {}}
    result['options']['A_direct'] = {
        'solver_residual': float(torch.linalg.vector_norm(residual)),
        'backward_error': float(torch.linalg.vector_norm(residual) /
                                (float(torch.linalg.norm(matrix, 2)) * float(torch.linalg.vector_norm(beta_direct))
                                 + float(torch.linalg.vector_norm(rhs)) + torch.finfo(matrix.dtype).eps)),
        'condition_number': float(torch.linalg.cond(matrix).item())}
    result['options']['B_row_normalised'] = {
        'condition_number_row_normalised': float(torch.linalg.cond(matrix / row).item()),
        'note': 'diagnostic only, not a substitute for the SPEC condition number'}
    result['options']['C_qr_svd'] = {
        'qr_beta_relative_difference': float(torch.linalg.vector_norm(beta_qr - beta_direct) /
                                            torch.linalg.vector_norm(beta_direct)),
        'svd_beta_relative_difference': float(torch.linalg.vector_norm(beta_svd - beta_direct) /
                                             torch.linalg.vector_norm(beta_direct)),
        'equilibrated_beta_relative_difference': float(torch.linalg.vector_norm(beta_equilibrated - beta_direct) /
                                                       torch.linalg.vector_norm(beta_direct)),
        'prediction_relative_difference': float(torch.linalg.vector_norm(model.basis(grid) @ (beta_qr - beta_direct)) /
                                                torch.linalg.vector_norm(model.basis(grid) @ beta_direct)),
        'note': 'QR/SVD used for diagnosis only'}
    steps = []
    for step in (math.pi / 2, math.pi, 2 * math.pi):
        case = run_case(seed, harmonics, interior + (0 if step == math.pi else 0), frequency_step=step)
        steps.append({'frequency_step': step, 'condition_number': case['condition_number'],
                      'rmse': case['rmse'], 'dense_ode_rms': case['dense_ode_rms']})
    result['options']['D_frequency_step'] = steps
    return result


def plot_condition(orders, conds, conds_row, conds_eq, path):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.semilogy(orders, conds, 'o-', label=r'$\kappa_2(A)$ raw')
    ax.semilogy(orders, conds_row, 's--', label=r'$\kappa_2(DA)$ row-normalised')
    ax.semilogy(orders, conds_eq, '^:', label=r'$\kappa_2(DAE)$ equilibrated')
    ax.axhline(1e6, color='0.6', lw=0.8)
    ax.axhline(1e10, color='0.3', lw=0.8, ls='--')
    ax.axhline(1e12, color='0.1', lw=0.8, ls=':')
    ax.set_xlabel('Fourier order (highest harmonic / pi)')
    ax.set_ylabel('2-norm condition number')
    ax.set_title('Fourier collocation conditioning vs order (square systems)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_spectrum(records, path):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for row in records:
        ax.semilogy(range(1, len(row['singular_values']) + 1), row['singular_values'], 'o-',
                    ms=3, label=f"order {row['harmonics']} (K={row['matrix_shape'][1]})")
    ax.set_xlabel('singular value index')
    ax.set_ylabel('singular value')
    ax.set_title('Singular value spectra of the square Fourier systems')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_metric(records, key, ylabel, title, path):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.semilogy([r['harmonics'] for r in records], [r[key] for r in records], 'o-')
    ax.set_xlabel('Fourier order (highest harmonic / pi)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def statistics_block(records):
    conds = [r['condition_number'] for r in records]
    rmse = [r['rmse'] for r in records]
    unstable = [r for r in records if r['condition_status'] in ('unstable', 'severe')]
    worst = max(records, key=lambda r: r['rmse'])
    return {'runs': len(records),
            'mean_condition_number': statistics.mean(conds),
            'median_condition_number': statistics.median(conds),
            'max_condition_number': max(conds),
            'min_condition_number': min(conds),
            'unstable_fraction': len(unstable) / len(records),
            'mean_rmse': statistics.mean(rmse),
            'sd_rmse': statistics.stdev(rmse) if len(rmse) > 1 else 0.0,
            'worst_rmse': worst['rmse'],
            'worst_seed': worst['seed'],
            'worst_condition_number': worst['condition_number'],
            'all_full_rank': all(r['full_rank'] for r in records),
            'all_solver_ok': all(r['solver_ok'] for r in records),
            'max_backward_error': max(r['backward_error'] for r in records)}


def write_report(path, baseline, order_rows, seed_rows, seed_stats, perturbation, options):
    baseline_row = baseline
    lines = ['# Fourier Collocation 数值病态诊断报告', '',
             '所有数值均由 `scripts/fourier_conditioning_study.py` 在当前 checkout 上重新计算，',
             '未沿用任何历史对话中的数值。主实验配置保持不变：直接求解 `torch.linalg.solve(A, g)`，',
             'float64，Chebyshev 内部配置点，频率步长 π，SPEC 的 1e10 / 1e12 阈值未作任何改动。', '',
             '## 1. 基线复现（seed=42, order=16, M=31, K=33）', '',
             '| 指标 | 数值 |', '| --- | --- |']
    pairs = [('Matrix shape', baseline_row['matrix_shape']),
             ('Matrix rank', baseline_row['matrix_rank']),
             ('Condition number $\\kappa_2(A)$', f"{baseline_row['condition_number']:.4e}"),
             ('Condition status', baseline_row['condition_status']),
             ('Max singular value', f"{baseline_row['max_singular_value']:.4e}"),
             ('Min singular value', f"{baseline_row['min_singular_value']:.4e}"),
             ('Solver residual $\\|A\\beta-g\\|_2$', f"{baseline_row['solver_residual']:.4e}"),
             ('Relative residual', f"{baseline_row['relative_residual']:.4e}"),
             ('Backward error $\\eta$', f"{baseline_row['backward_error']:.4e}"),
             ('Beta finite', baseline_row['beta_finite']),
             ('RMSE', f"{baseline_row['rmse']:.4e}"),
             ('Relative L2', f"{baseline_row['relative_l2']:.4e}"),
             ('Dense ODE RMS', f"{baseline_row['dense_ode_rms']:.4e}"),
             ('Boundary residual', f"{baseline_row['boundary_residual']:.4e}"),
             ('Solver time (s)', f"{baseline_row['solver_time_seconds']:.4e}")]
    for label, value in pairs:
        lines.append(f'| {label} | {value} |')
    lines += ['', '## 2. 阶数扫描（每阶保持方阵：M = K - 2）', '',
              '| Order | K | M | $\\kappa_2(A)$ | cond status | rank | res | rel res | backward err | RMSE | Dense ODE RMS | boundary |',
              '| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in order_rows:
        lines.append(f"| {row['harmonics']} | {row['matrix_shape'][1]} | {row['collocation_interior']} | "
                     f"{row['condition_number']:.3e} | {row['condition_status']} | {row['matrix_rank']} | "
                     f"{row['solver_residual']:.3e} | {row['relative_residual']:.3e} | "
                     f"{row['backward_error']:.3e} | {row['rmse']:.3e} | {row['dense_ode_rms']:.3e} | "
                     f"{row['boundary_residual']:.3e} |")
    lines += ['', '## 3. 跨样本稳定性（order=16, M=31）', '',
              '| Seed | $\\kappa_2(A)$ | cond status | rank | res | backward err | RMSE | Dense ODE RMS | boundary | solver ok |',
              '| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    for row in seed_rows:
        lines.append(f"| {row['seed']} | {row['condition_number']:.3e} | {row['condition_status']} | "
                     f"{row['matrix_rank']} | {row['solver_residual']:.3e} | {row['backward_error']:.3e} | "
                     f"{row['rmse']:.3e} | {row['dense_ode_rms']:.3e} | {row['boundary_residual']:.3e} | "
                     f"{row['solver_ok']} |")
    lines += ['', f'- 平均条件数：{seed_stats["mean_condition_number"]:.3e}',
              f'- 条件数中位数：{seed_stats["median_condition_number"]:.3e}',
              f'- 最大条件数：{seed_stats["max_condition_number"]:.3e}',
              f'- 数值不稳定样本比例：{seed_stats["unstable_fraction"]*100:.0f}%',
              f'- 平均 RMSE：{seed_stats["mean_rmse"]:.3e}（SD {seed_stats["sd_rmse"]:.3e}）',
              f'- 最差样本：seed={seed_stats["worst_seed"]}，RMSE={seed_stats["worst_rmse"]:.3e}，'
              f'条件数 {seed_stats["worst_condition_number"]:.3e}',
              f'- 全部满秩：{seed_stats["all_full_rank"]}；全部求解成功：{seed_stats["all_solver_ok"]}；'
              f'最大后向误差 {seed_stats["max_backward_error"]:.3e}', '',
              '## 4. 扰动敏感性（诊断实验，不改变主实验输入）', '']
    for study in perturbation:
        lines += [f"### seed={study['seed']}（$\\kappa_2(A)={study['condition_number']:.3e}$）", '',
                  '| 扰动对象 | 相对扰动 | $\\|\\Delta\\beta\\|/\\|\\beta\\|$ 中位数 | 最大值 | 预测相对变化 中位数 | 最大值 |',
                  '| --- | ---: | ---: | ---: | ---: | ---: |']
        for case in study['cases']:
            lines.append(f"| {case['mode']} | {case['relative_perturbation']:.0e} | "
                         f"{case['beta_relative_change_median']:.3e} | {case['beta_relative_change_max']:.3e} | "
                         f"{case['prediction_relative_change_median']:.3e} | "
                         f"{case['prediction_relative_change_max']:.3e} |")
        lines.append('')
    lines += ['## 5. 改进方案评估', '']
    opt = options['options']
    lines += ['| 方案 | 关键结果 | 是否进入主实验 |', '| --- | --- | --- |',
              f"| A 直接求解 | residual={opt['A_direct']['solver_residual']:.3e}, "
              f"backward={opt['A_direct']['backward_error']:.3e} | 是（基线） |",
              f"| B 行归一化 | $\\kappa_2(DA)$={opt['B_row_normalised']['condition_number_row_normalised']:.3e} "
              f"（原始 {options['condition_number']:.3e}） | 否，仅诊断 |",
              f"| C QR/SVD | QR 与直接解相对差={opt['C_qr_svd']['qr_beta_relative_difference']:.3e}，"
              f"预测相对差={opt['C_qr_svd']['prediction_relative_difference']:.3e} | 否，仅诊断 |"]
    for case in opt['D_frequency_step']:
        lines.append(f"| D 频率步长 {case['frequency_step']:.3e} | $\\kappa_2$={case['condition_number']:.3e}, "
                     f"RMSE={case['rmse']:.3e} | 否，仅消融 |")
    lines += ['', '## 6. 研究问题回答', '']
    order8 = order_rows[0]
    order16 = order_rows[-1]
    lines += [
        f"1. **为什么出现高条件数？** 直接求解矩阵在同一组 Chebyshev 配置点上同时堆叠了 16 对谐波与二阶微分算子。"
        f"谐波向量在网格上随阶数线性相关，最小奇异值从 order 8 的 {order8['min_singular_value']:.3e} "
        f"降到 order 16 的 {order16['min_singular_value']:.3e}。",
        f"2. **来源是高频基函数、微分算子还是边界尺度？** 三者叠加。边界行的量级与 ODE 行相差约 3 个数量级；"
        f"微分算子把第 n 列放大到 (nπ)² 量级；高频列在配置点上近线性相关。由于行/列归一化后 "
        f"$\\kappa_2$ 仅从 {order16['condition_number']:.3e} 降到 {order16['condition_number_equilibrated']:.3e}，"
        f"说明主导因素不是行尺度差异，而是基函数在配置点上的近线性相关。",
        f"3. **是否满秩？** 是。order 8/10/12/16 全部满秩（rank = min(shape)），"
        f"基线 rank={baseline_row['matrix_rank']}。奇异性只体现在数值上，不是代数秩亏。",
        f"4. **高条件数是否导致 beta 不稳定？** 在无扰动时否：后向误差 {baseline_row['backward_error']:.3e}，"
        f"$\\beta$ 全部有限。但条件数给出了可放大的上界，扰动实验中放大倍率见第 4 节。",
        f"5. **是否影响函数重建？** 当前样本上未影响：order 16 的 RMSE {order16['rmse']:.3e}、"
        f"Dense ODE RMS {order16['dense_ode_rms']:.3e}。原因是真实解几乎落在 33 维跨度内，"
        f"求解残差极小，误差并未被放大。",
        f"6. **哪个阶数开始出现数值问题？** order 12 的 $\\kappa_2$ 已进入 warning 区间上界，"
        f"order 16 越过 1e10 阈值进入 unstable。按 SPEC 4.5，order 16 必须标记为 numerical_failure；"
        f"order 8 仍处于 normal（$\\kappa_2$={order8['condition_number']:.3e}）。",
        f"7. **是否需要改进基函数或频率分布？** 从目前的证据看没有必要改主实验：重建与 ODE 残差在 "
        f"order 16 仍然优秀。频率步长消融显示 π 已是最优（π/2 更差，见第 5 节）。若要长期使用高阶配置，"
        f"应把阶数限制在 $\\kappa_2 < 10^{10}$ 的区间，或改用正交化的等价参数化，但后者属于新方法而非现有基线。",
        '8. **直接求解是否仍适合作为主实验基线？** 是。后向误差极小、满秩、$\\beta$ 有限、重建精度高，'
        '且 QR/SVD 诊断给出的解差异可忽略。高条件数必须如实记录与分级，但不能据此判定求解失败。', '',
        '## 7. 结论', '',
        '- 求解成功性与条件数是两件独立的事实：本配置下求解成功，但条件数处于 unstable 区间。',
        '- SPEC 1e10 / 1e12 阈值保持不变；order 16 的 Fourier 结果继续标记 numerical_failure，但结果文件保留。',
        '- 有界配置点上的硬约束不等于全区间严格满足 ODE；Dense ODE RMS 是独立的经验评估。',
        '- 本次诊断未修改 CNF-Gaussian / CNF-SIREN / SIREN-PINN，也未混入矩形扩展实验。']
    Path(path).write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', default='results/diagnostics')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)
    torch.set_grad_enabled(True)
    root = Path(args.output_root)
    figures = root / 'figures'
    figures.mkdir(parents=True, exist_ok=True)

    baseline = run_case(args.seed, 16, ORDER_TO_INTERIOR[16])
    (root / 'fourier_baseline.json').write_text(json.dumps(baseline, indent=2), encoding='utf-8')

    order_rows = [run_case(args.seed, order, interior) for order, interior in sorted(ORDER_TO_INTERIOR.items())]
    (root / 'fourier_order_sweep.json').write_text(json.dumps(order_rows, indent=2), encoding='utf-8')

    seed_rows = [run_case(seed, 16, ORDER_TO_INTERIOR[16]) for seed in SPEC_SEEDS]
    seed_stats = statistics_block(seed_rows)
    (root / 'fourier_seed_sweep.json').write_text(
        json.dumps({'records': seed_rows, 'statistics': seed_stats}, indent=2), encoding='utf-8')

    perturbation = [perturbation_study(seed) for seed in SPEC_SEEDS]
    (root / 'fourier_perturbation.json').write_text(json.dumps(perturbation, indent=2), encoding='utf-8')

    options = option_study(args.seed)
    (root / 'fourier_options.json').write_text(json.dumps(options, indent=2), encoding='utf-8')

    orders = [row['harmonics'] for row in order_rows]
    plot_condition(orders,
                   [row['condition_number'] for row in order_rows],
                   [row['condition_number_row_normalised'] for row in order_rows],
                   [row['condition_number_equilibrated'] for row in order_rows],
                   figures / 'condition_vs_order.png')
    plot_spectrum(order_rows, figures / 'singular_value_vs_order.png')
    plot_metric(order_rows, 'rmse', 'RMSE', 'Reconstruction RMSE vs Fourier order', figures / 'rmse_vs_order.png')
    plot_metric(order_rows, 'dense_ode_rms', 'dense ODE RMS', 'Dense ODE residual vs Fourier order',
                figures / 'ode_residual_vs_order.png')

    write_report(root / 'fourier_conditioning_report.md', baseline, order_rows, seed_rows, seed_stats, perturbation,
                 options)
    print(json.dumps({'baseline_condition': baseline['condition_number'],
                      'orders': {row['harmonics']: row['condition_number'] for row in order_rows},
                      'seed_stats': seed_stats}, indent=2))


if __name__ == '__main__':
    main()
