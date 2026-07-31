"""HiFi-ANOVA plotting — regularization-path, lambda and model-selection diagnostics.

Split from the original monolithic ``plots.py``; import via
``from hifi_anova.analysis.plots import ...`` as before.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple

from ._common import PALETTE, apply_style, _ensure_np, _var_color


# ============================================================================
# 1. Regularization path 4-panel
# ============================================================================

def plot_reg_path_panel(
    path,  # RegPathResult
    variable_names: Optional[List[str]] = None,
    show_top_n: int = 8,
    figsize: Tuple[float, float] = (12, 9),
) -> Tuple[plt.Figure, np.ndarray]:
    """Publication-quality 4-panel regularization path.

    Panels: L-curve | GCV/AIC/BIC | Sobol paths | Variance decomposition.
    GCV/BIC optima marked throughout.
    """
    apply_style()
    D = len(path.sobol_paths)
    if variable_names is None:
        variable_names = [f"$x_{{{i+1}}}$" for i in range(D)]

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    lambdas = _ensure_np(path.lambdas)
    gcv_idx = int(np.argmin(np.abs(lambdas - path.lambda_gcv_opt)))

    # BIC optimal
    bic_vals = _ensure_np(path.bic_values)
    bic_idx = int(np.argmin(bic_vals))

    # --- Panel A: L-curve ---
    ax = axes[0, 0]
    df = _ensure_np(path.df_values)
    mse = _ensure_np(path.mse_values)
    ax.loglog(df, mse, '-', color=PALETTE['order1'], linewidth=1.5, zorder=2)
    ax.plot(df[gcv_idx], mse[gcv_idx], '*', color=PALETTE['highlight'],
            markersize=14, zorder=3, label=f'GCV opt (df={df[gcv_idx]:.0f})')
    ax.plot(df[bic_idx], mse[bic_idx], 'D', color=PALETTE['bic'],
            markersize=8, zorder=3, label=f'BIC opt (df={df[bic_idx]:.0f})')
    ax.set_xlabel('Effective degrees of freedom')
    ax.set_ylabel('Training MSE')
    ax.set_title('(a) L-curve', loc='left', fontweight='bold')
    ax.legend(loc='best')

    # --- Panel B: Model selection criteria ---
    ax = axes[0, 1]
    gcv = _ensure_np(path.gcv_values)
    aic = _ensure_np(path.aic_values)

    # Normalise criteria to [0, 1] range for visual comparison
    def _norm(v):
        mn, mx = v.min(), v.max()
        return (v - mn) / (mx - mn) if mx > mn else np.zeros_like(v)

    ax.semilogx(lambdas, _norm(gcv), '-', color=PALETTE['gcv'],
                linewidth=1.5, label='GCV')
    ax.semilogx(lambdas, _norm(aic), '-', color=PALETTE['aic'],
                linewidth=1.5, label='AIC')
    ax.semilogx(lambdas, _norm(bic_vals), '-', color=PALETTE['bic'],
                linewidth=1.5, label='BIC')

    ax.axvline(lambdas[gcv_idx], color=PALETTE['gcv'], ls='--', alpha=0.4)
    ax.axvline(lambdas[bic_idx], color=PALETTE['bic'], ls='--', alpha=0.4)
    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel('Normalised criterion')
    ax.set_title('(b) Model selection', loc='left', fontweight='bold')
    ax.legend(loc='best')

    # --- Panel C: Sobol index paths ---
    ax = axes[1, 0]
    max_sobol = {i: float(np.max(_ensure_np(path.sobol_paths[i])))
                 for i in range(D)}
    top_vars = sorted(max_sobol, key=lambda i: -max_sobol[i])[:show_top_n]

    for rank, i in enumerate(top_vars):
        ax.semilogx(lambdas, _ensure_np(path.sobol_paths[i]), '-',
                    color=_var_color(rank), linewidth=1.5,
                    label=variable_names[i])

    # Top 2nd-order interactions
    if path.sobol_paths_2nd:
        max_2nd = {k: float(np.max(_ensure_np(v)))
                   for k, v in path.sobol_paths_2nd.items()}
        top_pairs = sorted(max_2nd, key=lambda k: -max_2nd[k])[:3]
        for (i, j) in top_pairs:
            if max_2nd[(i, j)] > 0.01:
                ax.semilogx(lambdas, _ensure_np(path.sobol_paths_2nd[(i, j)]),
                           '--', color=PALETTE['muted'], linewidth=1.2,
                           label=f'({variable_names[i]},{variable_names[j]})')

    ax.axvline(lambdas[gcv_idx], color='grey', ls=':', alpha=0.5)
    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel('Sobol index $S_i$')
    ax.set_title('(c) Sensitivity paths', loc='left', fontweight='bold')
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper right', fontsize=7, ncol=2)

    # --- Panel D: Variance decomposition stacked area ---
    ax = axes[1, 1]
    o1 = _ensure_np(path.var_order1)
    o2 = _ensure_np(path.var_order2)
    o3 = _ensure_np(path.var_order3)
    res = _ensure_np(path.var_residual)

    ax.fill_between(lambdas, 0, o1, alpha=0.7,
                    color=PALETTE['order1'], label='1st order')
    ax.fill_between(lambdas, o1, o1 + o2, alpha=0.7,
                    color=PALETTE['order2'], label='2nd order')
    if np.any(o3 > 0):
        ax.fill_between(lambdas, o1 + o2, o1 + o2 + o3, alpha=0.7,
                        color=PALETTE['order3'], label='3rd order')
    if np.any(res > 0):
        ax.fill_between(lambdas, o1 + o2 + o3, o1 + o2 + o3 + res, alpha=0.5,
                        color=PALETTE['residual'], label='Residual')

    ax.axvline(lambdas[gcv_idx], color='grey', ls=':', alpha=0.5)
    ax.set_xscale('log')
    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel('Explained variance')
    ax.set_title('(d) Variance decomposition', loc='left', fontweight='bold')
    ax.legend(loc='best')

    fig.tight_layout()
    return fig, axes


# ============================================================================
# 21. Regularization path: Sobol + variance decomposition dual-axis
# ============================================================================

def plot_sobol_variance_path(
    path,  # RegPathResult
    variable_names: Optional[List[str]] = None,
    show_top_n: int = 6,
    figsize: Tuple[float, float] = (12, 5),
) -> Tuple[plt.Figure, np.ndarray]:
    """Two-panel: Sobol paths with ribbon + variance order decomposition.

    Left: top-N Sobol paths with shaded ribbon.
    Right: stacked area of variance by interaction order, with GCV/BIC marks.
    """
    apply_style()
    D = len(path.sobol_paths)
    if variable_names is None:
        variable_names = [f"$x_{{{i+1}}}$" for i in range(D)]
    lambdas = _ensure_np(path.lambdas)
    gcv_idx = int(np.argmin(np.abs(lambdas - path.lambda_gcv_opt)))
    bic_idx = int(np.argmin(_ensure_np(path.bic_values)))

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # --- Left: Sobol paths ---
    ax = axes[0]
    max_sobol = {i: float(np.max(_ensure_np(path.sobol_paths[i])))
                 for i in range(D)}
    top_vars = sorted(max_sobol, key=lambda i: -max_sobol[i])[:show_top_n]

    for rank, i in enumerate(top_vars):
        sp = _ensure_np(path.sobol_paths[i])
        c = _var_color(rank)
        ax.semilogx(lambdas, sp, '-', color=c, linewidth=2,
                    label=variable_names[i])
        # Approximate uncertainty ribbon
        ax.fill_between(lambdas, sp * 0.9, sp * 1.1, alpha=0.08, color=c)

    # Top 2nd order
    if path.sobol_paths_2nd:
        max_2nd = {k: float(np.max(_ensure_np(v)))
                   for k, v in path.sobol_paths_2nd.items()}
        for (i, j) in sorted(max_2nd, key=lambda k: -max_2nd[k])[:2]:
            if max_2nd[(i, j)] > 0.01:
                ax.semilogx(lambdas,
                           _ensure_np(path.sobol_paths_2nd[(i, j)]),
                           '--', color=PALETTE['muted'], linewidth=1.5,
                           label=f'({variable_names[i]},{variable_names[j]})')

    ax.axvline(lambdas[gcv_idx], color=PALETTE['gcv'], ls=':', alpha=0.5,
               label='GCV opt')
    ax.axvline(lambdas[bic_idx], color=PALETTE['bic'], ls=':', alpha=0.5,
               label='BIC opt')
    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel('Sobol index $S_i$')
    ax.set_title('(a) Sensitivity paths', loc='left', fontweight='bold')
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper right', fontsize=7, ncol=2)

    # --- Right: variance decomposition ---
    ax = axes[1]
    o1 = _ensure_np(path.var_order1)
    o2 = _ensure_np(path.var_order2)
    o3 = _ensure_np(path.var_order3)
    res = _ensure_np(path.var_residual)
    vtot = o1 + o2 + o3 + res
    vtot[vtot < 1e-15] = 1.0

    ax.fill_between(lambdas, 0, o1 / vtot, alpha=0.7,
                    color=PALETTE['order1'], label='1st order')
    ax.fill_between(lambdas, o1 / vtot, (o1 + o2) / vtot, alpha=0.7,
                    color=PALETTE['order2'], label='2nd order')
    if np.any(o3 > 0):
        ax.fill_between(lambdas, (o1 + o2) / vtot,
                        (o1 + o2 + o3) / vtot, alpha=0.7,
                        color=PALETTE['order3'], label='3rd order')
    if np.any(res > 0):
        ax.fill_between(lambdas, (o1 + o2 + o3) / vtot, 1.0, alpha=0.5,
                        color=PALETTE['residual'], label='Residual')

    ax.axvline(lambdas[gcv_idx], color=PALETTE['gcv'], ls=':', alpha=0.5)
    ax.axvline(lambdas[bic_idx], color=PALETTE['bic'], ls=':', alpha=0.5)
    ax.set_xscale('log')
    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel('Fraction of explained variance')
    ax.set_title('(b) Variance order decomposition', loc='left',
                 fontweight='bold')
    ax.set_ylim(0, 1.02)
    ax.legend(loc='best', fontsize=8)

    fig.tight_layout()
    return fig, axes


# ============================================================================
# 31. Lambda triptych: complexity + noise + Sobol on shared x-axis
# ============================================================================

def plot_lambda_triptych(
    nc: Dict,
    sobol_paths: Dict[int, np.ndarray],
    lambdas: np.ndarray,
    variable_names: Optional[List[str]] = None,
    true_sigma: Optional[float] = None,
    ground_truth: Optional[Dict[int, float]] = None,
    figsize: Tuple[float, float] = (10, 10),
) -> Tuple[plt.Figure, np.ndarray]:
    """Three vertically stacked panels sharing the lambda x-axis.

    (a) Effective df — model complexity vs lambda.
    (b) Noise estimate sigma^2(lambda) — U-curve with minimum marked.
    (c) Sobol paths — sensitivity indices stabilising at the sweet spot.

    The vertical line at sigma^2 minimum shows where all three agree.

    Args:
        nc: output of noise_complexity_curve().
        sobol_paths: {var_i: (n_lambda,)} Sobol indices per variable.
        lambdas: (n_lambda,) the lambda grid (same as nc['lambdas']).
        variable_names: display names.
        true_sigma: if known, mark the true noise level.
        ground_truth: {var_i: true_sobol} for dotted reference lines.
    """
    apply_style()
    D = len(sobol_paths)
    if variable_names is None:
        variable_names = [f"$x_{{{i+1}}}$" for i in range(D)]

    nc_lam = _ensure_np(nc['lambdas'])
    nc_sig = _ensure_np(nc['sigma2'])
    nc_df = _ensure_np(nc['df'])
    sig_min_idx = int(np.argmin(nc_sig))
    col_opt = '#e74c3c'

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    # Panel (a): Effective df
    ax = axes[0]
    ax.semilogx(nc_lam, nc_df, color='#2c3e50', linewidth=2)
    ax.axvline(nc_lam[sig_min_idx], ls='--', color=col_opt, alpha=0.6)
    ax.fill_betweenx([0, max(nc_df) * 1.05], nc_lam[0], nc_lam[sig_min_idx],
                     alpha=0.04, color='blue')
    ax.fill_betweenx([0, max(nc_df) * 1.05], nc_lam[sig_min_idx], nc_lam[-1],
                     alpha=0.04, color='orange')
    ax.text(nc_lam[2], max(nc_df) * 0.9, 'complex', fontsize=9,
            color='blue', alpha=0.5)
    ax.text(nc_lam[-3], max(nc_df) * 0.9, 'simple', fontsize=9,
            color='orange', alpha=0.5, ha='right')
    ax.set_ylabel('Effective df')
    ax.set_title('(a) Model complexity', loc='left', fontweight='bold')

    # Panel (b): Noise estimate
    ax = axes[1]
    ax.semilogx(nc_lam, nc_sig, color='#27ae60', linewidth=2.5)
    ax.axvline(nc_lam[sig_min_idx], ls='--', color=col_opt, alpha=0.6)
    ax.plot(nc_lam[sig_min_idx], nc_sig[sig_min_idx], 'o', color=col_opt,
            markersize=10, zorder=5)
    if true_sigma is not None:
        ax.axhline(true_sigma ** 2, ls='--', color='red', alpha=0.5,
                   linewidth=1, label=f'True $\\sigma^2={true_sigma**2:.2f}$')
        ax.legend(fontsize=8)
    ax.set_ylabel(r'$\hat\sigma^2(\lambda)$')
    sig_min_val = nc_sig[sig_min_idx]
    ax.set_title(f'(b) Noise estimate (min $\\hat{{\\sigma}}='
                 f'{np.sqrt(sig_min_val):.3f}$)', loc='left', fontweight='bold')
    ax.annotate('overfitting\n(noise hidden)', xy=(nc_lam[2], nc_sig[2]),
                fontsize=8, color='gray')
    ax.annotate('underfitting\n(signal as noise)',
                xy=(nc_lam[-3], nc_sig[-3]), fontsize=8, color='gray',
                ha='right')

    # Panel (c): Sobol paths
    ax = axes[2]
    # Show top variables by max Sobol
    max_s = {i: float(np.max(_ensure_np(sobol_paths[i])))
             for i in sobol_paths}
    top = sorted(max_s, key=lambda i: -max_s[i])[:8]
    for rank, i in enumerate(top):
        sp = _ensure_np(sobol_paths[i])
        c = _var_color(rank)
        label = variable_names[i] if i < len(variable_names) else f'x{i+1}'
        if ground_truth and i in ground_truth:
            label += f' (true {ground_truth[i]:.3f})'
            ax.axhline(ground_truth[i], ls=':', color=c, alpha=0.2)
        ax.semilogx(lambdas, sp, color=c, linewidth=2, label=label)
    ax.axvline(nc_lam[sig_min_idx], ls='--', color=col_opt, alpha=0.6,
               label='$\\hat{\\sigma}^2$ minimum')
    ax.set_ylabel('Sobol $S_i$')
    ax.set_xlabel(r'$\lambda$')
    ax.set_title('(c) Sensitivity indices', loc='left', fontweight='bold')
    ax.legend(fontsize=7, ncol=3, loc='upper right')

    fig.suptitle('Lambda tradeoff: complexity, noise, and sensitivity',
                 fontsize=13, y=1.01)
    fig.tight_layout()
    return fig, axes


# ============================================================================
# 32. Stability fan: full-data vs per-fold Sobol spread
# ============================================================================

def plot_stability_fan(
    stability: Dict,
    variable_names: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (10, 5),
) -> Tuple[plt.Figure, plt.Axes]:
    """Full-data Sobol bars with per-fold translucent bars overlaid.

    Shows visually where the decomposition is robust vs uncertain.
    Red 95% CI error bars from the fold spread.

    Args:
        stability: output of stability_diagnostics().
        variable_names: display names.
    """
    apply_style()
    D = len(stability['full_data']['sobol'])
    if variable_names is None:
        variable_names = [f"$x_{{{i+1}}}$" for i in range(D)]

    fig, ax = plt.subplots(figsize=figsize)
    x_pos = np.arange(D)

    full_sobols = [stability['full_data']['sobol'].get(i, 0) for i in range(D)]
    ax.bar(x_pos, full_sobols, 0.6, color=PALETTE['order1'], alpha=0.9,
           edgecolor='0.3', linewidth=0.5, label='Full data', zorder=3)

    fold_colors = plt.cm.Set2(np.linspace(0, 1, len(stability['per_fold'])))
    for k, fold in enumerate(stability['per_fold']):
        fold_sobols = [fold['sobol'].get(i, 0) for i in range(D)]
        ax.bar(x_pos + (k - 2) * 0.08, fold_sobols, 0.08,
               color=fold_colors[k], alpha=0.5, edgecolor='none', zorder=2)

    means = [stability['sobol_mean'][i] for i in range(D)]
    stds = [stability['sobol_std'][i] for i in range(D)]
    ax.errorbar(x_pos, means, yerr=[s * 1.96 for s in stds], fmt='none',
                ecolor=PALETTE['residual'], elinewidth=1.5, capsize=4,
                capthick=1.5, zorder=4, label='95% CI (fold spread)')

    ax.set_xticks(x_pos)
    ax.set_xticklabels([n[:8] for n in variable_names], rotation=45,
                       ha='right', fontsize=9)
    ax.set_ylabel('First-order Sobol $S_i$')
    stab = stability.get('stability', '?')
    ax.set_title(f'Sobol stability: full data vs {len(stability["per_fold"])}'
                 f'-fold spread (stability: {stab})')
    ax.legend(fontsize=8)

    ax.text(0.98, 0.98,
            f'$\\hat{{\\sigma}}_{{full}}={stability["full_data"]["sigma_hat"]:.3f}$\n'
            f'$\\hat{{\\sigma}}_{{fold}}={stability["sigma_mean"]:.3f}'
            f'\\pm{stability["sigma_std"]:.3f}$\n'
            f'LOO-CV={stability["loo_cv"]:.4f}\n'
            f'{len(stability["per_fold"])}-fold CV={stability["kfold_cv"]:.4f}',
            transform=ax.transAxes, ha='right', va='top', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.tight_layout()
    return fig, ax


# ============================================================================
# 33. Strategy parallel coordinates
# ============================================================================

def plot_strategy_parallel(
    strategy_sobol: Dict[str, List[float]],
    strategy_rmse: Dict[str, float],
    variable_names: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (12, 5.5),
) -> Tuple[plt.Figure, plt.Axes]:
    """Parallel coordinates: one line per variable across regularization strategies.

    Lines that stay flat are robust attributions; lines that swing are
    strategy-dependent. RMSE annotated below each strategy.

    Args:
        strategy_sobol: {strategy_name: [S_0, S_1, ..., S_D]}.
        strategy_rmse: {strategy_name: rmse}.
        variable_names: display names.
    """
    apply_style()
    strategies = list(strategy_sobol.keys())
    D = len(strategy_sobol[strategies[0]])
    if variable_names is None:
        variable_names = [f"$x_{{{i+1}}}$" for i in range(D)]

    fig, ax = plt.subplots(figsize=figsize)
    x_pos = np.arange(len(strategies))
    colors = plt.cm.tab10(np.linspace(0, 1, D))

    for i in range(D):
        vals = [strategy_sobol[s][i] for s in strategies]
        lw = 2.5 if max(vals) > 0.05 else 0.8
        alpha = 1.0 if max(vals) > 0.05 else 0.3
        ax.plot(x_pos, vals, 'o-', color=colors[i], linewidth=lw,
                markersize=6, alpha=alpha,
                label=variable_names[i] if max(vals) > 0.02 else None)

    for j, strat in enumerate(strategies):
        ax.text(j, -0.015, f'RMSE\n{strategy_rmse[strat]:.3f}', ha='center',
                fontsize=7, color='gray', transform=ax.get_xaxis_transform())

    ax.set_xticks(x_pos)
    ax.set_xticklabels(strategies, fontsize=10)
    ax.set_ylabel('First-order Sobol $S_i$')
    ax.set_title('Strategy affects attribution, not prediction')
    ax.legend(fontsize=8, ncol=2, loc='upper right')
    ax.set_ylim(-0.02, None)

    fig.tight_layout()
    return fig, ax


# ============================================================================
# 34. Complexity staircase: RMSE/sigma stepping down, Sobol invariant
# ============================================================================

def plot_complexity_staircase(
    steps: List[Dict],
    sobol_per_step: List[List[float]],
    variable_names: Optional[List[str]] = None,
    true_sigma: Optional[float] = None,
    ground_truth: Optional[List[float]] = None,
    figsize: Tuple[float, float] = (13, 5),
) -> Tuple[plt.Figure, np.ndarray]:
    """Two-panel: RMSE/sigma staircase + Sobol invariance across model orders.

    Left: paired bars (RMSE solid, sigma hatched) stepping down.
    Right: per-variable Sobol lines staying flat as complexity grows.

    Args:
        steps: [{'label': str, 'rmse': float, 'sigma': float}, ...].
        sobol_per_step: [[S_0, S_1, ...], ...] one list per step.
        variable_names: for the Sobol panel.
        true_sigma: if known, draw reference line.
        ground_truth: [true_S_0, true_S_1, ...] for dotted lines.
    """
    apply_style()
    n_steps = len(steps)
    n_vars = len(sobol_per_step[0]) if sobol_per_step else 0
    if variable_names is None:
        variable_names = [f"$x_{{{i+1}}}$" for i in range(n_vars)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    labels = [s['label'] for s in steps]
    x_pos = np.arange(n_steps)
    bar_colors = [_var_color(i) for i in range(n_steps)]

    # Left: RMSE + sigma
    rmses = [s['rmse'] for s in steps]
    sigmas = [s['sigma'] for s in steps]
    width = 0.35
    bars1 = ax1.bar(x_pos - width / 2, rmses, width, color=bar_colors,
                    alpha=0.8, edgecolor='0.3', linewidth=0.5, label='RMSE')
    bars2 = ax1.bar(x_pos + width / 2, sigmas, width, color=bar_colors,
                    alpha=0.4, edgecolor='0.3', linewidth=0.5, hatch='///',
                    label='$\\hat{\\sigma}$')
    if true_sigma is not None:
        ax1.axhline(true_sigma, ls='--', color='red', alpha=0.5,
                    label=f'True $\\sigma={true_sigma:.1f}$')
    for b, r in zip(bars1, rmses):
        ax1.text(b.get_x() + b.get_width() / 2, r + 0.01, f'{r:.3f}',
                 ha='center', va='bottom', fontsize=8)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel('RMSE / $\\hat{\\sigma}$')
    ax1.set_title('(a) More complexity $\\rightarrow$ less noise',
                  loc='left', fontweight='bold')
    ax1.legend(fontsize=8)

    # Right: Sobol stability
    for i in range(n_vars):
        vals = [sobol_per_step[k][i] for k in range(n_steps)]
        c = _var_color(i)
        label = variable_names[i]
        if ground_truth and i < len(ground_truth):
            label += f' (true {ground_truth[i]:.3f})'
            ax2.axhline(ground_truth[i], ls=':', color=c, alpha=0.2)
        ax2.plot(x_pos, vals, 'o-', color=c, linewidth=2, markersize=8,
                 label=label)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel('First-order Sobol $S_i$')
    ax2.set_title('(b) Attribution unchanged by residual',
                  loc='left', fontweight='bold')
    ax2.legend(fontsize=7, ncol=2)

    fig.suptitle('Complexity staircase: noise reduces, Sobol stays invariant',
                 fontsize=13, y=1.02)
    fig.tight_layout()
    return fig, (ax1, ax2)


# ============================================================================
# 35. LOO vs K-fold agreement
# ============================================================================

def plot_loo_vs_kfold(
    lambdas: np.ndarray,
    loos: np.ndarray,
    gcvs: np.ndarray,
    kfolds: np.ndarray,
    figsize: Tuple[float, float] = (12, 5),
) -> Tuple[plt.Figure, np.ndarray]:
    """Two-panel: LOO/GCV/K-fold on lambda path + scatter agreement.

    Left: all three CV estimates as lines on log-log lambda axis.
    Right: scatter of LOO vs K-fold coloured by log(lambda).

    Args:
        lambdas: (n_lambda,) grid.
        loos: (n_lambda,) exact LOO-CV values.
        gcvs: (n_lambda,) GCV values.
        kfolds: (n_lambda,) K-fold CV values.
    """
    apply_style()
    lambdas = _ensure_np(lambdas)
    loos = _ensure_np(loos)
    gcvs = _ensure_np(gcvs)
    kfolds = _ensure_np(kfolds)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    ax1.loglog(lambdas, loos, 'o-', color=PALETTE['order1'], linewidth=2,
               markersize=5, label='Exact LOO-CV')
    ax1.loglog(lambdas, gcvs, 's--', color=PALETTE['order3'], linewidth=1.5,
               markersize=4, label='GCV (approx LOO)')
    ax1.loglog(lambdas, kfolds, '^-', color=PALETTE['residual'], linewidth=1.5,
               markersize=5, label='5-fold CV (Woodbury)')
    ax1.set_xlabel(r'$\lambda$')
    ax1.set_ylabel('CV score (MSE)')
    ax1.set_title('(a) Three CV estimates on the $\\lambda$ path',
                  loc='left', fontweight='bold')
    ax1.legend(fontsize=9)

    sc = ax2.scatter(loos, kfolds, c=np.log10(lambdas), cmap='coolwarm',
                     s=60, edgecolors='0.3', linewidth=0.5, zorder=3)
    mn = min(loos.min(), kfolds.min())
    mx = max(loos.max(), kfolds.max())
    ax2.plot([mn, mx], [mn, mx], 'k--', alpha=0.3, linewidth=1)
    ax2.set_xlabel('Exact LOO-CV')
    ax2.set_ylabel('5-fold CV (Woodbury)')
    ax2.set_title('(b) LOO vs K-fold agreement',
                  loc='left', fontweight='bold')
    ax2.set_aspect('equal')
    plt.colorbar(sc, ax=ax2, shrink=0.8, label='$\\log_{10}\\lambda$')

    fig.suptitle('Cross-validation agreement: all from one ridge solve',
                 fontsize=13, y=1.02)
    fig.tight_layout()
    return fig, (ax1, ax2)


# ============================================================================
# 17. Bias-variance tradeoff: mean vs variance decomposition dual panel
# ============================================================================

def plot_bias_variance_tradeoff(
    path,  # RegPathResult
    nc: Optional[Dict] = None,
    variable_names: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (13, 5),
) -> Tuple[plt.Figure, np.ndarray]:
    """Dual-panel bias-variance tradeoff.

    Left: Explained variance (stacked by order) + residual MSE vs lambda.
          Shows how model complexity trades off with fit quality.
    Right: Noise estimate sigma^2(lambda) with df on twin axis.
           Shows the sweet spot between overfitting and underfitting.
    """
    apply_style()
    lambdas = _ensure_np(path.lambdas)
    gcv_idx = int(np.argmin(np.abs(lambdas - path.lambda_gcv_opt)))
    bic_idx = int(np.argmin(_ensure_np(path.bic_values)))

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # --- Left: explained vs residual variance ---
    ax = axes[0]
    o1 = _ensure_np(path.var_order1)
    o2 = _ensure_np(path.var_order2)
    o3 = _ensure_np(path.var_order3)
    mse = _ensure_np(path.mse_values)
    df = _ensure_np(path.df_values)

    # Normalise variance components to fraction of total
    vtot = o1 + o2 + o3
    vtot[vtot < 1e-15] = 1.0

    ax.fill_between(lambdas, 0, o1 / vtot, alpha=0.6,
                    color=PALETTE['order1'], label='1st order')
    ax.fill_between(lambdas, o1 / vtot, (o1 + o2) / vtot, alpha=0.6,
                    color=PALETTE['order2'], label='2nd order')
    if np.any(o3 > 0):
        ax.fill_between(lambdas, (o1 + o2) / vtot, 1.0, alpha=0.6,
                        color=PALETTE['order3'], label='3rd order')

    ax2 = ax.twinx()
    ax2.semilogx(lambdas, mse, '-', color=PALETTE['residual'], linewidth=2,
                 label='MSE')
    ax2.set_ylabel('Training MSE', color=PALETTE['residual'])
    ax2.tick_params(axis='y', labelcolor=PALETTE['residual'])

    ax.axvline(lambdas[gcv_idx], color='grey', ls=':', alpha=0.6)
    ax.axvline(lambdas[bic_idx], color=PALETTE['bic'], ls=':', alpha=0.6)
    ax.set_xscale('log')
    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel('Fraction of explained variance by order')
    ax.set_title('(a) Complexity tradeoff', loc='left', fontweight='bold')
    ax.set_ylim(0, 1.02)
    ax.legend(loc='upper left', fontsize=8)

    # --- Right: noise curve + df ---
    ax = axes[1]
    if nc is not None:
        nc_lam = _ensure_np(nc['lambdas'])
        nc_sig = _ensure_np(nc['sigma2'])
        nc_df = _ensure_np(nc['df'])
    else:
        # Build from reg path data
        nc_lam = lambdas
        n_obs = len(lambdas)
        nc_sig = mse * n_obs / np.maximum(n_obs - df, 1)
        nc_df = df

    ax.semilogx(nc_lam, nc_sig, '-', color=PALETTE['order1'], linewidth=2.5,
                label=r'$\hat\sigma^2(\lambda)$')

    ax3 = ax.twinx()
    ax3.semilogx(nc_lam, nc_df, '-', color=PALETTE['order2'], linewidth=1.5,
                 alpha=0.6, label='df')
    ax3.set_ylabel('Effective df', color=PALETTE['order2'])
    ax3.tick_params(axis='y', labelcolor=PALETTE['order2'])

    # Mark optima
    if nc is not None:
        for key, name, color, mk in [
            ('lambda_gcv_opt', 'GCV', PALETTE['gcv'], 'o'),
            ('lambda_bic_opt', 'BIC', PALETTE['bic'], 's'),
        ]:
            if key in nc:
                lv = nc[key]
                idx = int(np.argmin(np.abs(nc_lam - lv)))
                ax.plot(nc_lam[idx], nc_sig[idx], mk, color=color,
                        markersize=10, zorder=5, label=f'{name} opt')

    sig_min = nc_sig.min()
    ax.axhline(sig_min, color=PALETTE['muted'], ls='--', alpha=0.4,
               label=rf'$\hat\sigma^2_{{min}}={sig_min:.4f}$')

    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel(r'$\hat\sigma^2(\lambda)$')
    ax.set_title('(b) Noise estimation', loc='left', fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)

    fig.tight_layout()
    return fig, axes


# ============================================================================
# 5. Noise-complexity curve
# ============================================================================

def plot_noise_complexity(
    nc: Dict,
    figsize: Tuple[float, float] = (8, 5),
) -> Tuple[plt.Figure, plt.Axes]:
    """sigma^2(lambda) with LOO/GCV/BIC optima marked.

    Args:
        nc: output of noise_complexity_curve()
    """
    apply_style()
    fig, ax = plt.subplots(figsize=figsize)

    lambdas = _ensure_np(nc['lambdas'])
    sigma2 = _ensure_np(nc['sigma2'])

    ax.semilogx(lambdas, sigma2, '-', color=PALETTE['order1'], linewidth=2,
                label=r'$\hat\sigma^2(\lambda)$')

    # Mark optima
    markers = [
        ('lambda_gcv_opt', 'GCV', PALETTE['gcv'], 'o'),
        ('lambda_bic_opt', 'BIC', PALETTE['bic'], 's'),
        ('lambda_loo_opt', 'LOO', PALETTE['loo'], '^'),
    ]
    for key, name, color, marker in markers:
        if key in nc:
            lam_opt = nc[key]
            idx = int(np.argmin(np.abs(lambdas - lam_opt)))
            ax.plot(lambdas[idx], sigma2[idx], marker, color=color,
                    markersize=10, zorder=5, label=f'{name} optimum')

    # Noise floor line
    ax.axhline(nc['sigma2_min'], color=PALETTE['muted'], ls='--', alpha=0.5,
               label=rf'$\hat\sigma^2_{{min}} = {nc["sigma2_min"]:.4f}$')

    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel(r'$\hat\sigma^2(\lambda)$')
    ax.set_title('Noise-complexity tradeoff')
    ax.legend(loc='best')

    fig.tight_layout()
    return fig, ax


# ============================================================================
# 29. Model complexity Pareto frontier
# ============================================================================

def plot_pareto_models(
    model_results: List[Dict[str, float]],
    figsize: Tuple[float, float] = (8, 6),
) -> Tuple[plt.Figure, plt.Axes]:
    """Pareto frontier: RMSE vs effective df across model configurations.

    Each point is a model config. The Pareto front connects non-dominated
    models. Colour optionally encodes first-order fraction.

    Args:
        model_results: list of dicts with 'label', 'rmse', 'df'.
            Optional: 'order1_frac'.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=figsize)
    dfs = np.array([m['df'] for m in model_results])
    rmses = np.array([m['rmse'] for m in model_results])
    labels = [m.get('label', f'M{i}') for i, m in enumerate(model_results)]

    has_order = all('order1_frac' in m for m in model_results)
    if has_order:
        o1 = np.array([m['order1_frac'] for m in model_results])
        sc = ax.scatter(dfs, rmses, c=o1, cmap='RdYlBu', s=80,
                        edgecolors='white', linewidth=0.8, zorder=3,
                        vmin=0, vmax=1)
        plt.colorbar(sc, ax=ax, label='1st-order fraction')
    else:
        ax.scatter(dfs, rmses, s=80, color=PALETTE['order1'],
                   edgecolors='white', linewidth=0.8, zorder=3)

    sorted_idx = np.argsort(dfs)
    pareto_idx = [sorted_idx[0]]
    best_rmse = rmses[sorted_idx[0]]
    for idx in sorted_idx[1:]:
        if rmses[idx] < best_rmse:
            pareto_idx.append(idx)
            best_rmse = rmses[idx]
    if len(pareto_idx) > 1:
        ax.step(dfs[pareto_idx], rmses[pareto_idx], where='post',
                color=PALETTE['highlight'], linewidth=2, alpha=0.7,
                label='Pareto front', zorder=2)
        ax.scatter(dfs[pareto_idx], rmses[pareto_idx], s=120,
                   facecolors='none', edgecolors=PALETTE['highlight'],
                   linewidth=2, zorder=4)

    for i, label in enumerate(labels):
        ax.annotate(label, (dfs[i], rmses[i]), fontsize=7,
                    xytext=(4, 4), textcoords='offset points')

    ax.set_xlabel('Effective degrees of freedom (complexity)')
    ax.set_ylabel('Validation RMSE')
    ax.set_title('Model Pareto frontier: complexity vs accuracy')
    ax.legend(loc='upper right')
    fig.tight_layout()
    return fig, ax


# ============================================================================
# 24. Multi-model comparison radar
# ============================================================================

def plot_model_comparison(
    models: Dict[str, Dict[str, float]],
    figsize: Tuple[float, float] = (8, 6),
) -> Tuple[plt.Figure, plt.Axes]:
    """Radar (spider) chart comparing multiple models across metrics.

    Args:
        models: {model_name: {metric_name: value, ...}}
            Typical metrics: 'RMSE', 'R2', 'df', 'Sobol_sum',
            'order1_frac', 'order2_frac', 'GCV', 'BIC'.
    """
    apply_style()
    model_names = list(models.keys())
    metrics = list(models[model_names[0]].keys())
    n_metrics = len(metrics)
    n_models = len(model_names)

    # Normalise each metric to [0, 1] for radar
    raw = np.zeros((n_models, n_metrics))
    for i, name in enumerate(model_names):
        for j, metric in enumerate(metrics):
            raw[i, j] = models[name].get(metric, 0)

    # For each metric: normalise. Higher = "better" visually.
    norm = np.zeros_like(raw)
    for j in range(n_metrics):
        col = raw[:, j]
        mn, mx = col.min(), col.max()
        if mx > mn:
            norm[:, j] = (col - mn) / (mx - mn)
        else:
            norm[:, j] = 0.5

    # Radar angles
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # close polygon

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': 'polar'})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    for i, name in enumerate(model_names):
        values = norm[i].tolist() + [norm[i, 0]]
        c = _var_color(i)
        ax.plot(angles, values, '-o', color=c, linewidth=2, markersize=5,
                label=name)
        ax.fill(angles, values, alpha=0.1, color=c)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['', '', '', ''], fontsize=7)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
    ax.set_title('Model comparison radar', fontweight='bold', pad=25)

    fig.tight_layout()
    return fig, ax
