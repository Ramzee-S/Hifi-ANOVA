"""Publication-quality plotting library for HiFi-ANOVA.

Every function takes data + returns (fig, axes). All figures are designed
for direct inclusion in papers: LaTeX-compatible fonts, clean axes,
consistent palette, 300 DPI export.

Usage:
    from hifi_anova.analysis.plots import (
        plot_reg_path_panel, plot_sobol_ci_bars, plot_dual_sobol,
        plot_sensitivity_ellipses,
        plot_variance_spectrum, plot_noise_complexity, plot_components,
        plot_interaction_grid, plot_frequency_content, plot_projection_matrix,
        plot_calibration, plot_leverage, plot_residual_diagnostics,
        save_fig, apply_style,
    )
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import LogLocator, NullFormatter
from typing import Dict, List, Optional, Tuple, Any

# ============================================================================
# Style & palette
# ============================================================================

# Consistent colour palette (colour-blind safe, prints well in greyscale)
PALETTE = {
    'order1':    '#3274A1',  # steel blue
    'order2':    '#E1812C',  # burnt orange
    'order3':    '#3A923A',  # forest green
    'residual':  '#C03D3E',  # muted red
    'gcv':       '#3274A1',
    'aic':       '#E1812C',
    'bic':       '#9372B2',  # muted purple
    'evidence':  '#C03D3E',
    'loo':       '#3A923A',
    'mean_sobol':  '#3274A1',
    'var_sobol':   '#C03D3E',
    'ci':          '#B0C4DE',  # light steel blue
    'highlight':   '#E1812C',
    'muted':       '#AAAAAA',
    'grid':        '#E0E0E0',
}

# Qualitative palette for per-variable lines (8 colours, CB-safe)
VAR_COLORS = [
    '#3274A1', '#E1812C', '#3A923A', '#C03D3E',
    '#9372B2', '#845B53', '#D584BD', '#7F7F7F',
    '#BCBD22', '#17BECF', '#AEC7E8', '#FFBB78',
]


def apply_style():
    """Set matplotlib rcParams for publication-quality output."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 8,
        'legend.framealpha': 0.8,
        'legend.edgecolor': '0.8',
        'axes.grid': True,
        'grid.alpha': 0.25,
        'grid.color': PALETTE['grid'],
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'mathtext.fontset': 'cm',
    })


def save_fig(fig, path: str, **kwargs):
    """Save figure at publication quality."""
    fig.savefig(path, dpi=kwargs.pop('dpi', 300),
                bbox_inches='tight', **kwargs)


def _ensure_np(arr):
    """Convert JAX/torch arrays to numpy."""
    if arr is None:
        return None
    return np.asarray(arr)


def _var_color(i: int) -> str:
    return VAR_COLORS[i % len(VAR_COLORS)]


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
# 2. Sobol bar chart with CI error bars
# ============================================================================

def plot_sobol_ci_bars(
    sobol_ci: Dict,
    variable_names: Optional[List[str]] = None,
    threshold: float = 0.01,
    figsize: Tuple[float, float] = (10, 5),
) -> Tuple[plt.Figure, plt.Axes]:
    """Horizontal Sobol bars with 95% CI whiskers.

    Args:
        sobol_ci: output of sobol_confidence_intervals()
        variable_names: names for each variable
        threshold: variables below this are greyed out
    """
    apply_style()
    first = sobol_ci['first_order']
    D = len(first)
    if variable_names is None:
        variable_names = [f"$x_{{{i+1}}}$" for i in range(D)]

    # Sort by Sobol value
    indices = sorted(first.keys(), key=lambda i: first[i][0], reverse=True)
    names = [variable_names[i] for i in indices]
    values = np.array([first[i][0] for i in indices])
    lo = np.array([first[i][1] for i in indices])
    hi = np.array([first[i][2] for i in indices])
    errors = np.array([values - lo, hi - values])

    fig, ax = plt.subplots(figsize=figsize)
    y_pos = np.arange(D)

    colors = [PALETTE['mean_sobol'] if v >= threshold else PALETTE['muted']
              for v in values]

    ax.barh(y_pos, values, xerr=errors, height=0.65,
            color=colors, edgecolor='white', linewidth=0.5,
            capsize=3, error_kw={'linewidth': 1.0, 'color': '0.3'})
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel('First-order Sobol index $S_i$')
    ax.set_title('Sensitivity indices with 95% CI')
    ax.invert_yaxis()
    ax.axvline(threshold, color=PALETTE['muted'], ls='--', alpha=0.5,
               label=f'threshold = {threshold}')

    # Second-order annotation
    if sobol_ci.get('second_order'):
        second = sobol_ci['second_order']
        top_2nd = sorted(second.items(), key=lambda kv: -kv[1][0])[:5]
        text_lines = []
        for (i, j), (s, slo, shi) in top_2nd:
            if s > threshold:
                ni = variable_names[i]
                nj = variable_names[j]
                text_lines.append(f"({ni},{nj}): {s:.3f} [{slo:.3f}, {shi:.3f}]")
        if text_lines:
            ax.text(0.98, 0.98, 'Top interactions:\n' + '\n'.join(text_lines),
                    transform=ax.transAxes, fontsize=7, va='top', ha='right',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat',
                              alpha=0.5))

    ax.legend(loc='lower right')

    fig.tight_layout()
    return fig, ax


# ============================================================================
# 3. Dual Sobol spectrum (mean vs variance sensitivity)
# ============================================================================

def plot_dual_sobol(
    sobol_results: Dict,
    variable_names: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (10, 5),
) -> Tuple[plt.Figure, plt.Axes]:
    """Paired bar chart: mean S_i^f (blue) vs variance S_i^h (red)."""
    apply_style()
    mean_first = sobol_results['mean_sobol']['first_order']
    D = len(mean_first)
    if variable_names is None:
        variable_names = [f"$x_{{{i+1}}}$" for i in range(D)]

    has_variance = 'variance_sobol' in sobol_results
    var_first = (sobol_results['variance_sobol']['first_order']
                 if has_variance else {i: 0.0 for i in range(D)})

    # Sort by total = mean + variance
    order = sorted(range(D),
                   key=lambda i: mean_first.get(i, 0) + var_first.get(i, 0),
                   reverse=True)

    x_pos = np.arange(D)
    width = 0.38

    mean_vals = [mean_first.get(i, 0) for i in order]
    var_vals = [var_first.get(i, 0) for i in order]
    names = [variable_names[i] for i in order]

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x_pos - width / 2, mean_vals, width, color=PALETTE['mean_sobol'],
           alpha=0.85, label=r'Mean sensitivity $S_i^f$', edgecolor='white',
           linewidth=0.5)
    if has_variance:
        ax.bar(x_pos + width / 2, var_vals, width, color=PALETTE['var_sobol'],
               alpha=0.85, label=r'Variance sensitivity $S_i^h$',
               edgecolor='white', linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Sobol index')
    ax.set_title('Dual Sobol spectrum: mean vs variance sensitivity')
    ax.legend()

    fig.tight_layout()
    return fig, ax


# ============================================================================
# 3b. Dual-sensitivity ellipses (mean vs variance, per variable)
# ============================================================================

def plot_sensitivity_ellipses(
    sobol_results: Dict,
    variable_names: Optional[List[str]] = None,
    mode: str = 'glyph',
    mean_ci: Optional[Dict] = None,
    var_ci: Optional[Dict] = None,
    use: str = 'first_order',
    top_k: Optional[int] = None,
    ci_scale: float = 1.0,
    figsize: Optional[Tuple[float, float]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Dual (mean vs variance) Sobol spectrum as one ellipse per variable.

    Two views of the idea that every variable carries a *pair* of sensitivities
    — one for the mean E[y|x], one for the variance Var[y|x]:

    ``mode='glyph'`` (default) — one ellipse per variable, its **width** the
        mean sensitivity ``S_i^f`` and its **height** the variance sensitivity
        ``S_i^h``. Shape is the message: wide/flat = mean driver, tall/narrow =
        variance driver, circular = balanced dual-role variable.

    ``mode='plane'`` — the quantitative scatter, each variable at
        ``(S_i^f, S_i^h)``. Bottom-right = mean drivers, top-left = hidden
        variance drivers, top-right = dual-role. With ``mean_ci``/``var_ci``
        (dicts ``{i: (lo, hi)}``) each marker becomes a CI ellipse, magnified by
        ``ci_scale`` (stated in the legend) for visibility.

    Returns (fig, ax), following the module convention (use ``save_fig`` to write).
    """
    apply_style()
    mean_first = sobol_results['mean_sobol'][use]
    D = len(mean_first)
    if variable_names is None:
        variable_names = [f"$x_{{{i+1}}}$" for i in range(D)]

    has_var = 'variance_sobol' in sobol_results
    var_first = (sobol_results['variance_sobol'][use] if has_var
                 else {i: 0.0 for i in range(D)})

    idx = sorted(range(D), key=lambda i: float(mean_first.get(i, 0))
                 + float(var_first.get(i, 0)), reverse=True)
    if top_k is not None:
        idx = idx[:top_k]
    lvl = 'first-order' if use == 'first_order' else 'total-order'

    if mode == 'plane':
        return _plot_ellipse_plane(idx, mean_first, var_first, variable_names,
                                   mean_ci, var_ci, ci_scale, lvl, figsize)

    # ---- glyph mode -------------------------------------------------------
    vals = [max(float(mean_first.get(i, 0)), float(var_first.get(i, 0)))
            for i in idx]
    max_val = max(max(vals) if vals else 1.0, 1e-6)
    radius_max = 0.42
    scale = radius_max / max_val
    min_semi = 0.02

    fig, ax = plt.subplots(figsize=figsize or (1.7 * len(idx) + 1.5, 4.2))
    for slot, i in enumerate(idx):
        sf = float(mean_first.get(i, 0.0))
        sh = float(var_first.get(i, 0.0))
        color = _var_color(i)
        ax.add_patch(mpl.patches.Ellipse(
            (slot, 0), width=2 * max(scale * sf, min_semi),
            height=2 * max(scale * sh, min_semi), facecolor=color,
            edgecolor=color, alpha=0.45, linewidth=1.6))
        ax.annotate(variable_names[i], (slot, -radius_max - 0.14), ha='center',
                    va='top', fontsize=11, color=color, fontweight='bold')
        ax.annotate(f"$S^f$={sf:.2f}\n$S^h$={sh:.2f}", (slot, radius_max + 0.06),
                    ha='center', va='bottom', fontsize=7.5,
                    color=PALETTE['muted'])

    ax.set_xlim(-0.7, len(idx) - 0.3)
    ax.set_ylim(-radius_max - 0.5, radius_max + 0.5)
    ax.set_aspect('equal')
    ax.axhline(0, color=PALETTE['grid'], lw=0.8, zorder=0)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ('left', 'right', 'top', 'bottom'):
        ax.spines[sp].set_visible(False)
    ax.set_title('Dual-sensitivity glyphs: mean vs variance')
    ax.text(0.5, -0.02,
            r'width $\propto$ mean sensitivity $S^f$    ·    '
            r'height $\propto$ variance sensitivity $S^h$',
            transform=ax.transAxes, ha='center', va='top', fontsize=9,
            color=PALETTE['muted'])

    fig.tight_layout()
    return fig, ax


def _plot_ellipse_plane(idx, mean_first, var_first, variable_names,
                        mean_ci, var_ci, ci_scale, lvl, figsize):
    """(S^f, S^h) plane with per-variable CI ellipses; returns (fig, ax)."""
    min_radius = 0.010

    def _half(ci, i):
        if ci is not None and i in ci:
            lo, hi = ci[i]
            return max((float(hi) - float(lo)) / 2.0 * ci_scale, min_radius)
        return min_radius

    fig, ax = plt.subplots(figsize=figsize or (6.6, 6.2))
    max_val = 0.0
    for i in idx:
        mx, vy = float(mean_first.get(i, 0.0)), float(var_first.get(i, 0.0))
        max_val = max(max_val, mx, vy)
        color = _var_color(i)
        ax.add_patch(mpl.patches.Ellipse(
            (mx, vy), width=2 * _half(mean_ci, i), height=2 * _half(var_ci, i),
            facecolor=color, edgecolor=color, alpha=0.35, linewidth=1.4, zorder=2))
        ax.plot(mx, vy, 'o', color=color, markersize=4, zorder=3)
        ax.annotate(variable_names[i], (mx, vy), textcoords='offset points',
                    xytext=(7, 6), fontsize=10, color=color,
                    fontweight='bold', zorder=4)

    hi = max(max_val * 1.2, 0.1)
    lbl = 'equal influence' if ci_scale == 1.0 else f'equal (CI ×{ci_scale:g})'
    ax.plot([0, hi], [0, hi], '--', color=PALETTE['muted'], lw=1, zorder=1,
            label=lbl)
    ax.text(0.97 * hi, 0.05 * hi, 'mean\ndrivers', ha='right', va='bottom',
            fontsize=8, color=PALETTE['muted'], style='italic')
    ax.text(0.03 * hi, 0.97 * hi, 'hidden variance\ndrivers', ha='left',
            va='top', fontsize=8, color=PALETTE['muted'], style='italic')
    ax.text(0.97 * hi, 0.97 * hi, 'dual-role', ha='right', va='top',
            fontsize=8, color=PALETTE['muted'], style='italic')

    ax.set_xlim(-0.02 * hi, hi)
    ax.set_ylim(-0.02 * hi, hi)
    ax.set_aspect('equal')
    ax.set_xlabel(f'Mean {lvl} Sobol  $S_i^f$')
    ax.set_ylabel(f'Variance {lvl} Sobol  $S_i^h$')
    ax.set_title('Dual-sensitivity plane: mean vs variance')
    ax.legend(loc='lower right')

    fig.tight_layout()
    return fig, ax


# ============================================================================
# 4. Interaction-order variance spectrum (cross-dataset)
# ============================================================================

def plot_variance_spectrum(
    spectra: Dict[str, Dict[str, float]],
    figsize: Tuple[float, float] = (8, 5),
) -> Tuple[plt.Figure, plt.Axes]:
    """Stacked bars: one bar per dataset, orders 1/2/3 + residual.

    Args:
        spectra: {dataset_name: {'order1': frac, 'order2': frac,
                                 'order3': frac, 'residual': frac}}
    """
    apply_style()
    names = list(spectra.keys())
    n = len(names)

    o1 = [spectra[k].get('order1', 0) for k in names]
    o2 = [spectra[k].get('order2', 0) for k in names]
    o3 = [spectra[k].get('order3', 0) for k in names]
    res = [spectra[k].get('residual', 0) for k in names]

    fig, ax = plt.subplots(figsize=figsize)
    x_pos = np.arange(n)
    w = 0.55

    bars1 = ax.bar(x_pos, o1, w, color=PALETTE['order1'], label='1st order')
    bars2 = ax.bar(x_pos, o2, w, bottom=o1, color=PALETTE['order2'],
                   label='2nd order')
    bot3 = [a + b for a, b in zip(o1, o2)]
    if any(v > 0 for v in o3):
        bars3 = ax.bar(x_pos, o3, w, bottom=bot3, color=PALETTE['order3'],
                       label='3rd order')
    bot4 = [a + b for a, b in zip(bot3, o3)]
    if any(v > 0 for v in res):
        ax.bar(x_pos, res, w, bottom=bot4, color=PALETTE['residual'],
               alpha=0.7, label='Residual')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=30, ha='right')
    ax.set_ylabel('Fraction of explained variance')
    ax.set_title('Interaction-order variance spectrum')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper right')

    fig.tight_layout()
    return fig, ax


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
# 6. Component function curves
# ============================================================================

def plot_components(
    model,
    active_vars: Optional[List[int]] = None,
    variable_names: Optional[List[str]] = None,
    n_grid: int = 200,
    ncols: int = 4,
    figsize_per: Tuple[float, float] = (3.2, 2.8),
) -> Tuple[plt.Figure, np.ndarray]:
    """f_i(x_i) component curves for each active variable.

    Args:
        model: fitted HiFiANOVA
        active_vars: which variables to plot (default: all)
        variable_names: display names
        n_grid: grid resolution
        ncols: columns in subplot grid
    """
    import jax.numpy as jnp
    from ..core.features import build_per_variable_basis
    apply_style()

    D = model.D
    K1 = model.K1
    if active_vars is None:
        active_vars = list(range(D))
    if variable_names is None:
        variable_names = [f"$x_{{{i+1}}}$" for i in range(D)]

    n_vars = len(active_vars)
    nrows = max(1, (n_vars + ncols - 1) // ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(figsize_per[0] * ncols,
                                      figsize_per[1] * nrows),
                             squeeze=False)

    x_grid = jnp.linspace(0, 1, n_grid)
    x_1d = x_grid[:, None]
    basis = build_per_variable_basis(x_1d, K1)  # (n_grid, 1, 2K1+1)
    basis_1d = np.asarray(basis[:, 0, :])  # (n_grid, 2K1+1)
    x_np = np.asarray(x_grid)

    for idx, var_i in enumerate(active_vars):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        wi = np.asarray(model.mean_model.get_coefficients_for_variable(var_i))
        f_i = basis_1d @ wi

        ax.plot(x_np, f_i, '-', color=_var_color(idx), linewidth=2)
        ax.axhline(0, color=PALETTE['muted'], ls='-', alpha=0.3, linewidth=0.5)
        ax.set_xlabel(variable_names[var_i], fontsize=9)
        ax.set_ylabel(f'$f_{{{var_i + 1}}}$', fontsize=9)
        ax.set_xlim(0, 1)

    # Hide unused axes
    for idx in range(n_vars, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle('First-order component functions', fontweight='bold', y=1.01)
    fig.tight_layout()
    return fig, axes


# ============================================================================
# 7. Interaction heatmap grid
# ============================================================================

def plot_interaction_grid(
    model,
    top_k: int = 6,
    sobol_results: Optional[Dict] = None,
    variable_names: Optional[List[str]] = None,
    n_grid: int = 60,
    figsize_per: float = 3.5,
) -> Tuple[plt.Figure, np.ndarray]:
    """Grid of 2D heatmaps for top-K interaction pairs.

    Args:
        model: fitted HiFiANOVA with K2 > 0
        top_k: number of pairs to show
        sobol_results: if provided, ranks pairs by second-order Sobol
        variable_names: display names
        n_grid: grid resolution
    """
    import jax.numpy as jnp
    from ..core.features import build_per_variable_basis, basis_size
    apply_style()

    K2 = model.K2
    if K2 == 0 or model.pair_indices is None:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.text(0.5, 0.5, 'No second-order terms', ha='center', va='center',
                transform=ax.transAxes)
        return fig, np.array([[ax]])

    D = model.D
    if variable_names is None:
        variable_names = [f"$x_{{{i+1}}}$" for i in range(D)]

    pair_indices = np.asarray(model.pair_indices)
    P = pair_indices.shape[0]

    # Rank pairs
    if sobol_results and 'mean_sobol' in sobol_results:
        second_order = sobol_results['mean_sobol'].get('second_order', {})
        pair_list = []
        for p in range(P):
            i, j = int(pair_indices[p, 0]), int(pair_indices[p, 1])
            s = second_order.get((i, j), 0.0)
            pair_list.append((p, i, j, s))
        pair_list.sort(key=lambda x: -x[3])
    else:
        pair_list = [(p, int(pair_indices[p, 0]), int(pair_indices[p, 1]), 0.0)
                     for p in range(P)]

    show = pair_list[:min(top_k, P)]
    n_show = len(show)
    ncols = min(3, n_show)
    nrows = max(1, (n_show + ncols - 1) // ncols)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(figsize_per * ncols,
                                      (figsize_per - 0.3) * nrows),
                             squeeze=False)

    incl_lin = getattr(model, 'include_linear_2', True)
    B = basis_size(K2, incl_lin)
    x_grid = jnp.linspace(0, 1, n_grid)
    x_1d = x_grid[:, None]
    basis = build_per_variable_basis(x_1d, K2, include_linear=incl_lin)
    basis_1d = np.asarray(basis[:, 0, :])  # (n_grid, B)

    for idx, (p, i, j, s) in enumerate(show):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        wp = np.asarray(model.mean_model.get_coefficients_for_pair(p))
        W = wp.reshape(B, B)
        Z = basis_1d @ W @ basis_1d.T

        vmax = max(abs(Z.max()), abs(Z.min()))
        if vmax < 1e-10:
            vmax = 1.0
        im = ax.imshow(Z, origin='lower', extent=[0, 1, 0, 1],
                       cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='equal')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xlabel(variable_names[j])
        ax.set_ylabel(variable_names[i])
        title = f'$f_{{{i + 1},{j + 1}}}$'
        if s > 0:
            title += f'  ($S$ = {s:.3f})'
        ax.set_title(title, fontsize=9)

    for idx in range(n_show, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle('Interaction surfaces', fontweight='bold', y=1.01)
    fig.tight_layout()
    return fig, axes


# ============================================================================
# 8. Frequency content (per-variable bar chart)
# ============================================================================

def plot_frequency_content(
    model,
    active_vars: Optional[List[int]] = None,
    variable_names: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (10, 5),
) -> Tuple[plt.Figure, plt.Axes]:
    """Per-variable breakdown: variance from linear vs cos1 vs sin1 vs cos2 ...

    Uses analytic Gram: Var contributed by basis function j of variable i
    equals w_{i,j}^2 (since G1 is diagonal for the canonical Fourier basis).
    """
    from ..core.gram import build_gram_matrix
    apply_style()

    D = model.D
    K1 = model.K1
    if active_vars is None:
        active_vars = list(range(D))
    if variable_names is None:
        variable_names = [f"$x_{{{i+1}}}$" for i in range(D)]

    G1 = np.asarray(build_gram_matrix(K1), dtype=np.float64)
    G1_diag = np.diag(G1)
    block = 2 * K1 + 1

    # Basis labels: lin, cos1, sin1, cos2, sin2, ...
    basis_labels = ['lin']
    for k in range(1, K1 + 1):
        basis_labels.extend([f'cos{k}', f'sin{k}'])

    n_vars = len(active_vars)
    n_basis = block
    data = np.zeros((n_vars, n_basis))

    for idx, vi in enumerate(active_vars):
        wi = np.asarray(model.mean_model.get_coefficients_for_variable(vi))
        data[idx] = wi ** 2 * G1_diag  # per-basis variance

    # Normalise each variable to sum to 1
    row_sums = data.sum(axis=1, keepdims=True)
    row_sums[row_sums < 1e-15] = 1.0
    data_frac = data / row_sums

    # Stacked bar chart
    fig, ax = plt.subplots(figsize=figsize)
    x_pos = np.arange(n_vars)
    w = 0.6

    # Colour map for basis functions
    freq_colors = ['#2C3E50']  # lin: dark
    freq_cmap = plt.cm.Spectral_r
    for k in range(1, K1 + 1):
        c = freq_cmap(k / (K1 + 1))
        freq_colors.extend([c, c])  # same colour for cos/sin pair

    bottom = np.zeros(n_vars)
    for j in range(n_basis):
        ax.bar(x_pos, data_frac[:, j], w, bottom=bottom,
               color=freq_colors[j], label=basis_labels[j] if j < 7 else '',
               edgecolor='white', linewidth=0.3)
        bottom += data_frac[:, j]

    ax.set_xticks(x_pos)
    ax.set_xticklabels([variable_names[i] for i in active_vars],
                       rotation=45, ha='right')
    ax.set_ylabel('Fraction of component variance')
    ax.set_title('Frequency content per variable')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper right', fontsize=7, ncol=2)

    fig.tight_layout()
    return fig, ax


# ============================================================================
# 9. Projection diagnostic heatmap
# ============================================================================

def plot_projection_matrix(
    discovery_result,  # MissingPairResult
    D: int,
    variable_names: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (7, 6),
) -> Tuple[plt.Figure, plt.Axes]:
    """D x D matrix of per-pair residual capture scores.

    Args:
        discovery_result: output of scan_missing_pairs()
        D: number of variables
        variable_names: display names
    """
    apply_style()
    if variable_names is None:
        variable_names = [f"$x_{{{i+1}}}$" for i in range(D)]

    mat = np.zeros((D, D))
    for (i, j), score in discovery_result.pair_scores.items():
        mat[i, j] = score
        mat[j, i] = score

    fig, ax = plt.subplots(figsize=figsize)
    vmax = max(mat.max(), 0.01)
    im = ax.imshow(mat, cmap='YlOrRd', vmin=0, vmax=vmax,
                   interpolation='nearest')
    plt.colorbar(im, ax=ax, label='Fraction of residual variance captured')

    ax.set_xticks(range(D))
    ax.set_xticklabels(variable_names, rotation=45, ha='right')
    ax.set_yticks(range(D))
    ax.set_yticklabels(variable_names)
    ax.set_title('Interaction discovery: per-pair projection scores')

    # Annotate significant cells
    for (i, j), score in discovery_result.pair_scores.items():
        if score > 0.005:
            ax.text(j, i, f'{score:.3f}', ha='center', va='center',
                    fontsize=7, color='white' if score > vmax * 0.6 else 'black')

    fig.tight_layout()
    return fig, ax


# ============================================================================
# 10. Calibration plot
# ============================================================================

def plot_calibration(
    cal_report: Dict,
    figsize: Tuple[float, float] = (6, 5),
) -> Tuple[plt.Figure, plt.Axes]:
    """Observed vs nominal coverage at multiple confidence levels."""
    apply_style()
    fig, ax = plt.subplots(figsize=figsize)

    nominals = []
    observed = []
    for key, val in cal_report.items():
        if key.startswith('coverage_'):
            nom = float(key.split('_')[1])
            nominals.append(nom)
            observed.append(val)

    nominals = np.array(nominals)
    observed = np.array(observed)
    order = np.argsort(nominals)
    nominals = nominals[order]
    observed = observed[order]

    ax.plot([0, 1], [0, 1], '--', color=PALETTE['muted'], linewidth=1,
            label='Perfect calibration')
    ax.plot(nominals, observed, 'o-', color=PALETTE['order1'], linewidth=2,
            markersize=8, label='Observed')

    ax.set_xlabel('Nominal coverage')
    ax.set_ylabel('Observed coverage')
    ax.set_title('Calibration: predicted vs actual coverage')
    ax.set_xlim(0.4, 1.02)
    ax.set_ylim(0.4, 1.02)
    ax.set_aspect('equal')
    ax.legend()

    # Annotate moments
    text = (f"$\\bar{{z}} = {cal_report.get('mean_standardized_residual', 0):.3f}$\n"
            f"$\\mathrm{{Var}}(z) = {cal_report.get('var_standardized_residual', 0):.3f}$")
    ax.text(0.98, 0.02, text, transform=ax.transAxes, fontsize=8,
            va='bottom', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                      alpha=0.8))

    fig.tight_layout()
    return fig, ax


# ============================================================================
# 11. Leverage / influence plot
# ============================================================================

def plot_leverage(
    analytics: Dict,
    figsize: Tuple[float, float] = (8, 5),
) -> Tuple[plt.Figure, plt.Axes]:
    """Per-observation leverages H_ii with Cook's-distance-like influence.

    Args:
        analytics: output of ridge_analytics()
    """
    apply_style()
    leverages = _ensure_np(analytics['leverages'])
    residuals = _ensure_np(analytics['residuals'])
    N = len(leverages)
    sigma2 = analytics['sigma2_hat']
    df = analytics['df']

    # Standardised residuals
    std_res = residuals / np.sqrt(max(sigma2, 1e-15))

    # Cook's-distance analogue
    cook = (std_res ** 2 * leverages) / (df * (1 - leverages) ** 2 + 1e-15)

    fig, ax = plt.subplots(figsize=figsize)
    scatter = ax.scatter(leverages, std_res, c=cook, cmap='YlOrRd',
                         s=12, alpha=0.7, edgecolors='none')
    plt.colorbar(scatter, ax=ax, label="Cook's distance analogue")

    # Threshold lines
    h_bar = df / N
    ax.axvline(2 * h_bar, color=PALETTE['muted'], ls='--', alpha=0.6,
               label=f'$2\\bar{{h}} = {2 * h_bar:.3f}$')
    ax.axhline(2, color=PALETTE['residual'], ls=':', alpha=0.5)
    ax.axhline(-2, color=PALETTE['residual'], ls=':', alpha=0.5)

    ax.set_xlabel('Leverage $H_{ii}$')
    ax.set_ylabel('Standardised residual')
    ax.set_title('Leverage-residual plot')
    ax.legend(loc='best')

    fig.tight_layout()
    return fig, ax


# ============================================================================
# 12. Residual diagnostic panel
# ============================================================================

def plot_residual_diagnostics(
    analytics: Dict,
    figsize: Tuple[float, float] = (12, 4.5),
) -> Tuple[plt.Figure, np.ndarray]:
    """3-panel residual diagnostics: histogram, QQ, residual-vs-fitted.

    Args:
        analytics: output of ridge_analytics()
    """
    apply_style()
    residuals = _ensure_np(analytics['residuals'])
    leverages = _ensure_np(analytics['leverages'])
    sigma = np.sqrt(max(analytics['sigma2_hat'], 1e-15))
    std_res = residuals / sigma
    w = analytics['w']

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Panel A: histogram of standardised residuals
    ax = axes[0]
    ax.hist(std_res, bins=50, density=True, color=PALETTE['order1'],
            alpha=0.7, edgecolor='white', linewidth=0.3)
    x_norm = np.linspace(-4, 4, 200)
    from scipy.stats import norm
    ax.plot(x_norm, norm.pdf(x_norm), '-', color=PALETTE['residual'],
            linewidth=1.5, label='$\\mathcal{N}(0,1)$')
    ax.set_xlabel('Standardised residual')
    ax.set_ylabel('Density')
    ax.set_title('(a) Residual distribution', loc='left', fontweight='bold')
    ax.legend()

    # Panel B: QQ plot
    ax = axes[1]
    sorted_res = np.sort(std_res)
    n = len(sorted_res)
    theoretical = norm.ppf(np.linspace(1 / (n + 1), n / (n + 1), n))
    ax.scatter(theoretical, sorted_res, s=4, alpha=0.5,
               color=PALETTE['order1'], edgecolors='none')
    lim = max(abs(theoretical.min()), abs(theoretical.max()),
              abs(sorted_res.min()), abs(sorted_res.max()))
    ax.plot([-lim, lim], [-lim, lim], '--', color=PALETTE['muted'])
    ax.set_xlabel('Theoretical quantiles')
    ax.set_ylabel('Sample quantiles')
    ax.set_title('(b) Q-Q plot', loc='left', fontweight='bold')
    ax.set_aspect('equal')

    # Panel C: residuals vs fitted
    ax = axes[2]
    # Fitted = y - residual; but we don't have y. Use index instead.
    ax.scatter(np.arange(n), std_res, s=4, alpha=0.4,
               color=PALETTE['order1'], edgecolors='none')
    ax.axhline(0, color=PALETTE['muted'], ls='-', alpha=0.5)
    ax.axhline(2, color=PALETTE['residual'], ls=':', alpha=0.4)
    ax.axhline(-2, color=PALETTE['residual'], ls=':', alpha=0.4)
    ax.set_xlabel('Observation index')
    ax.set_ylabel('Standardised residual')
    ax.set_title('(c) Residuals vs index', loc='left', fontweight='bold')

    fig.tight_layout()
    return fig, axes


# ============================================================================
# 13. Sobol bars with frequency / degree sub-decomposition
# ============================================================================

def plot_sobol_spectrum(
    model,
    sobol_ci: Optional[Dict] = None,
    variable_names: Optional[List[str]] = None,
    show_top_n: int = 10,
    figsize: Tuple[float, float] = (11, 6),
) -> Tuple[plt.Figure, plt.Axes]:
    """Horizontal Sobol bars where each bar is split by frequency content.

    Each variable's Sobol index S_i is decomposed into sub-contributions
    from each basis function: S_{i,j} = w_{i,j}^2 * G[j,j] / V_total.
    The bar is colour-coded by frequency/degree.

    If sobol_ci is provided, 95% CI whiskers are overlaid.
    """
    from ..core.gram import build_gram_matrix
    from ..core.features import basis_size
    apply_style()

    D = model.D
    K1 = model.K1
    basis_name = getattr(model, 'basis_name', 'fourier')
    incl_lin = getattr(model, 'include_linear_1', True)
    if variable_names is None:
        variable_names = [f"$x_{{{i+1}}}$" for i in range(D)]

    G1 = np.asarray(build_gram_matrix(K1, include_linear=incl_lin,
                                       basis_name=basis_name), dtype=np.float64)
    G1_diag = np.diag(G1)
    B = basis_size(K1, incl_lin, basis_name)

    # Compute per-basis-function variance for each variable
    var_per_bf = np.zeros((D, B))
    for i in range(D):
        wi = np.asarray(model.mean_model.get_coefficients_for_variable(i))
        var_per_bf[i] = wi ** 2 * G1_diag

    total_var = var_per_bf.sum()
    if total_var < 1e-15:
        total_var = 1.0

    sobol_per_var = var_per_bf.sum(axis=1) / total_var

    # Sort by total Sobol
    order = np.argsort(sobol_per_var)[::-1][:show_top_n]

    # Build basis-function labels and colour map
    if basis_name == 'fourier':
        if incl_lin:
            labels = ['linear']
            for k in range(1, K1 + 1):
                labels.extend([f'$\\cos {k}\\pi$', f'$\\sin {k}\\pi$'])
        else:
            labels = []
            for k in range(1, K1 + 1):
                labels.extend([f'$\\cos {k}\\pi$', f'$\\sin {k}\\pi$'])
    elif basis_name == 'legendre':
        labels = [f'$P_{{{k}}}$' for k in range(1, K1 + 1)]
    elif basis_name == 'haar':
        labels = [f'$\\psi_{{{j}}}$' for j in range(B)]
    else:
        labels = [f'$\\phi_{{{j}}}$' for j in range(B)]

    # Colours: gradient from dark to light across frequencies
    cmap = plt.cm.viridis_r
    bf_colors = [cmap(j / max(B - 1, 1)) for j in range(B)]
    if basis_name == 'fourier' and incl_lin:
        bf_colors[0] = '#2C3E50'  # distinct dark for linear

    fig, ax = plt.subplots(figsize=figsize)
    n_show = len(order)
    y_pos = np.arange(n_show)

    for j in range(B):
        lefts = np.zeros(n_show)
        for rank in range(n_show):
            lefts[rank] = sum(var_per_bf[order[rank], :j]) / total_var
        widths = var_per_bf[order, j] / total_var
        lbl = labels[j] if j < len(labels) and j < 5 else (labels[j] if j == B - 1 else '')
        ax.barh(y_pos, widths, left=lefts, height=0.7, color=bf_colors[j],
                edgecolor='white', linewidth=0.3, label=lbl)

    # CI whiskers
    if sobol_ci and 'first_order' in sobol_ci:
        for rank, i in enumerate(order):
            if i in sobol_ci['first_order']:
                s, lo, hi = sobol_ci['first_order'][i]
                ax.plot([lo, hi], [rank, rank], '-', color='0.2',
                        linewidth=1.5, zorder=5)
                ax.plot([lo, hi], [rank, rank], '|', color='0.2',
                        markersize=6, zorder=5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([variable_names[i] for i in order])
    ax.set_xlabel('Sobol index $S_i$ (coloured by basis function)')
    ax.set_title('Sensitivity spectrum: Sobol index decomposed by frequency')
    ax.invert_yaxis()
    ax.legend(loc='lower right', fontsize=7, ncol=2,
              title=f'{basis_name.title()} basis', title_fontsize=8)

    fig.tight_layout()
    return fig, ax


# ============================================================================
# 14. Cross-basis Sobol comparison (Fourier vs Legendre vs Haar)
# ============================================================================

def plot_cross_basis_sobol(
    basis_comparison: Dict,
    variable_names: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (10, 6),
) -> Tuple[plt.Figure, plt.Axes]:
    """Grouped bars comparing Sobol indices from three independent basis fits.

    Args:
        basis_comparison: output of multi_basis_fit() — must have 'per_variable'
            with var_legendre, var_fourier, var_haar per variable.
        variable_names: display names.
    """
    apply_style()
    per_var = basis_comparison['per_variable']
    D = len(per_var)
    if variable_names is None:
        variable_names = [f"$x_{{{i+1}}}$" for i in range(D)]

    # Normalise each basis to Sobol fractions
    totals = {}
    for basis in ('legendre', 'fourier', 'haar'):
        key = f'var_{basis}'
        t = sum(per_var[i].get(key, 0) for i in range(D))
        totals[basis] = max(t, 1e-15)

    # Sort by average Sobol
    avg = {i: np.mean([(per_var[i].get(f'var_{b}', 0) / totals[b])
                        for b in ('legendre', 'fourier', 'haar')])
           for i in range(D)}
    order = sorted(range(D), key=lambda i: -avg[i])

    x = np.arange(D)
    w = 0.25
    colors = {'legendre': '#9372B2', 'fourier': '#3274A1', 'haar': '#3A923A'}

    fig, ax = plt.subplots(figsize=figsize)
    for idx, basis in enumerate(('legendre', 'fourier', 'haar')):
        key = f'var_{basis}'
        vals = [per_var[order[r]].get(key, 0) / totals[basis] for r in range(D)]
        ax.bar(x + idx * w, vals, w, color=colors[basis], alpha=0.85,
               edgecolor='white', linewidth=0.5,
               label=basis.title())

    # Mark best basis per variable
    for r in range(D):
        best = per_var[order[r]].get('best_basis', '')
        if best and best in ('legendre', 'fourier', 'haar'):
            col_idx = list(('legendre', 'fourier', 'haar')).index(best)
            ax.plot(r + col_idx * w,
                    per_var[order[r]].get(f'var_{best}', 0) / totals[best] + 0.01,
                    '*', color='#E1812C', markersize=8, zorder=5)

    ax.set_xticks(x + w)
    ax.set_xticklabels([variable_names[i] for i in order],
                       rotation=45, ha='right')
    ax.set_ylabel('Sobol index $S_i$')
    ax.set_title('Cross-basis comparison: Legendre vs Fourier vs Haar')
    ax.legend()

    fig.tight_layout()
    return fig, ax


# ============================================================================
# 15. Second-order interaction matrix (symmetric heatmap)
# ============================================================================

def plot_interaction_matrix(
    sobol_results: Dict,
    variable_names: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (7, 6.5),
) -> Tuple[plt.Figure, plt.Axes]:
    """Symmetric D x D heatmap of second-order Sobol indices.

    Diagonal shows first-order S_i, off-diagonal shows S_ij.
    Upper triangle: filled. Lower triangle: annotated values.
    """
    apply_style()
    first = sobol_results['mean_sobol']['first_order']
    second = sobol_results['mean_sobol'].get('second_order', {})
    D = len(first)
    if variable_names is None:
        variable_names = [f"$x_{{{i+1}}}$" for i in range(D)]

    # Build matrix
    mat = np.zeros((D, D))
    for i in range(D):
        mat[i, i] = first.get(i, 0)
    for (i, j), s in second.items():
        mat[i, j] = s
        mat[j, i] = s

    # Sort by first-order
    order = sorted(range(D), key=lambda i: -first.get(i, 0))
    mat_sorted = mat[np.ix_(order, order)]
    names_sorted = [variable_names[i] for i in order]

    fig, ax = plt.subplots(figsize=figsize)

    # Upper triangle + diagonal: heatmap
    mask_lower = np.tril(np.ones_like(mat_sorted, dtype=bool), k=-1)
    display = np.where(mask_lower, np.nan, mat_sorted)

    vmax = max(np.nanmax(display), 0.01)
    cmap_im = plt.cm.YlOrRd.copy()
    cmap_im.set_bad('white')
    im = ax.imshow(display, cmap=cmap_im, vmin=0, vmax=vmax,
                   interpolation='nearest', aspect='equal')
    plt.colorbar(im, ax=ax, label='Sobol index', shrink=0.85)

    # Lower triangle: text annotations
    for r in range(D):
        for c in range(D):
            val = mat_sorted[r, c]
            if r > c and val > 0.001:
                ax.text(c, r, f'{val:.3f}', ha='center', va='center',
                        fontsize=7, color=PALETTE['order1'])
            elif r == c:
                ax.text(c, r, f'{val:.3f}', ha='center', va='center',
                        fontsize=7, fontweight='bold',
                        color='white' if val > vmax * 0.5 else 'black')

    ax.set_xticks(range(D))
    ax.set_xticklabels(names_sorted, rotation=45, ha='right')
    ax.set_yticks(range(D))
    ax.set_yticklabels(names_sorted)
    ax.set_title('Interaction matrix: $S_i$ (diagonal) + $S_{ij}$ (off-diagonal)')

    fig.tight_layout()
    return fig, ax


# ============================================================================
# 16. Sobol forest plot with CI & significance markers
# ============================================================================

def plot_sobol_forest(
    sobol_ci: Dict,
    variable_names: Optional[List[str]] = None,
    threshold: float = 0.01,
    figsize: Tuple[float, float] = (8, 7),
) -> Tuple[plt.Figure, plt.Axes]:
    """Forest plot: point estimate + CI for all components (1st + 2nd + 3rd).

    Components are grouped by order and sorted by magnitude.
    Significant entries (CI excludes 0 and S > threshold) are highlighted.
    """
    apply_style()
    first = sobol_ci.get('first_order', {})
    second = sobol_ci.get('second_order', {})
    third = sobol_ci.get('third_order', {})

    D = len(first)
    if variable_names is None:
        variable_names = [f"$x_{{{i+1}}}$" for i in range(D)]

    # Collect all components
    entries = []  # (label, S, lo, hi, order)
    for i in sorted(first.keys()):
        s, lo, hi = first[i]
        entries.append((variable_names[i], s, lo, hi, 1))
    for (i, j) in sorted(second.keys()):
        s, lo, hi = second[(i, j)]
        if s > 0.001:
            entries.append((f'({variable_names[i]},{variable_names[j]})',
                            s, lo, hi, 2))
    for key in sorted(third.keys()):
        s, lo, hi = third[key]
        if s > 0.001:
            names_t = ','.join(variable_names[k] for k in key)
            entries.append((f'({names_t})', s, lo, hi, 3))

    # Sort within each order by S descending
    entries.sort(key=lambda e: (-e[4], -e[1]))

    n = len(entries)
    fig, ax = plt.subplots(figsize=figsize)
    y_pos = np.arange(n)

    order_colors = {1: PALETTE['order1'], 2: PALETTE['order2'],
                    3: PALETTE['order3']}
    order_labels_shown = set()

    for idx, (label, s, lo, hi, order) in enumerate(entries):
        c = order_colors[order]
        sig = lo > 0 and s > threshold

        # CI line
        ax.plot([lo, hi], [idx, idx], '-', color=c,
                linewidth=2.5 if sig else 1.0, alpha=1.0 if sig else 0.4)
        # Point estimate
        marker = 'D' if sig else 'o'
        ax.plot(s, idx, marker, color=c, markersize=7 if sig else 4,
                markeredgecolor='white', markeredgewidth=0.5, zorder=5)

        # Legend entry (once per order)
        olabel = {1: '1st order', 2: '2nd order', 3: '3rd order'}[order]
        if order not in order_labels_shown:
            ax.plot([], [], 's', color=c, label=olabel, markersize=8)
            order_labels_shown.add(order)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([e[0] for e in entries], fontsize=8)
    ax.axvline(threshold, color=PALETTE['muted'], ls='--', alpha=0.4,
               label=f'Threshold = {threshold}')
    ax.axvline(0, color='0.5', ls='-', linewidth=0.5)
    ax.set_xlabel('Sobol index $S$')
    ax.set_title('Sensitivity forest: all components with 95% CI')
    ax.invert_yaxis()
    ax.legend(loc='lower right', fontsize=8)

    fig.tight_layout()
    return fig, ax


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
# 18. Basis character ternary plot
# ============================================================================

def plot_basis_ternary(
    characterization: Dict,
    variable_names: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (7, 6.5),
) -> Tuple[plt.Figure, plt.Axes]:
    """Ternary-style plot: polynomial vs oscillatory vs localized per variable.

    Uses a 2D projection of the simplex (equilateral triangle).
    Each variable is a point; size encodes share_of_total.

    Args:
        characterization: output of cross_residual_characterization() or
            sequential_projection_characterization().
    """
    apply_style()
    per_var = characterization['per_variable']
    D = len(per_var)
    if variable_names is None:
        variable_names = [f"$x_{{{i+1}}}$" for i in range(D)]

    # Ternary coordinates -> 2D cartesian
    # Vertices: Polynomial=(0,0), Oscillatory=(1,0), Localized=(0.5, sqrt(3)/2)
    def ternary_to_xy(p, o, l):
        total = p + o + l
        if total < 1e-15:
            return 0.33, 0.33 * np.sqrt(3) / 2
        p, o, l = p / total, o / total, l / total
        x = o + l * 0.5
        y = l * np.sqrt(3) / 2
        return x, y

    fig, ax = plt.subplots(figsize=figsize)

    # Draw triangle
    tri_x = [0, 1, 0.5, 0]
    tri_y = [0, 0, np.sqrt(3) / 2, 0]
    ax.plot(tri_x, tri_y, '-', color='0.3', linewidth=1.5, zorder=1)

    # Grid lines (10%, 20%, ..., 90%)
    verts = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]])
    for frac in np.arange(0.1, 1.0, 0.1):
        for p1, p2, p3 in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
            start = verts[p1] * (1 - frac) + verts[p2] * frac
            end = verts[p1] * (1 - frac) + verts[p3] * frac
            ax.plot([start[0], end[0]], [start[1], end[1]], '-',
                    color='0.9', linewidth=0.5, zorder=0)

    # Vertex labels
    offset = 0.06
    ax.text(0, -offset, 'Polynomial\n(Legendre)', ha='center', va='top',
            fontsize=10, fontweight='bold', color='#9372B2')
    ax.text(1, -offset, 'Oscillatory\n(Fourier)', ha='center', va='top',
            fontsize=10, fontweight='bold', color='#3274A1')
    ax.text(0.5, np.sqrt(3) / 2 + offset, 'Localized\n(Haar)', ha='center',
            va='bottom', fontsize=10, fontweight='bold', color='#3A923A')

    # Plot variables
    char_colors = {
        'polynomial': '#9372B2', 'oscillatory': '#3274A1',
        'localized': '#3A923A', 'mixed': '#E1812C', 'negligible': '#AAAAAA',
    }

    for i in sorted(per_var.keys()):
        info = per_var[i]
        p = info.get('poly_fraction', 0)
        o = info.get('osc_fraction', 0)
        l = info.get('local_fraction', 0)
        share = info.get('share_of_total', 0.01)
        char = info.get('character', 'mixed')

        x, y = ternary_to_xy(p, o, l)
        size = max(30, min(500, share * 3000))

        ax.scatter(x, y, s=size, color=char_colors.get(char, '#AAAAAA'),
                   alpha=0.75, edgecolors='white', linewidth=0.8, zorder=3)
        name = variable_names[i] if i < len(variable_names) else f"x{i+1}"
        ax.annotate(name, (x, y), fontsize=7, ha='center', va='bottom',
                    xytext=(0, 6), textcoords='offset points')

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.15, np.sqrt(3) / 2 + 0.15)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Basis character: polynomial vs oscillatory vs localized',
                 fontweight='bold', pad=15)

    fig.tight_layout()
    return fig, ax


# ============================================================================
# 19. Sobol waterfall chart
# ============================================================================

def plot_sobol_waterfall(
    sobol_results: Dict,
    variable_names: Optional[List[str]] = None,
    show_top_n: int = 12,
    figsize: Tuple[float, float] = (10, 5),
) -> Tuple[plt.Figure, plt.Axes]:
    """Waterfall chart: cumulative Sobol build-up from individual variables.

    Shows how total explained variance accumulates as you add each
    variable/pair/triple, making the marginal contribution of each visible.
    """
    apply_style()
    first = sobol_results['mean_sobol']['first_order']
    second = sobol_results['mean_sobol'].get('second_order', {})
    third = sobol_results['mean_sobol'].get('third_order', {})
    residual = sobol_results['mean_sobol'].get('residual', 0)

    D = len(first)
    if variable_names is None:
        variable_names = [f"$x_{{{i+1}}}$" for i in range(D)]

    # Collect all components
    items = []
    for i in sorted(first.keys(), key=lambda i: -first[i]):
        items.append((variable_names[i], first[i], 1))
    for (i, j) in sorted(second.keys(), key=lambda k: -second[k]):
        if second[(i, j)] > 0.005:
            items.append((f'({variable_names[i]},{variable_names[j]})',
                          second[(i, j)], 2))
    for key in sorted(third.keys(), key=lambda k: -third[k]):
        if third[key] > 0.005:
            names_t = ','.join(variable_names[k] for k in key)
            items.append((f'({names_t})', third[key], 3))

    # Truncate
    items = items[:show_top_n]
    if residual > 0.005:
        items.append(('Residual', residual, 0))

    n = len(items)
    fig, ax = plt.subplots(figsize=figsize)

    order_colors = {0: PALETTE['residual'], 1: PALETTE['order1'],
                    2: PALETTE['order2'], 3: PALETTE['order3']}

    cumulative = 0.0
    for idx, (label, val, order) in enumerate(items):
        ax.bar(idx, val, bottom=cumulative, color=order_colors[order],
               edgecolor='white', linewidth=0.5, width=0.7)
        # Connector line
        if idx < n - 1:
            ax.plot([idx + 0.35, idx + 0.65], [cumulative + val] * 2,
                    '-', color='0.5', linewidth=0.8)
        # Value label
        if val > 0.01:
            ax.text(idx, cumulative + val / 2, f'{val:.3f}',
                    ha='center', va='center', fontsize=7, color='white',
                    fontweight='bold')
        cumulative += val

    # Total line
    ax.axhline(cumulative, color='0.3', ls='--', alpha=0.5,
               label=f'Total = {cumulative:.3f}')

    ax.set_xticks(range(n))
    ax.set_xticklabels([item[0] for item in items], rotation=45, ha='right',
                       fontsize=8)
    ax.set_ylabel('Cumulative Sobol index')
    ax.set_title('Variance waterfall: cumulative sensitivity decomposition')
    ax.legend(loc='upper left')

    fig.tight_layout()
    return fig, ax


# ============================================================================
# 20. Component functions with confidence bands
# ============================================================================

def plot_components_with_ci(
    model,
    Phi: np.ndarray,
    y: np.ndarray,
    reg_diag: np.ndarray,
    active_vars: Optional[List[int]] = None,
    variable_names: Optional[List[str]] = None,
    n_grid: int = 200,
    ncols: int = 4,
    alpha: float = 0.05,
    figsize_per: Tuple[float, float] = (3.2, 2.8),
) -> Tuple[plt.Figure, np.ndarray]:
    """Component curves f_i(x_i) with sandwich-estimated confidence bands.

    The band at each grid point is: f_i(x) +/- z * sqrt(phi_i^T Cov_w phi_i).
    """
    import jax.numpy as jnp
    from scipy.stats import norm as sp_norm
    from ..core.features import build_per_variable_basis
    from .automl import sandwich_covariance, ridge_analytics
    apply_style()

    D = model.D
    K1 = model.K1
    basis_name = getattr(model, 'basis_name', 'fourier')
    incl_lin = getattr(model, 'include_linear_1', True)
    if active_vars is None:
        active_vars = list(range(min(D, 8)))
    if variable_names is None:
        variable_names = [f"$x_{{{i+1}}}$" for i in range(D)]

    # Get covariance
    analytics = ridge_analytics(Phi, y, reg_diag)
    Cov_w = sandwich_covariance(Phi, analytics['A_inv'], analytics['residuals'])

    from ..core.features import basis_size
    block = basis_size(K1, incl_lin, basis_name)
    z_crit = sp_norm.ppf(1 - alpha / 2)

    # Grid
    x_grid = jnp.linspace(0, 1, n_grid)
    x_1d = x_grid[:, None]
    basis = build_per_variable_basis(x_1d, K1, include_linear=incl_lin,
                                      basis_name=basis_name)
    basis_1d = np.asarray(basis[:, 0, :])  # (n_grid, block)
    x_np = np.asarray(x_grid)

    n_vars = len(active_vars)
    nrows = max(1, (n_vars + ncols - 1) // ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(figsize_per[0] * ncols,
                                      figsize_per[1] * nrows),
                             squeeze=False)

    for idx, var_i in enumerate(active_vars):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        wi = np.asarray(model.mean_model.get_coefficients_for_variable(var_i))
        f_i = basis_1d @ wi

        # Confidence band
        sl = slice(var_i * block, (var_i + 1) * block)
        Cov_i = Cov_w[sl, sl]
        var_fi = np.array([b @ Cov_i @ b for b in basis_1d])
        se_fi = np.sqrt(np.maximum(var_fi, 0))

        ax.fill_between(x_np, f_i - z_crit * se_fi, f_i + z_crit * se_fi,
                        alpha=0.2, color=_var_color(idx))
        ax.plot(x_np, f_i, '-', color=_var_color(idx), linewidth=2)
        ax.axhline(0, color=PALETTE['muted'], ls='-', alpha=0.3, linewidth=0.5)
        ax.set_xlabel(variable_names[var_i], fontsize=9)
        ax.set_ylabel(f'$f_{{{var_i + 1}}}$', fontsize=9)
        ax.set_xlim(0, 1)

    for idx in range(n_vars, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle('Component functions with 95% confidence bands',
                 fontweight='bold', y=1.01)
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
# 22. Hierarchical variance sunburst
# ============================================================================

def plot_variance_sunburst(
    sobol_results: Dict,
    variable_names: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (8, 8),
) -> Tuple[plt.Figure, plt.Axes]:
    """Nested donut chart: inner ring = interaction order, outer = components.

    Inner ring segments: 1st-order, 2nd-order, 3rd-order, residual.
    Outer ring: each segment breaks into individual variables/pairs/triples.
    The visual weight of each slice is proportional to its Sobol index.
    """
    apply_style()
    va = sobol_results.get('variance_accounting', {})
    first = sobol_results['mean_sobol']['first_order']
    second = sobol_results['mean_sobol'].get('second_order', {})
    third = sobol_results['mean_sobol'].get('third_order', {})
    residual_frac = sobol_results['mean_sobol'].get('residual', 0)
    D = len(first)
    if variable_names is None:
        variable_names = [f"$x_{{{i+1}}}$" for i in range(D)]

    total = sum(first.values()) + sum(second.values()) + \
            sum(third.values()) + residual_frac
    if total < 1e-15:
        total = 1.0

    # --- Inner ring: order totals ---
    order_vals = [
        sum(first.values()) / total,
        sum(second.values()) / total,
        sum(third.values()) / total,
        residual_frac / total,
    ]
    order_labels = ['1st order', '2nd order', '3rd order', 'Residual']
    order_colors = [PALETTE['order1'], PALETTE['order2'],
                    PALETTE['order3'], PALETTE['residual']]

    # Remove zero-width slices
    inner_vals, inner_labels, inner_colors = [], [], []
    for v, l, c in zip(order_vals, order_labels, order_colors):
        if v > 0.001:
            inner_vals.append(v)
            inner_labels.append(l)
            inner_colors.append(c)

    # --- Outer ring: individual components ---
    outer_vals, outer_labels, outer_colors = [], [], []

    # 1st order components (sorted)
    for i in sorted(first.keys(), key=lambda i: -first[i]):
        v = first[i] / total
        if v > 0.002:
            outer_vals.append(v)
            outer_labels.append(variable_names[i])
            outer_colors.append(PALETTE['order1'])
        elif sum(first.values()) / total > 0.001:
            # Bundle tiny ones
            pass
    # Bundle remaining 1st order
    accounted_1st = sum(v for v, c in zip(outer_vals, outer_colors)
                         if c == PALETTE['order1'])
    leftover_1st = sum(first.values()) / total - accounted_1st
    if leftover_1st > 0.001:
        outer_vals.append(leftover_1st)
        outer_labels.append('other 1st')
        outer_colors.append(PALETTE['order1'])

    # 2nd order components
    for (i, j) in sorted(second.keys(), key=lambda k: -second[k]):
        v = second[(i, j)] / total
        if v > 0.005:
            outer_vals.append(v)
            ni, nj = variable_names[i], variable_names[j]
            outer_labels.append(f'{ni},{nj}')
            outer_colors.append(PALETTE['order2'])
    accounted_2nd = sum(v for v, c in zip(outer_vals, outer_colors)
                         if c == PALETTE['order2'])
    leftover_2nd = sum(second.values()) / total - accounted_2nd
    if leftover_2nd > 0.001:
        outer_vals.append(leftover_2nd)
        outer_labels.append('other 2nd')
        outer_colors.append(PALETTE['order2'])

    # 3rd order
    for key in sorted(third.keys(), key=lambda k: -third[k]):
        v = third[key] / total
        if v > 0.005:
            outer_vals.append(v)
            outer_labels.append(','.join(variable_names[k] for k in key))
            outer_colors.append(PALETTE['order3'])
    leftover_3rd = sum(third.values()) / total - \
        sum(v for v, c in zip(outer_vals, outer_colors)
            if c == PALETTE['order3'])
    if leftover_3rd > 0.001:
        outer_vals.append(leftover_3rd)
        outer_labels.append('other 3rd')
        outer_colors.append(PALETTE['order3'])

    # Residual
    if residual_frac / total > 0.001:
        outer_vals.append(residual_frac / total)
        outer_labels.append('residual')
        outer_colors.append(PALETTE['residual'])

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': 'polar'})

    # Convert to polar wedges manually for a donut chart
    # Use standard axes instead
    fig, ax = plt.subplots(figsize=figsize)

    # Inner donut
    wedges1, texts1 = ax.pie(
        inner_vals, labels=None, colors=inner_colors, radius=0.7,
        wedgeprops=dict(width=0.35, edgecolor='white', linewidth=2),
        startangle=90, counterclock=False)

    # Outer donut
    outer_colors_faded = [c + 'CC' if len(c) == 7 else c
                          for c in outer_colors]
    wedges2, texts2 = ax.pie(
        outer_vals, labels=None, colors=outer_colors, radius=1.05,
        wedgeprops=dict(width=0.3, edgecolor='white', linewidth=1),
        startangle=90, counterclock=False)

    # Label outer wedges (only significant ones)
    for wedge, label, val in zip(wedges2, outer_labels, outer_vals):
        if val > 0.03:
            ang = (wedge.theta1 + wedge.theta2) / 2
            x = 1.25 * np.cos(np.radians(ang))
            y = 1.25 * np.sin(np.radians(ang))
            ax.annotate(f'{label}\n{val:.1%}', xy=(x, y),
                        fontsize=7, ha='center', va='center')

    # Label inner wedges
    for wedge, label, val in zip(wedges1, inner_labels, inner_vals):
        if val > 0.03:
            ang = (wedge.theta1 + wedge.theta2) / 2
            x = 0.52 * np.cos(np.radians(ang))
            y = 0.52 * np.sin(np.radians(ang))
            ax.text(x, y, f'{label}\n{val:.0%}', ha='center', va='center',
                    fontsize=8, fontweight='bold', color='white')

    ax.set_title('Variance decomposition sunburst', fontweight='bold', pad=20)
    ax.set_aspect('equal')

    fig.tight_layout()
    return fig, ax


# ============================================================================
# 23. Residual sieve cascade ("peeling" diagram)
# ============================================================================

def plot_sieve_cascade(
    sieve_result,  # SieveResult
    variable_names: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (11, 5.5),
) -> Tuple[plt.Figure, np.ndarray]:
    """Two-panel sieve visualisation.

    Left: Horizontal cascade — start from total residual, peel off layers:
        first-order -> second-order -> third-order -> RBF -> noise.
        Each layer is a horizontal bar showing how much it captures.
        Bars narrow as structure is peeled away.

    Right: Top missed items per level — which specific variables/pairs/triples
        are hiding in the residual?
    """
    apply_style()
    sr = sieve_result

    fig, axes = plt.subplots(1, 2, figsize=figsize,
                             gridspec_kw={'width_ratios': [1.2, 1]})

    # --- Left: cascade bars ---
    ax = axes[0]
    levels = [
        ('Total residual', 1.0, '0.4'),
        ('1st order\n(re-projection)', sr.first_order_fraction, PALETTE['order1']),
        ('2nd order\n(missing pairs)', sr.second_order_fraction, PALETTE['order2']),
        ('3rd order\n(missing triples)', sr.third_order_fraction, PALETTE['order3']),
        ('Smooth\n(RBF subspace)', sr.residual_rbf_fraction, '#845B53'),
        ('Noise\n(unexplained)', sr.noise_fraction, PALETTE['muted']),
    ]

    y_pos = np.arange(len(levels))
    bars = ax.barh(y_pos, [v for _, v, _ in levels], height=0.65,
                   color=[c for _, _, c in levels], edgecolor='white',
                   linewidth=1)

    # Value labels
    for idx, (label, val, _) in enumerate(levels):
        if val > 0.005:
            ax.text(val + 0.01, idx, f'{val:.1%}', va='center', fontsize=9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([l for l, _, _ in levels])
    ax.set_xlabel('Fraction of residual variance')
    ax.set_title('(a) Residual decomposition by level', loc='left',
                 fontweight='bold')
    ax.invert_yaxis()
    ax.set_xlim(0, max(1.0, max(v for _, v, _ in levels)) * 1.15)

    # Annotation: recommendation
    if sr.recommendation:
        ax.text(0.98, 0.98, sr.recommendation, transform=ax.transAxes,
                fontsize=7, va='top', ha='right', style='italic',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                          alpha=0.8))

    # --- Right: top items per level ---
    ax = axes[1]

    items = []
    # Top missed pairs
    for (i, j), score in sr.top_pairs[:5]:
        if score > 0.001:
            if variable_names and i < len(variable_names):
                label = f'({variable_names[i]},{variable_names[j]})'
            else:
                label = f'({i},{j})'
            items.append((label, score, PALETTE['order2']))
    # Top missed triples
    for key, score in sr.top_triples[:3]:
        if score > 0.001:
            if variable_names:
                label = '(' + ','.join(
                    variable_names[k] if k < len(variable_names)
                    else str(k) for k in key) + ')'
            else:
                label = str(key)
            items.append((label, score, PALETTE['order3']))

    if items:
        items.sort(key=lambda x: -x[1])
        y2 = np.arange(len(items))
        ax.barh(y2, [v for _, v, _ in items], height=0.6,
                color=[c for _, _, c in items], edgecolor='white',
                linewidth=0.5)
        for idx, (label, val, _) in enumerate(items):
            ax.text(val + 0.002, idx, f'{val:.1%}', va='center', fontsize=8)
        ax.set_yticks(y2)
        ax.set_yticklabels([l for l, _, _ in items], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel('Fraction of residual')
        ax.set_title('(b) Top missed interactions', loc='left',
                     fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No significant\nmissed interactions',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=12, color=PALETTE['muted'])
        ax.set_title('(b) Top missed interactions', loc='left',
                     fontweight='bold')

    fig.tight_layout()
    return fig, axes


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


# ============================================================================
# 25. Interaction chord diagram
# ============================================================================

def plot_interaction_chord(
    sobol_results: Dict,
    variable_names: Optional[List[str]] = None,
    threshold: float = 0.005,
    figsize: Tuple[float, float] = (8, 8),
) -> Tuple[plt.Figure, plt.Axes]:
    """Chord-style diagram for pairwise interactions.

    Variables are placed on a circle. Arc width at each variable is
    proportional to first-order Sobol. Chords connect interacting pairs
    with width proportional to S_ij.
    """
    from matplotlib.patches import FancyArrowPatch, Arc
    from matplotlib.path import Path as MPath
    import matplotlib.patches as mpatches
    apply_style()

    first = sobol_results['mean_sobol']['first_order']
    second = sobol_results['mean_sobol'].get('second_order', {})
    D = len(first)
    if variable_names is None:
        variable_names = [f"$x_{{{i+1}}}$" for i in range(D)]

    # Filter significant interactions
    sig_pairs = {k: v for k, v in second.items() if v > threshold}
    if not sig_pairs:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f'No interactions above threshold ({threshold})',
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Interaction chord diagram')
        return fig, ax

    fig, ax = plt.subplots(figsize=figsize)

    # Place variables on circle
    angles = np.linspace(0, 2 * np.pi, D, endpoint=False)
    radius = 1.0
    node_x = radius * np.cos(angles)
    node_y = radius * np.sin(angles)

    # Draw nodes (size ~ first-order Sobol)
    max_s1 = max(first.values()) if first else 1
    for i in range(D):
        s = first.get(i, 0)
        size = max(100, min(1200, s / max(max_s1, 1e-10) * 1000))
        ax.scatter(node_x[i], node_y[i], s=size, color=_var_color(i),
                   edgecolors='white', linewidth=1.5, zorder=5)
        # Label
        label_r = 1.18
        ax.text(label_r * np.cos(angles[i]), label_r * np.sin(angles[i]),
                f'{variable_names[i]}\n$S$={s:.3f}',
                ha='center', va='center', fontsize=8, fontweight='bold')

    # Draw chords (Bezier curves through center)
    max_s2 = max(sig_pairs.values()) if sig_pairs else 1
    for (i, j), sij in sig_pairs.items():
        width = max(0.5, sij / max(max_s2, 1e-10) * 6)
        alpha_val = max(0.2, min(0.8, sij / max(max_s2, 1e-10)))

        # Quadratic Bezier through a point pulled toward center
        mid_x = 0.3 * (node_x[i] + node_x[j]) / 2
        mid_y = 0.3 * (node_y[i] + node_y[j]) / 2
        verts = [(node_x[i], node_y[i]),
                 (mid_x, mid_y),
                 (node_x[j], node_y[j])]
        codes = [MPath.MOVETO, MPath.CURVE3, MPath.CURVE3]
        path = MPath(verts, codes)
        patch = mpatches.PathPatch(
            path, facecolor='none', edgecolor=PALETTE['order2'],
            linewidth=width, alpha=alpha_val, capstyle='round')
        ax.add_patch(patch)

        # Label the chord at midpoint
        if sij > 0.02:
            ax.text(mid_x, mid_y, f'{sij:.3f}', fontsize=7,
                    ha='center', va='center', color=PALETTE['order2'],
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                              alpha=0.7, edgecolor='none'))

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Interaction chord diagram', fontweight='bold', pad=15)

    fig.tight_layout()
    return fig, ax


# ============================================================================
# 26. Structural vs correlative Sobol scatter
# ============================================================================

def plot_structural_vs_correlative(
    correlation_diag: Dict,
    figsize: Tuple[float, float] = (7, 6.5),
) -> Tuple[plt.Figure, plt.Axes]:
    """Scatter: structural (analytic) vs correlative (empirical) Sobol.

    Points on the 45-degree line = inputs are independent for that variable.
    Deviation = input correlations are confounding the attribution.
    Point size encodes the divergence.

    Args:
        correlation_diag: output of correlation_diagnostic().
    """
    apply_style()
    struct = correlation_diag['structural_indices']
    corr = correlation_diag['correlative_indices']
    div = correlation_diag.get('divergence', {})
    var_names = correlation_diag.get('variable_names', [])
    D = len(struct)
    if not var_names:
        var_names = [f"$x_{{{i+1}}}$" for i in range(D)]

    fig, ax = plt.subplots(figsize=figsize)

    # 45-degree reference
    lim = max(max(struct.values()), max(corr.values())) * 1.15
    ax.plot([0, lim], [0, lim], '--', color=PALETTE['muted'], linewidth=1,
            label='Perfect agreement')

    # Uncertainty band around diagonal (±0.05)
    ax.fill_between([0, lim], [0 - 0.05, lim - 0.05],
                    [0 + 0.05, lim + 0.05],
                    alpha=0.08, color=PALETTE['muted'])

    for i in sorted(struct.keys()):
        s = struct[i]
        c = corr[i]
        d = div.get(i, abs(s - c))
        size = max(40, min(400, d * 3000))
        color = PALETTE['residual'] if d > 0.05 else PALETTE['order1']
        ax.scatter(s, c, s=size, color=color, edgecolors='white',
                   linewidth=0.8, zorder=3, alpha=0.8)
        name = var_names[i] if i < len(var_names) else f"x{i+1}"
        ax.annotate(name, (s, c), fontsize=8, ha='left',
                    xytext=(5, 3), textcoords='offset points')

    ax.set_xlabel('Structural Sobol $S_i$ (independence-assuming)')
    ax.set_ylabel('Correlative Sobol $S_i^{\\mathrm{corr}}$ (data-aware)')
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect('equal')
    level = correlation_diag.get('correlation_level', '?')
    ax.set_title(f'Structural vs correlative Sobol (correlation: {level})')
    ax.legend(loc='upper left')

    # Summary stats
    text = (f"Sum structural: {correlation_diag.get('sum_structural', 0):.3f}\n"
            f"Sum correlative: {correlation_diag.get('sum_correlative', 0):.3f}\n"
            f"Max divergence: {correlation_diag.get('max_divergence', 0):.3f}")
    ax.text(0.98, 0.02, text, transform=ax.transAxes, fontsize=7,
            va='bottom', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                      alpha=0.8))

    fig.tight_layout()
    return fig, ax


# ============================================================================
# 27. Variance decomposition treemap
# ============================================================================

def plot_variance_treemap(
    sobol_results: Dict,
    variable_names: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (12, 7),
) -> Tuple[plt.Figure, plt.Axes]:
    """Treemap: nested rectangles showing hierarchical variance partition.

    Area of each rectangle is proportional to its Sobol index.
    Colour encodes interaction order. Labels inside if room permits.
    Inspired by financial treemaps (Shneiderman 1992) — uncommon in
    sensitivity analysis but excellent for communicating relative magnitudes.
    """
    import matplotlib.colors as mcolors
    apply_style()
    first = sobol_results['mean_sobol']['first_order']
    second = sobol_results['mean_sobol'].get('second_order', {})
    third = sobol_results['mean_sobol'].get('third_order', {})
    residual = sobol_results['mean_sobol'].get('residual', 0)
    D = len(first)
    if variable_names is None:
        variable_names = [f"x{i+1}" for i in range(D)]

    order_colors = {0: PALETTE['residual'], 1: PALETTE['order1'],
                    2: PALETTE['order2'], 3: PALETTE['order3']}
    order_labels_map = {0: 'Residual', 1: '1st order', 2: '2nd order',
                        3: '3rd order'}

    items = []
    for i in sorted(first.keys(), key=lambda i: -first[i]):
        if first[i] > 0.001:
            items.append((variable_names[i], first[i], 1))
    for (i, j) in sorted(second.keys(), key=lambda k: -second[k]):
        if second[(i, j)] > 0.001:
            items.append((f'{variable_names[i]},{variable_names[j]}',
                          second[(i, j)], 2))
    for key in sorted(third.keys(), key=lambda k: -third[k]):
        if third[key] > 0.001:
            lbl = ','.join(variable_names[k] for k in key)
            items.append((lbl, third[key], 3))
    if residual > 0.001:
        items.append(('Residual', residual, 0))

    if not items:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No significant components', ha='center',
                va='center', transform=ax.transAxes)
        return fig, ax

    # Slice-and-dice treemap layout
    def _layout(values, x, y, w, h, depth=0):
        if len(values) == 0:
            return []
        if len(values) == 1:
            return [(x, y, w, h)]
        total = sum(values)
        if total < 1e-15:
            return [(x, y, w, h)] * len(values)
        accum, split = 0, 1
        for k, v in enumerate(values):
            accum += v
            if accum >= total / 2:
                split = k + 1
                break
        split = max(1, min(split, len(values) - 1))
        frac = sum(values[:split]) / total

        if (depth % 2 == 0) == (w >= h):
            return (_layout(values[:split], x, y, w * frac, h, depth + 1) +
                    _layout(values[split:], x + w * frac, y,
                            w * (1 - frac), h, depth + 1))
        else:
            return (_layout(values[:split], x, y, w, h * frac, depth + 1) +
                    _layout(values[split:], x, y + h * frac,
                            w, h * (1 - frac), depth + 1))

    values = [v for _, v, _ in items]
    rects = _layout(values, 0, 0, 1, 1)

    fig, ax = plt.subplots(figsize=figsize)
    shown_orders = set()

    for (label, value, order), (rx, ry, rw, rh) in zip(items, rects):
        color = order_colors.get(order, PALETTE['muted'])
        rgb = mcolors.to_rgb(color)
        fill = tuple(min(1, c * 1.2 + 0.1) for c in rgb)
        rect = plt.Rectangle((rx, ry), rw, rh, facecolor=fill,
                              edgecolor=color, linewidth=2, alpha=0.85)
        ax.add_patch(rect)
        if rw * rh > 0.002:
            fontsize = max(6, min(11, int(np.sqrt(rw * rh) * 30)))
            ax.text(rx + rw / 2, ry + rh / 2, f'{label}\n{value:.1%}',
                    ha='center', va='center', fontsize=fontsize,
                    fontweight='bold', color='0.15')
        if order not in shown_orders:
            ax.plot([], [], 's', color=color, markersize=12,
                    label=order_labels_map.get(order, ''))
            shown_orders.add(order)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis('off')
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax.set_title('Variance treemap: area $\\propto$ Sobol index',
                 fontweight='bold', pad=10)
    fig.tight_layout()
    return fig, ax


# ============================================================================
# 28. Sieve orthogonal peeling strip
# ============================================================================

def plot_sieve_peeling(
    sieve_result,  # SieveResult
    variable_names: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (10, 4.5),
) -> Tuple[plt.Figure, plt.Axes]:
    """Progressive peeling diagram showing residual shrinkage.

    A horizontal bar starts at 100% (total residual). Successive
    orthogonal projections peel off layers. Arrows annotate the
    reduction at each step — the "sieve" metaphor made visual.
    """
    apply_style()
    sr = sieve_result
    fig, ax = plt.subplots(figsize=figsize)

    layers = [
        ('Original\nresidual', 1.0, '0.4'),
        ('After 1st-order\nprojection',
         max(0, 1.0 - sr.first_order_fraction), PALETTE['order1']),
        ('After 2nd-order\nprojection',
         max(0, 1.0 - sr.first_order_fraction - sr.second_order_fraction),
         PALETTE['order2']),
    ]
    if sr.third_order_fraction > 0.001:
        layers.append((
            'After 3rd-order\nprojection',
            max(0, 1.0 - sr.first_order_fraction - sr.second_order_fraction
                - sr.third_order_fraction), PALETTE['order3']))
    if sr.residual_rbf_fraction > 0.001:
        prev = layers[-1][1]
        layers.append(('After smooth\nprojection',
                       max(0, prev - sr.residual_rbf_fraction), '#845B53'))
    layers.append(('Irreducible\nnoise', max(0, sr.noise_fraction),
                   PALETTE['muted']))

    n = len(layers)
    for idx, (label, val, color) in enumerate(layers):
        ax.barh(idx, val, height=0.6, color=color,
                edgecolor='white', linewidth=1.5, alpha=0.85)
        if idx > 0 and idx < n - 1:
            prev_val = max(0, layers[idx - 1][1])
            reduction = prev_val - val
            if reduction > 0.005:
                ax.annotate('', xy=(val, idx - 0.35),
                            xytext=(prev_val, idx - 0.35),
                            arrowprops=dict(arrowstyle='->', color=color,
                                            lw=1.5))
                ax.text((val + prev_val) / 2, idx - 0.45,
                        f'$-${reduction:.1%}', ha='center', va='top',
                        fontsize=7, color=color, fontweight='bold')
        ax.text(max(val, 0) + 0.015, idx, f'{val:.1%}',
                va='center', fontsize=9, fontweight='bold')

    ax.set_yticks(range(n))
    ax.set_yticklabels([l for l, _, _ in layers], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Fraction of original residual variance remaining')
    ax.set_xlim(0, 1.15)
    ax.set_title('Residual sieve: progressive orthogonal peeling',
                 fontweight='bold')
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
# 30. Order-decomposition stacked bar
# ============================================================================

def plot_order_decomposition(
    sobol_results: Dict,
    variable_names: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (10, 5.5),
) -> Tuple[plt.Figure, np.ndarray]:
    """Two-panel order decomposition of total explained variance.

    Left: Single stacked bar — total variance split into 1st/2nd/3rd/residual.
    Right: Per-variable stacked bar — each variable's total-order contribution
        decomposed by the interaction orders it participates in.
    """
    apply_style()
    first = sobol_results['mean_sobol']['first_order']
    second = sobol_results['mean_sobol'].get('second_order', {})
    third = sobol_results['mean_sobol'].get('third_order', {})
    total_order = sobol_results['mean_sobol'].get('total_order', {})
    residual_frac = sobol_results['mean_sobol'].get('residual', 0)
    D = len(first)
    if variable_names is None:
        variable_names = [f"$x_{{{i+1}}}$" for i in range(D)]

    fig, axes = plt.subplots(1, 2, figsize=figsize,
                             gridspec_kw={'width_ratios': [0.35, 1]})

    # --- Left: aggregate order bar ---
    ax = axes[0]
    parts = [
        (sum(first.values()), '1st order', PALETTE['order1']),
        (sum(second.values()), '2nd order', PALETTE['order2']),
        (sum(third.values()), '3rd order', PALETTE['order3']),
        (residual_frac, 'Residual', PALETTE['residual']),
    ]
    parts = [(v, l, c) for v, l, c in parts if v > 0.001]
    cumulative = 0
    for val, lbl, col in parts:
        ax.bar(0, val, bottom=cumulative, color=col, width=0.6,
               edgecolor='white', linewidth=1.5)
        if val > 0.02:
            ax.text(0, cumulative + val / 2, f'{lbl}\n{val:.1%}',
                    ha='center', va='center', fontsize=8, fontweight='bold',
                    color='white')
        cumulative += val
    ax.set_xlim(-0.5, 0.5); ax.set_xticks([])
    ax.set_ylim(0, max(cumulative * 1.05, 1.0))
    ax.set_ylabel('Fraction of total model variance')
    ax.set_title('(a) By order', loc='left', fontweight='bold')

    # --- Right: per-variable stacked by order ---
    ax = axes[1]
    order_vars = sorted(range(D),
                        key=lambda i: -total_order.get(i, first.get(i, 0)))
    n = len(order_vars)
    x_pos = np.arange(n)

    v1 = np.array([first.get(order_vars[r], 0) for r in range(n)])
    v2 = np.zeros(n)
    for r, vi in enumerate(order_vars):
        for (a, b), s in second.items():
            if a == vi or b == vi:
                v2[r] += s
    v3 = np.zeros(n)
    for r, vi in enumerate(order_vars):
        for key, s in third.items():
            if vi in key:
                v3[r] += s

    ax.bar(x_pos, v1, 0.7, color=PALETTE['order1'], label='1st order',
           edgecolor='white', linewidth=0.5)
    ax.bar(x_pos, v2, 0.7, bottom=v1, color=PALETTE['order2'],
           label='2nd order', edgecolor='white', linewidth=0.5)
    if np.any(v3 > 0.001):
        ax.bar(x_pos, v3, 0.7, bottom=v1 + v2, color=PALETTE['order3'],
               label='3rd order', edgecolor='white', linewidth=0.5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([variable_names[i] for i in order_vars],
                       rotation=45, ha='right')
    ax.set_ylabel('Sobol index (total-order contribution)')
    ax.set_title('(b) Per variable, by interaction order', loc='left',
                 fontweight='bold')
    ax.legend(fontsize=8)
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
