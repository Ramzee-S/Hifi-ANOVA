"""HiFi-ANOVA plotting — Sobol sensitivity spectra and sensitivity glyphs.

Split from the original monolithic ``plots.py``; import via
``from hifi_anova.analysis.plots import ...`` as before.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple

from ._common import PALETTE, apply_style, _var_color


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
