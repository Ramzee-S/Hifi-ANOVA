"""HiFi-ANOVA plotting — fit / calibration / residual / sieve diagnostics.

Split from the original monolithic ``plots.py``; import via
``from hifi_anova.analysis.plots import ...`` as before.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple

from ._common import PALETTE, apply_style, _ensure_np


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
    sigma = np.sqrt(max(analytics['sigma2_hat'], 1e-15))
    std_res = residuals / sigma

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
    ax.barh(y_pos, [v for _, v, _ in levels], height=0.65,
                   color=[c for _, _, c in levels], edgecolor='white',
                   linewidth=1)

    # Value labels
    for idx, (label, val, _) in enumerate(levels):
        if val > 0.005:
            ax.text(val + 0.01, idx, f'{val:.1%}', va='center', fontsize=9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([lbl for lbl, _, _ in levels])
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
        ax.set_yticklabels([lbl for lbl, _, _ in items], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel('Fraction of residual')
        ax.set_title('(b) Top missed interactions', loc='left',
                     fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No flagged\nmissed interactions',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=12, color=PALETTE['muted'])
        ax.set_title('(b) Top missed interactions', loc='left',
                     fontweight='bold')

    fig.tight_layout()
    return fig, axes


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
    ax.set_yticklabels([lbl for lbl, _, _ in layers], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Fraction of original residual variance remaining')
    ax.set_xlim(0, 1.15)
    ax.set_title('Residual sieve: progressive orthogonal peeling',
                 fontweight='bold')
    fig.tight_layout()
    return fig, ax


# ============================================================================
# Parity plot (predicted vs actual)
# ============================================================================

def plot_parity(actual, predicted,
                xlabel: str = 'Observed y',
                ylabel: str = 'Predicted mean',
                title: Optional[str] = None,
                color_by=None,
                color_label: Optional[str] = None,
                figsize: Tuple[float, float] = (6, 6)) -> Tuple[plt.Figure, plt.Axes]:
    """Predicted-vs-actual parity scatter with the 45-degree line and R^2.

    Works for any dataset. Note that against *noisy observations* the scatter
    cannot collapse to the line — its spread is the irreducible noise, so the R^2
    here is bounded by the noise floor. To judge how well the *mean* is
    recovered, pass a noiseless reference as `actual` when one is available (e.g.
    the true function for synthetic data).

    Args:
        actual: (N,) observed targets (or a noiseless reference).
        predicted: (N,) model mean predictions.
        color_by: optional (N,) values to color points by (e.g. true sigma(x)).
        color_label: colorbar label when color_by is given.

    Returns (fig, ax).
    """
    apply_style()
    a = np.asarray(actual, dtype=float).ravel()
    p = np.asarray(predicted, dtype=float).ravel()
    va = float(np.var(a))
    r2 = 1.0 - float(np.var(a - p)) / va if va > 0 else 0.0

    fig, ax = plt.subplots(figsize=figsize)
    if color_by is not None:
        sc = ax.scatter(a, p, s=8, alpha=0.3, c=np.asarray(color_by).ravel(),
                        cmap='viridis')
        cb = fig.colorbar(sc, ax=ax)
        if color_label:
            cb.set_label(color_label)
    else:
        ax.scatter(a, p, s=8, alpha=0.3, color=PALETTE['order1'])

    lo = min(a.min(), p.min())
    hi = max(a.max(), p.max())
    pad = 0.03 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], '--',
            color=PALETTE['residual'], lw=1.5, label='ideal (45$\\degree$)')
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect('equal')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title or f'Predicted vs actual  ($R^2$ = {r2:.3f})')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig, ax
