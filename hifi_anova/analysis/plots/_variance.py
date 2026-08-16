"""HiFi-ANOVA plotting — hierarchical variance-decomposition views.

Split from the original monolithic ``plots.py``; import via
``from hifi_anova.analysis.plots import ...`` as before.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple

from ._common import PALETTE, apply_style


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
    for v, lbl, c in zip(order_vals, order_labels, order_colors):
        if v > 0.001:
            inner_vals.append(v)
            inner_labels.append(lbl)
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
    [c + 'CC' if len(c) == 7 else c
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

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax.set_title('Variance treemap: area $\\propto$ Sobol index',
                 fontweight='bold', pad=10)
    fig.tight_layout()
    return fig, ax
