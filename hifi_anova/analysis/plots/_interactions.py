"""HiFi-ANOVA plotting — interaction-structure visualizations.

Split from the original monolithic ``plots.py``; import via
``from hifi_anova.analysis.plots import ...`` as before.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple

from ._common import PALETTE, apply_style, _var_color


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
