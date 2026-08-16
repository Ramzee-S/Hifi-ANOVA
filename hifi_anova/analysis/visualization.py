"""Visualization: component plots, Sobol bars, heatmaps."""

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict

from ..core.features import basis_size
from .._result_aliases import canonical_result_mapping as _canonical_result_mapping

# Qualitative per-variable palette (colour-blind safe), mirrors plots.VAR_COLORS.
VAR_COLORS = [
    '#3274A1', '#E1812C', '#3A923A', '#C03D3E',
    '#9372B2', '#845B53', '#D584BD', '#7F7F7F',
    '#BCBD22', '#17BECF', '#AEC7E8', '#FFBB78',
]


def plot_sobol_bars(sobol_results: dict, variable_names: Optional[list] = None,
                   title: str = "Sobol Sensitivity Indices",
                   save_path: Optional[str] = None):
    """Bar chart of first-order Sobol indices."""
    first_order = sobol_results['mean_sobol']['first_order']
    D = len(first_order)

    if variable_names is None:
        variable_names = [f"x{i+1}" for i in range(D)]

    indices = [first_order[i] for i in range(D)]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(variable_names, indices, color='steelblue', alpha=0.8)
    ax.set_ylabel("First-Order Sobol Index")
    ax.set_title(title)
    ax.set_ylim(0, max(indices) * 1.2 if max(indices) > 0 else 1.0)

    # Add second-order info if available
    if sobol_results['mean_sobol'].get('second_order'):
        total_order = sobol_results['mean_sobol'].get('total_order', {})
        if total_order:
            total_indices = [total_order.get(i, indices[i]) for i in range(D)]
            ax.bar(variable_names, total_indices, alpha=0.3, color='orange',
                   label='Total Order')
            ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_dual_sobol(sobol_results: dict, variable_names: Optional[list] = None,
                    save_path: Optional[str] = None):
    """Dual spectrum: mean sensitivity and log-variance index side by side."""
    sobol_results = _canonical_result_mapping(
        sobol_results, warn_legacy=True)
    if 'log_variance_sobol' not in sobol_results:
        return plot_sobol_bars(sobol_results, variable_names, save_path=save_path)

    mean_first = sobol_results['mean_sobol']['first_order']
    var_first = sobol_results['log_variance_sobol']['first_order']
    D = len(mean_first)

    if variable_names is None:
        variable_names = [f"x{i+1}" for i in range(D)]

    x_pos = np.arange(D)
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x_pos - width/2, [mean_first[i] for i in range(D)],
           width, label='Mean Sobol', color='steelblue', alpha=0.8)
    ax.bar(x_pos + width/2, [var_first[i] for i in range(D)],
           width, label=r'Log-variance index $S^h$', color='coral', alpha=0.8)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(variable_names)
    ax.set_ylabel("Sobol Index")
    ax.set_title("Dual spectrum (mean and log-variance index)")
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_component_functions(model, variable_indices: list,
                            variable_names: Optional[list] = None,
                            save_path: Optional[str] = None):
    """Plot individual first-order component functions."""
    from ..core.features import build_per_variable_basis

    K1 = model.K1
    n_vars = len(variable_indices)
    _bn = getattr(model, 'basis_name', 'fourier')
    _il1 = getattr(model, 'include_linear_1', True)

    if variable_names is None:
        variable_names = [f"x{i+1}" for i in variable_indices]

    # Evaluate on grid
    x_grid = jnp.linspace(0, 1, 200)

    fig, axes = plt.subplots(1, n_vars, figsize=(4 * n_vars, 3))
    if n_vars == 1:
        axes = [axes]

    for idx, (var_i, name) in enumerate(zip(variable_indices, variable_names)):
        # Build basis for this variable
        x_1d = x_grid[:, None]  # (200, 1)
        basis = build_per_variable_basis(x_1d, K1, include_linear=_il1, basis_name=_bn)
        basis_i = basis[:, 0, :]

        # Get coefficients for this variable
        wi = model.mean_model.get_coefficients_for_variable(var_i)

        # Component function value
        f_i = basis_i @ wi

        axes[idx].plot(np.array(x_grid), np.array(f_i), 'b-', linewidth=2)
        axes[idx].axhline(0, color='gray', linestyle='--', alpha=0.5)
        axes[idx].set_xlabel(name)
        axes[idx].set_ylabel(f"f_{var_i+1}({name})")
        axes[idx].set_title(f"Component: {name}")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_interaction_heatmap(model, pair_index: int,
                            pair_manager=None,
                            save_path: Optional[str] = None):
    """Heatmap of a second-order interaction component."""
    from ..core.features import build_per_variable_basis

    K2 = model.K2
    if K2 == 0:
        return None

    # Get pair variables
    if pair_manager is not None:
        i, j = pair_manager.pair_to_variables(pair_index)
    else:
        i = int(model.pair_indices[pair_index, 0])
        j = int(model.pair_indices[pair_index, 1])

    # Get coefficients. For a term-structure model K2 holds max(pair_k2), so
    # this pair's block/basis must use its own order — else reshape and the
    # basis width mismatch the (smaller) coefficient vector on the ragged layout.
    _bn = getattr(model, 'basis_name', 'fourier')
    _il2 = getattr(model, 'include_linear_2', True)
    _pair_k2 = getattr(model, 'pair_k2', None)
    K2_p = int(_pair_k2[pair_index]) if _pair_k2 is not None else K2
    wp = model.mean_model.get_coefficients_for_pair(pair_index)
    block = basis_size(K2_p, _il2, _bn)
    W = wp.reshape(block, block)

    # Evaluate on 2D grid
    n_grid = 50
    x_grid = jnp.linspace(0, 1, n_grid)
    x_1d = x_grid[:, None]
    basis = build_per_variable_basis(x_1d, K2_p, include_linear=_il2, basis_name=_bn)[:, 0, :]

    # f_ij(xi, xj) = basis_i^T W basis_j
    Z = np.array(basis @ W @ basis.T)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(Z, origin='lower', extent=[0, 1, 0, 1],
                   cmap='RdBu_r', aspect='equal')
    plt.colorbar(im, ax=ax)
    ax.set_xlabel(f"x{j+1}")
    ax.set_ylabel(f"x{i+1}")
    ax.set_title(f"Interaction f_{{{i+1},{j+1}}}")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_sensitivity_ellipses(sobol_results: dict,
                              variable_names: Optional[list] = None,
                              mode: str = 'glyph',
                              mean_ci: Optional[Dict] = None,
                              var_ci: Optional[Dict] = None,
                              use: str = 'first_order',
                              top_k: Optional[int] = None,
                              ci_scale: float = 1.0,
                              title: Optional[str] = None,
                              save_path: Optional[str] = None):
    """Visualize the dual mean/log-variance Sobol spectrum as ellipses.

    Two complementary views of the same idea — that every variable carries a
    *pair* of sensitivities, one for the mean E[y|x] and one for the fitted
    log-residual scale h(x):

    ``mode='glyph'`` (default) — one ellipse per variable, its **width**
        proportional to the mean sensitivity ``S_i^f`` and its **height**
        proportional to the log-variance index ``S_i^h``. The *shape* is the
        message: a wide, flat ellipse is a mean driver; a tall, narrow ellipse
        is a multiplicative residual-scale driver; a circle is a balanced
        dual-role variable. This
        reads the whole spectrum at a glance without axes to trace.

    ``mode='plane'`` — a quantitative scatter placing each variable at
        ``(S_i^f, S_i^h)``. Bottom-right = mean drivers, top-left = hidden
        log-residual-scale drivers, top-right = dual-role. When
        ``mean_ci``/``var_ci``
        (dicts ``{i: (lo, hi)}``) are supplied, each point becomes an ellipse
        whose semi-axes are the CI half-widths (optionally magnified by
        ``ci_scale`` for visibility) — a joint uncertainty region.

    Args:
        sobol_results: output of ``compute_sobol_indices`` (needs
            ``log_variance_sobol`` for the log-variance axis; falls back to 0
            otherwise).
        variable_names: labels; default x1, x2, ...
        mode: 'glyph' or 'plane'.
        mean_ci, var_ci: optional ``{i: (lo, hi)}`` CIs (used in 'plane' mode).
        use: 'first_order' or 'total_order'.
        top_k: show only the k variables with largest S^f + S^h.
        ci_scale: magnification for the CI ellipses in 'plane' mode (stated in
            the legend); 1.0 = true 95% CI.
        title, save_path: cosmetic / output.

    Returns:
        matplotlib Figure.
    """
    from matplotlib.patches import Ellipse

    sobol_results = _canonical_result_mapping(
        sobol_results, warn_legacy=True)
    mean_first = sobol_results['mean_sobol'][use]
    D = len(mean_first)
    if variable_names is None:
        variable_names = [f"$x_{{{i+1}}}$" for i in range(D)]

    has_var = 'log_variance_sobol' in sobol_results
    var_first = (sobol_results['log_variance_sobol'][use] if has_var
                 else {i: 0.0 for i in range(D)})

    idx = sorted(range(D), key=lambda i: float(mean_first.get(i, 0))
                 + float(var_first.get(i, 0)), reverse=True)
    if top_k is not None:
        idx = idx[:top_k]

    lvl = 'first-order' if use == 'first_order' else 'total-order'

    if mode == 'plane':
        return _ellipse_plane(idx, mean_first, var_first, variable_names,
                              mean_ci, var_ci, ci_scale, lvl, title, save_path)

    # ---- glyph mode: width = mean sensitivity, height = log-variance index ---
    vals = [max(float(mean_first.get(i, 0)), float(var_first.get(i, 0)))
            for i in idx]
    max_val = max(vals) if vals else 1.0
    max_val = max(max_val, 1e-6)
    radius_max = 0.42          # largest semi-axis, in slot units
    scale = radius_max / max_val
    min_semi = 0.02            # keep a near-zero axis visible as a sliver

    fig, ax = plt.subplots(figsize=(1.7 * len(idx) + 1.5, 4.2))
    for slot, i in enumerate(idx):
        sf = float(mean_first.get(i, 0.0))
        sh = float(var_first.get(i, 0.0))
        color = VAR_COLORS[i % len(VAR_COLORS)]
        rx = max(scale * sf, min_semi)
        ry = max(scale * sh, min_semi)
        ax.add_patch(Ellipse((slot, 0), width=2 * rx, height=2 * ry,
                             facecolor=color, edgecolor=color,
                             alpha=0.45, linewidth=1.6))
        ax.annotate(variable_names[i], (slot, -radius_max - 0.14),
                    ha='center', va='top', fontsize=11, color=color,
                    fontweight='bold')
        ax.annotate(f"$S^f$={sf:.2f}\n$S^h$={sh:.2f}",
                    (slot, radius_max + 0.06), ha='center', va='bottom',
                    fontsize=7.5, color='#555555')

    ax.set_xlim(-0.7, len(idx) - 0.3)
    ax.set_ylim(-radius_max - 0.5, radius_max + 0.5)
    ax.set_aspect('equal')
    ax.axhline(0, color='#DDDDDD', lw=0.8, zorder=0)
    ax.set_yticks([])
    ax.set_xticks([])
    for sp in ('left', 'right', 'top', 'bottom'):
        ax.spines[sp].set_visible(False)
    ax.set_title(title or 'Dual-sensitivity glyphs (mean vs log variance)')
    ax.text(0.5, -0.02,
            r'width $\propto$ mean sensitivity $S^f$'
            r'    ·    height $\propto$ log-variance index $S^h$',
            transform=ax.transAxes, ha='center', va='top', fontsize=9,
            color='#666666')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def _ellipse_plane(idx, mean_first, var_first, variable_names,
                   mean_ci, var_ci, ci_scale, lvl, title, save_path):
    """Quantitative (S^f, S^h) plane; ellipses = CI regions if CIs are given."""
    from matplotlib.patches import Ellipse

    min_radius = 0.010

    def _half(ci, i):
        if ci is not None and i in ci:
            lo, hi = ci[i]
            return max((float(hi) - float(lo)) / 2.0 * ci_scale, min_radius)
        return min_radius

    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    max_val = 0.0
    for i in idx:
        mx, vy = float(mean_first.get(i, 0.0)), float(var_first.get(i, 0.0))
        max_val = max(max_val, mx, vy)
        color = VAR_COLORS[i % len(VAR_COLORS)]
        ax.add_patch(Ellipse((mx, vy), width=2 * _half(mean_ci, i),
                             height=2 * _half(var_ci, i), facecolor=color,
                             edgecolor=color, alpha=0.35, linewidth=1.4, zorder=2))
        ax.plot(mx, vy, 'o', color=color, markersize=4, zorder=3)
        ax.annotate(variable_names[i], (mx, vy), textcoords='offset points',
                    xytext=(7, 6), fontsize=10, color=color,
                    fontweight='bold', zorder=4)

    hi = max(max_val * 1.2, 0.1)
    lbl = 'equal influence' if ci_scale == 1.0 else f'equal (CI ×{ci_scale:g})'
    ax.plot([0, hi], [0, hi], '--', color='#AAAAAA', lw=1, zorder=1, label=lbl)
    ax.text(0.97 * hi, 0.05 * hi, 'mean\ndrivers', ha='right', va='bottom',
            fontsize=8, color='#999999', style='italic')
    ax.text(0.03 * hi, 0.97 * hi, 'hidden log-variance\ndrivers', ha='left',
            va='top', fontsize=8, color='#999999', style='italic')
    ax.text(0.97 * hi, 0.97 * hi, 'dual-role', ha='right', va='top',
            fontsize=8, color='#999999', style='italic')

    ax.set_xlim(-0.02 * hi, hi)
    ax.set_ylim(-0.02 * hi, hi)
    ax.set_aspect('equal')
    ax.set_xlabel(f'Mean {lvl} Sobol  $S_i^f$')
    ax.set_ylabel(f'Log-variance {lvl} index  $S_i^h$')
    ax.set_title(title or 'Dual-sensitivity plane (mean vs log variance)')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig
