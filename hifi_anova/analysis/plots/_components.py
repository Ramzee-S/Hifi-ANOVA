"""HiFi-ANOVA plotting — learned component curves and per-variable content.

Split from the original monolithic ``plots.py``; import via
``from hifi_anova.analysis.plots import ...`` as before.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple

from ._common import PALETTE, apply_style, _var_color


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
    from ...core.features import build_per_variable_basis
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

    Fourier-only: the ``2K1+1`` linear/cos/sin block layout and the notion of
    "frequency content" are specific to the Fourier basis. For Legendre/Haar
    models raise a clear error rather than mislabel the coefficients.
    """
    from ...core.gram import build_gram_matrix
    apply_style()

    basis_name = getattr(model, 'basis_name', 'fourier')
    if basis_name != 'fourier':
        raise ValueError(
            f"plot_frequency_content requires a Fourier model; got "
            f"basis_name={basis_name!r}. Frequency content (linear/cos/sin "
            f"per-harmonic breakdown) is only defined for the Fourier basis. "
            f"Use plot_components() for a basis-agnostic component view."
        )

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

    # Annotate cells above the display threshold
    for (i, j), score in discovery_result.pair_scores.items():
        if score > 0.005:
            ax.text(j, i, f'{score:.3f}', ha='center', va='center',
                    fontsize=7, color='white' if score > vmax * 0.6 else 'black')

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
    from ...core.features import build_per_variable_basis
    from ..automl import sandwich_covariance, ridge_analytics
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

    from ...core.features import basis_size
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
    def ternary_to_xy(p, o, loc):
        total = p + o + loc
        if total < 1e-15:
            return 0.33, 0.33 * np.sqrt(3) / 2
        p, o, loc = p / total, o / total, loc / total
        x = o + loc * 0.5
        y = loc * np.sqrt(3) / 2
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
        loc = info.get('local_fraction', 0)
        share = info.get('share_of_total', 0.01)
        char = info.get('character', 'mixed')

        x, y = ternary_to_xy(p, o, loc)
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
