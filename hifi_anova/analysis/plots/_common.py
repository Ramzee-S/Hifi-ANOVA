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
import matplotlib.pyplot as plt

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
