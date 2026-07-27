"""Visualization: component plots, Sobol bars, heatmaps."""

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict

from ..core.features import basis_size


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
    bars = ax.bar(variable_names, indices, color='steelblue', alpha=0.8)
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
    """Dual Sobol spectrum: mean and variance indices side by side."""
    if 'variance_sobol' not in sobol_results:
        return plot_sobol_bars(sobol_results, variable_names, save_path=save_path)

    mean_first = sobol_results['mean_sobol']['first_order']
    var_first = sobol_results['variance_sobol']['first_order']
    D = len(mean_first)

    if variable_names is None:
        variable_names = [f"x{i+1}" for i in range(D)]

    x_pos = np.arange(D)
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x_pos - width/2, [mean_first[i] for i in range(D)],
           width, label='Mean Sobol', color='steelblue', alpha=0.8)
    ax.bar(x_pos + width/2, [var_first[i] for i in range(D)],
           width, label='Variance Sobol', color='coral', alpha=0.8)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(variable_names)
    ax.set_ylabel("Sobol Index")
    ax.set_title("Dual Sobol Spectrum (Mean & Variance)")
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
    from ..core.gram import build_gram_matrix

    K1 = model.K1
    D = model.D
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

    # Get coefficients
    _bn = getattr(model, 'basis_name', 'fourier')
    _il2 = getattr(model, 'include_linear_2', True)
    wp = model.mean_model.get_coefficients_for_pair(pair_index)
    block = basis_size(K2, _il2, _bn)
    W = wp.reshape(block, block)

    # Evaluate on 2D grid
    n_grid = 50
    x_grid = jnp.linspace(0, 1, n_grid)
    x_1d = x_grid[:, None]
    basis = build_per_variable_basis(x_1d, K2, include_linear=_il2, basis_name=_bn)[:, 0, :]

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
