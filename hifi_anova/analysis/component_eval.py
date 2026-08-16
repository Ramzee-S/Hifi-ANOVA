"""Evaluate individual Hoeffding components at arbitrary points.

Given a fitted model, evaluate the learned component functions:
  - f_i(x_i): first-order effect of variable i
  - f_ij(x_i, x_j): second-order interaction of pair (i,j)
  - f_ijk(x_i, x_j, x_k): third-order interaction of triple (i,j,k)

These are the actual learned shapes — the coefficients w times the basis
evaluated at given points. Useful for partial dependence plots, effect
interpretation, and comparison with ground truth on synthetic data.

Usage:
    from hifi_anova.analysis.component_eval import (
        evaluate_first_order, evaluate_second_order,
        evaluate_all_first_order, first_order_on_grid,
    )

    # Single variable on a grid
    x_grid, f_vals = first_order_on_grid(model, variable=0, n_points=200)

    # All first-order components on test data
    components = evaluate_all_first_order(model, x_data)
    # components[i] is (N,) array of f_i(x_i) values

    # Second-order interaction surface
    xi_grid, xj_grid, f_ij = second_order_on_grid(model, pair_index=0)
"""

from ..array_backend import xp as jnp  # switchable array backend (numpy exact core)
import numpy as np
from typing import Dict, Tuple

from ..core.features import build_per_variable_basis


def evaluate_first_order(
    model,
    variable: int,
    x_values: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate the first-order component f_i(x_i) at given points.

    Args:
        model: fitted HiFiANOVA
        variable: variable index i
        x_values: (N,) values of x_i in [0,1]

    Returns:
        (N,) array of f_i(x_i) values
    """
    mm = model.mean_model
    wi = mm.get_coefficients_for_variable(variable)

    # Build per-variable basis at the given points
    # Need to create a (N, 1) input to get basis for one variable
    x_2d = jnp.asarray(x_values)[:, None]  # (N, 1)

    if mm.var_specs is not None:
        bn, K, il, _, _ = mm.var_specs[variable]
    else:
        bn = mm.basis_name
        K = mm.K1
        il = mm.include_linear_1

    basis = build_per_variable_basis(x_2d, K, include_linear=il, basis_name=bn)
    # basis shape: (N, 1, B) — squeeze the variable dimension
    phi_i = basis[:, 0, :]  # (N, B)

    return phi_i @ wi


def evaluate_second_order(
    model,
    pair_index: int,
    x_i_values: jnp.ndarray,
    x_j_values: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate the second-order component f_ij(x_i, x_j) at given points.

    Args:
        model: fitted HiFiANOVA
        pair_index: index into model.pair_indices
        x_i_values: (N,) values of x_i in [0,1]
        x_j_values: (N,) values of x_j in [0,1]

    Returns:
        (N,) array of f_ij(x_i, x_j) values
    """
    mm = model.mean_model
    wij = mm.get_coefficients_for_pair(pair_index)

    if mm.pair_block_info is not None:
        vi, vj, Bi, Bj, _, _ = mm.pair_block_info[pair_index]
        if mm.pair_k2 is not None:
            # Per-pair K2 (uniform basis family): both sides of the block were
            # built at the PAIR's own order (build_second_order_features_per_pair),
            # not the variables' first-order specs.
            bni = bnj = mm.basis_name
            Ki = Kj = int(mm.pair_k2[pair_index])
            ili = ilj = mm.include_linear_2
        elif mm.var_specs is not None:
            # Mixed per-variable bases: each side uses its variable's spec.
            bni, Ki, ili, _, _ = mm.var_specs[vi]
            bnj, Kj, ilj, _, _ = mm.var_specs[vj]
        else:
            # Uniform scalar K2 with block info present.
            bni = bnj = mm.basis_name
            Ki = Kj = mm.K2
            ili = ilj = mm.include_linear_2
    else:
        vi, vj = int(model.pair_indices[pair_index, 0]), int(model.pair_indices[pair_index, 1])
        bni = bnj = mm.basis_name
        Ki = Kj = mm.K2
        ili = ilj = mm.include_linear_2

    N = len(x_i_values)
    xi_2d = jnp.asarray(x_i_values)[:, None]
    xj_2d = jnp.asarray(x_j_values)[:, None]

    basis_i = build_per_variable_basis(xi_2d, Ki, include_linear=ili, basis_name=bni)[:, 0, :]
    basis_j = build_per_variable_basis(xj_2d, Kj, include_linear=ilj, basis_name=bnj)[:, 0, :]

    # Outer product per sample
    products = basis_i[:, :, None] * basis_j[:, None, :]  # (N, Bi, Bj)
    phi_ij = products.reshape(N, -1)  # (N, Bi*Bj)

    return phi_ij @ wij


def evaluate_all_first_order(
    model,
    x_data: jnp.ndarray,
) -> Dict[int, jnp.ndarray]:
    """Evaluate all first-order components on data.

    Args:
        model: fitted HiFiANOVA
        x_data: (N, D) input data in [0,1]

    Returns:
        dict {variable_index: (N,) array of f_i(x_i) values}
    """
    D = model.D
    components = {}
    for i in range(D):
        components[i] = evaluate_first_order(model, i, x_data[:, i])
    return components


def first_order_on_grid(
    model,
    variable: int,
    n_points: int = 200,
    x_range: Tuple[float, float] = (0.0, 1.0),
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate first-order component on an evenly-spaced grid.

    Args:
        model: fitted HiFiANOVA
        variable: variable index
        n_points: number of grid points
        x_range: (min, max) range in [0,1]

    Returns:
        (x_grid, f_values) — both (n_points,) numpy arrays
    """
    x_grid = jnp.linspace(x_range[0], x_range[1], n_points)
    f_values = evaluate_first_order(model, variable, x_grid)
    return np.asarray(x_grid), np.asarray(f_values)


def second_order_on_grid(
    model,
    pair_index: int,
    n_points: int = 50,
    x_range: Tuple[float, float] = (0.0, 1.0),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate second-order interaction on a 2D grid.

    Args:
        model: fitted HiFiANOVA
        pair_index: index into model.pair_indices
        n_points: grid points per axis
        x_range: (min, max) range

    Returns:
        (xi_grid, xj_grid, f_surface) where:
          xi_grid: (n_points,) x-axis values
          xj_grid: (n_points,) y-axis values
          f_surface: (n_points, n_points) interaction surface
    """
    xi_1d = jnp.linspace(x_range[0], x_range[1], n_points)
    xj_1d = jnp.linspace(x_range[0], x_range[1], n_points)

    # Create meshgrid
    xi_mesh, xj_mesh = jnp.meshgrid(xi_1d, xj_1d, indexing='ij')
    xi_flat = xi_mesh.ravel()
    xj_flat = xj_mesh.ravel()

    f_flat = evaluate_second_order(model, pair_index, xi_flat, xj_flat)
    f_surface = np.asarray(f_flat).reshape(n_points, n_points)

    return np.asarray(xi_1d), np.asarray(xj_1d), f_surface


def frequency_decomposition(
    model,
    variable: int,
) -> Dict[str, float]:
    """Decompose a variable's Sobol index by frequency content.

    Returns the fraction of the variable's variance attributable to
    each basis function (linear, cos1, sin1, cos2, sin2, ...).

    Args:
        model: fitted HiFiANOVA
        variable: variable index

    Returns:
        dict {'linear': 0.4, 'cos1': 0.3, 'sin1': 0.2, ...}
        with values summing to ~1.0 (the variable's share, normalized)
    """
    mm = model.mean_model
    wi = np.asarray(mm.get_coefficients_for_variable(variable))
    Gi = np.asarray(mm.get_var_gram(variable))

    if mm.var_specs is not None:
        bn, K, il, _, _ = mm.var_specs[variable]
    else:
        bn = mm.basis_name
        K = mm.K1
        il = mm.include_linear_1

    # Total variance for this variable
    total_var = float(wi @ Gi @ wi)
    if total_var < 1e-15:
        return {}

    result = {}
    if bn == 'fourier':
        idx = 0
        if il:
            # Linear term: diagonal contribution w[0]^2 * G[0,0]
            linear_var = float(wi[0] ** 2 * Gi[0, 0])
            # Distribute linear-sine cross-terms symmetrically: half to linear, half to sin
            cross_total = 0.0
            for k in range(1, K + 1):
                sin_idx_k = 1 + 2 * (k - 1) + 1  # offset 1 (linear) + 2*(k-1) cos + 1 sin
                if sin_idx_k < len(Gi):
                    cross_total += 2.0 * float(wi[0] * wi[sin_idx_k] * Gi[0, sin_idx_k])
            linear_var += cross_total / 2.0
            result['linear'] = linear_var
            idx = 1
        for k in range(1, K + 1):
            cos_idx = idx
            sin_idx = idx + 1
            cos_var = float(wi[cos_idx] ** 2 * Gi[cos_idx, cos_idx])
            sin_var = float(wi[sin_idx] ** 2 * Gi[sin_idx, sin_idx])
            # Add half of cross-term from linear to sin component
            if il and sin_idx < len(Gi):
                cross = 2.0 * float(wi[0] * wi[sin_idx] * Gi[0, sin_idx])
                sin_var += cross / 2.0
            result[f'cos{k}'] = cos_var / total_var
            result[f'sin{k}'] = sin_var / total_var
            idx += 2
        if il:
            result['linear'] = result.get('linear', 0) / total_var

    elif bn == 'legendre':
        for j in range(K):
            result[f'P{j+1}'] = float(wi[j] ** 2 * Gi[j, j]) / total_var

    elif bn == 'haar':
        idx = 0
        for j in range(1, K + 1):
            n_at_scale = 2 ** (j - 1)
            scale_var = sum(float(wi[idx + k] ** 2) for k in range(n_at_scale))
            result[f'scale{j}'] = scale_var / total_var
            idx += n_at_scale

    return result


def interaction_strength_matrix(model) -> np.ndarray:
    """Compute D×D matrix of pairwise interaction strengths.

    Returns S_ij Sobol indices for all pairs, including zeros
    for pairs not in the model. Diagonal is first-order Sobol.

    Args:
        model: fitted HiFiANOVA

    Returns:
        (D, D) numpy array. Symmetric. Diagonal = S_i, off-diagonal = S_ij.
    """
    from ..analysis.sobol import compute_sobol_indices

    sobol = compute_sobol_indices(model)
    ms = sobol['mean_sobol']
    D = model.D

    matrix = np.zeros((D, D))

    # Diagonal: first-order
    for i, s in ms['first_order'].items():
        matrix[i, i] = s

    # Off-diagonal: second-order
    for (i, j), s in ms.get('second_order', {}).items():
        matrix[i, j] = s
        matrix[j, i] = s

    return matrix
