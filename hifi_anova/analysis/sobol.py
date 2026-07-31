"""Analytic and correlative Sobol indices.

Two types of indices are computed:
  - Structural (analytic): w^T G w, assuming independent inputs.
    Sum to 1. Characterize the function's intrinsic sensitivity.
  - Correlative (empirical): based on covariance of component outputs
    on actual data. Account for input correlations. May not sum to 1.

For independent inputs, both types agree.
The divergence between them diagnoses the impact of input correlations.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Optional

from ..core.gram import build_gram_matrix, build_gram_matrix_2d, build_gram_matrix_3d
from ..core.features import basis_size


def _block_variances(w, gram, n_blocks: int, block: int) -> np.ndarray:
    """Batched per-block variances ``max(0, w_b^T G w_b)`` for contiguous blocks.

    Replaces a Python loop of ``n_blocks`` tiny JAX quadratic forms (each of which
    forced a device sync via ``float(...)``) with a single numpy einsum. Numerics
    are identical to the per-block ``w_b @ G @ w_b`` up to float64 round-off.
    """
    W = np.asarray(w, dtype=np.float64).reshape(n_blocks, block)
    G = np.asarray(gram, dtype=np.float64)
    return np.maximum(0.0, np.einsum('bi,ij,bj->b', W, G, W))


def compute_sobol_indices(model, x_data: Optional[jnp.ndarray] = None) -> dict:
    """Compute the full dual Sobol spectrum.

    Args:
        model: HiFiANOVA instance
        x_data: (N, D) data for empirical NN variance (optional)

    Returns:
        dict with mean_sobol, variance_sobol, variance_accounting
    """
    G1 = jnp.asarray(model.G1, dtype=jnp.float64) if model.G1 is not None else None
    D = model.D
    K1 = model.K1
    K2 = model.K2

    results = {}

    # --- Mean Sobol indices ---
    mean_sobol = {'first_order': {}, 'second_order': {}, 'third_order': {},
                  'total_order': {}}

    # First-order variances — handle both mixed and uniform modes
    _bn = getattr(model, 'basis_name', 'fourier')
    _il1 = getattr(model, 'include_linear_1', True)
    _is_mixed = getattr(model, 'var_specs', None) is not None
    w1 = jnp.asarray(model.mean_model.w1, dtype=jnp.float64)
    first_order_vars = {}

    if _is_mixed:
        # Mixed mode: per-variable Gram and block sizes
        for i in range(D):
            wi = jnp.asarray(model.mean_model.get_coefficients_for_variable(i),
                             dtype=jnp.float64)
            Gi = jnp.asarray(model.mean_model.get_var_gram(i), dtype=jnp.float64)
            var_i = jnp.maximum(0.0, wi @ Gi @ wi)
            first_order_vars[i] = float(var_i)
    else:
        block1 = basis_size(K1, _il1, _bn)
        v1 = _block_variances(w1, G1, D, block1)
        for i in range(D):
            first_order_vars[i] = float(v1[i])

    # Second-order variances — handle mixed pairs (G_i ⊗ G_j per pair)
    second_order_vars = {}
    _has_pairs = model.pair_indices is not None and (
        K2 > 0 or _is_mixed)
    if _has_pairs and model.pair_indices is not None:
        w2 = jnp.asarray(model.mean_model.w2, dtype=jnp.float64)
        P = model.pair_indices.shape[0]

        if _is_mixed and model.mean_model.pair_block_info is not None:
            # Mixed: per-pair Gram via get_pair_gram
            for p in range(P):
                wp = jnp.asarray(model.mean_model.get_coefficients_for_pair(p),
                                 dtype=jnp.float64)
                Gp = jnp.asarray(model.mean_model.get_pair_gram(p),
                                 dtype=jnp.float64)
                var_p = jnp.maximum(0.0, wp @ Gp @ wp)
                i, j = int(model.pair_indices[p, 0]), int(model.pair_indices[p, 1])
                second_order_vars[(i, j)] = float(var_p)
        elif K2 > 0:
            incl_lin_2 = getattr(model, 'include_linear_2', True)
            G2 = jnp.asarray(model.G2, dtype=jnp.float64) if model.G2 is not None else build_gram_matrix_2d(build_gram_matrix(K2, incl_lin_2, _bn))
            G2 = jnp.asarray(G2, dtype=jnp.float64)
            block2 = basis_size(K2, incl_lin_2, _bn) ** 2

            v2 = _block_variances(w2, G2, P, block2)
            for p in range(P):
                i, j = int(model.pair_indices[p, 0]), int(model.pair_indices[p, 1])
                second_order_vars[(i, j)] = float(v2[p])

    # Third-order variances
    third_order_vars = {}
    K3 = getattr(model, 'K3', 0)
    if K3 > 0 and model.triple_indices is not None:
        incl_lin_3 = getattr(model, 'include_linear_3', True)
        G3 = jnp.asarray(model.G3, dtype=jnp.float64) if model.G3 is not None else build_gram_matrix_3d(build_gram_matrix(K3, incl_lin_3, _bn))
        G3 = jnp.asarray(G3, dtype=jnp.float64)
        w3 = jnp.asarray(model.mean_model.w3, dtype=jnp.float64)
        block3 = basis_size(K3, incl_lin_3, _bn) ** 3
        T = model.triple_indices.shape[0]

        v3 = _block_variances(w3, G3, T, block3)
        for t in range(T):
            i, j, k = (int(model.triple_indices[t, l]) for l in range(3))
            third_order_vars[(i, j, k)] = float(v3[t])

    # Residual variance (empirical — works for both NN and linear residuals)
    residual_var = 0.0
    if model.residual_net is not None and x_data is not None:
        res_pred = jax.vmap(model.residual_net)(x_data)
        if res_pred.ndim > 1:
            res_pred = res_pred.squeeze(-1)
        residual_var = float(jnp.var(res_pred))

    # Total variance
    total_var = (sum(first_order_vars.values()) +
                 sum(second_order_vars.values()) +
                 sum(third_order_vars.values()) +
                 residual_var)

    if total_var > 0:
        for i, v in first_order_vars.items():
            mean_sobol['first_order'][i] = v / total_var
        for (i, j), v in second_order_vars.items():
            mean_sobol['second_order'][(i, j)] = v / total_var
        for (i, j, k), v in third_order_vars.items():
            mean_sobol['third_order'][(i, j, k)] = v / total_var
        mean_sobol['residual'] = residual_var / total_var
    else:
        for i in range(D):
            mean_sobol['first_order'][i] = 0.0
        mean_sobol['residual'] = 0.0

    # Total-order indices (first + second + third order involving variable i)
    for i in range(D):
        total_i = first_order_vars.get(i, 0.0)
        for (a, b), v in second_order_vars.items():
            if a == i or b == i:
                total_i += v
        for key, v in third_order_vars.items():
            if i in key:
                total_i += v
        mean_sobol['total_order'][i] = total_i / total_var if total_var > 0 else 0.0

    results['mean_sobol'] = mean_sobol

    # --- Variance Sobol indices (if heteroscedastic) ---
    if model.variance_model is not None:
        vm = model.variance_model
        Kh = model.Kh
        _ilh1 = getattr(model, 'include_linear_h1', getattr(vm, 'include_linear_h1', True))
        _ilh2 = getattr(model, 'include_linear_h2', getattr(vm, 'include_linear_h2', True))
        _ilh3 = getattr(model, 'include_linear_h3', getattr(vm, 'include_linear_h3', True))
        _bn_h = getattr(vm, 'basis_name', _bn)
        Gh = build_gram_matrix(Kh, _ilh1, _bn_h)
        Gh = jnp.asarray(Gh, dtype=jnp.float64)
        wh = jnp.asarray(vm.w1, dtype=jnp.float64)
        block_h = basis_size(Kh, _ilh1, _bn_h)

        variance_sobol = {'first_order': {}, 'second_order': {}, 'third_order': {},
                          'total_order': {}}

        # First-order variance
        var_h_first = {}
        vh1 = _block_variances(wh, Gh, D, block_h)
        for i in range(D):
            var_h_first[i] = float(vh1[i])

        # Second-order variance (if present)
        var_h_second = {}
        K2h = getattr(vm, 'K2h', 0)
        if K2h > 0 and hasattr(vm, 'w2') and len(vm.w2) > 0:
            G2h = build_gram_matrix_2d(build_gram_matrix(K2h, _ilh2, _bn_h))
            G2h = jnp.asarray(G2h, dtype=jnp.float64)
            w2h = jnp.asarray(vm.w2, dtype=jnp.float64)
            block_h2 = basis_size(K2h, _ilh2, _bn_h) ** 2
            pair_idx_h = vm.pair_indices_h
            if pair_idx_h is not None:
                Ph = pair_idx_h.shape[0]
                vh2 = _block_variances(w2h, G2h, Ph, block_h2)
                for p in range(Ph):
                    i, j = int(pair_idx_h[p, 0]), int(pair_idx_h[p, 1])
                    var_h_second[(i, j)] = float(vh2[p])

        # Third-order variance (if present)
        var_h_third = {}
        K3h = getattr(vm, 'K3h', 0)
        if K3h > 0 and hasattr(vm, 'w3') and len(vm.w3) > 0:
            G3h = build_gram_matrix_3d(build_gram_matrix(K3h, _ilh3, _bn_h))
            G3h = jnp.asarray(G3h, dtype=jnp.float64)
            w3h = jnp.asarray(vm.w3, dtype=jnp.float64)
            block_h3 = basis_size(K3h, _ilh3, _bn_h) ** 3
            triple_idx_h = getattr(vm, 'triple_indices_h', None)
            if triple_idx_h is not None:
                Th = triple_idx_h.shape[0]
                vh3 = _block_variances(w3h, G3h, Th, block_h3)
                for t in range(Th):
                    i, j, k = (int(triple_idx_h[t, l]) for l in range(3))
                    var_h_third[(i, j, k)] = float(vh3[t])

        # Variance residual contribution (empirical, if present)
        var_h_residual = 0.0
        if (hasattr(vm, 'has_variance_residual') and vm.has_variance_residual
                and x_data is not None):
            psi1_data = model.build_psi1(x_data)
            z_h = vm.variance_residual.build_features(x_data)
            if (vm.variance_residual.proj_coeffs.ndim >= 2 and
                    vm.variance_residual.proj_coeffs.shape[0] > 0):
                z_h_proj = z_h - psi1_data @ vm.variance_residual.proj_coeffs
            else:
                z_h_proj = z_h
            h_res = z_h_proj @ vm.w_var_residual
            var_h_residual = float(jnp.var(h_res))

        total_var_h = (sum(var_h_first.values()) +
                       sum(var_h_second.values()) +
                       sum(var_h_third.values()) +
                       var_h_residual)

        if total_var_h > 0:
            for i, v in var_h_first.items():
                variance_sobol['first_order'][i] = v / total_var_h
            for (i, j), v in var_h_second.items():
                variance_sobol['second_order'][(i, j)] = v / total_var_h
            for (i, j, k), v in var_h_third.items():
                variance_sobol['third_order'][(i, j, k)] = v / total_var_h
            variance_sobol['residual'] = var_h_residual / total_var_h
        else:
            for i in range(D):
                variance_sobol['first_order'][i] = 0.0
            variance_sobol['residual'] = 0.0

        # Total-order variance Sobol (first + second + third order involving variable i)
        for i in range(D):
            total_i = var_h_first.get(i, 0.0)
            for (a, b), v in var_h_second.items():
                if a == i or b == i:
                    total_i += v
            for key, v in var_h_third.items():
                if i in key:
                    total_i += v
            variance_sobol['total_order'][i] = total_i / total_var_h if total_var_h > 0 else 0.0

        variance_sobol['variance_accounting'] = {
            'first_order_total': sum(var_h_first.values()),
            'second_order_total': sum(var_h_second.values()),
            'third_order_total': sum(var_h_third.values()),
            'residual': var_h_residual,
            'total': total_var_h,
            'per_variable_first_order': var_h_first,
            'per_pair_second_order': var_h_second,
            'per_triple_third_order': var_h_third,
        }

        results['variance_sobol'] = variance_sobol

    # --- Variance accounting ---
    results['variance_accounting'] = {
        'first_order_total': sum(first_order_vars.values()),
        'second_order_total': sum(second_order_vars.values()),
        'third_order_total': sum(third_order_vars.values()),
        'residual': residual_var,
        'residual_nn': residual_var,  # backward compat alias
        'total_model_variance': total_var,
        'per_variable_first_order': first_order_vars,
        'per_pair_second_order': second_order_vars,
        'per_triple_third_order': third_order_vars,
    }

    # --- Correlative indices (if data provided) ---
    if x_data is not None:
        corr_results = compute_correlative_sobol(model, x_data)
        results['correlative_sobol'] = corr_results

    return results


def compute_correlative_sobol(model, x_data: jnp.ndarray) -> dict:
    """Compute correlative Sobol indices from empirical component covariances.

    Unlike structural indices (which assume independence via the analytic G),
    correlative indices use the actual covariance of component outputs on data.
    They account for input correlations.

    For independent inputs: correlative ≈ structural.
    For correlated inputs: they may diverge, and don't sum to 1.

    Args:
        model: fitted HiFiANOVA
        x_data: (N, D) input data (transformed to [0,1])

    Returns:
        dict with:
          first_order: {i: S_i^corr}
          cross_correlation_matrix: (D, D) matrix of component correlations
          max_abs_cross_correlation: scalar diagnostic
          correlation_level: 'clean' / 'mild' / 'strong'
    """
    D = model.D
    K1 = model.K1
    K2 = model.K2
    _il1 = getattr(model, 'include_linear_1', True)
    _bn = getattr(model, 'basis_name', 'fourier')
    block1 = basis_size(K1, _il1, _bn)

    # Evaluate each first-order component on the data
    phi1 = model.build_phi1(x_data)
    w1 = model.mean_model.w1

    # Component outputs: f_i(x) = phi_i @ w_i for each variable
    component_outputs = []
    if hasattr(model.mean_model, 'var_specs') and model.mean_model.var_specs is not None:
        for i in range(D):
            _, _, _, block_i, offset_i = model.mean_model.var_specs[i]
            phi_i = phi1[:, offset_i: offset_i + block_i]
            wi = w1[offset_i: offset_i + block_i]
            fi = phi_i @ wi
            component_outputs.append(np.array(fi))
    else:
        for i in range(D):
            phi_i = phi1[:, i * block1: (i + 1) * block1]
            wi = w1[i * block1: (i + 1) * block1]
            fi = phi_i @ wi  # (N,)
            component_outputs.append(np.array(fi))

    component_outputs = np.array(component_outputs)  # (D, N)

    # Empirical covariance matrix of component outputs
    # Cov[f_i, f_j] = (1/N) * sum_n (f_i(x_n) - mean_i)(f_j(x_n) - mean_j)
    # Since each component has ~zero mean by construction, we can use raw outputs
    means = component_outputs.mean(axis=1, keepdims=True)
    centered = component_outputs - means
    cov_matrix = (centered @ centered.T) / component_outputs.shape[1]  # (D, D)

    # Variances (diagonal)
    variances = np.diag(cov_matrix)

    # Correlation matrix
    stds = np.sqrt(np.maximum(variances, 1e-20))
    corr_matrix = cov_matrix / (stds[:, None] * stds[None, :])
    np.fill_diagonal(corr_matrix, 1.0)

    # Max absolute off-diagonal correlation
    mask = ~np.eye(D, dtype=bool)
    max_abs_cross = float(np.max(np.abs(corr_matrix[mask]))) if D > 1 else 0.0

    # Correlative Sobol: S_i^corr = sum_j Cov(f_i, f_j) / Var(f_total)
    # where f_total = sum_i f_i
    total_pred = component_outputs.sum(axis=0)  # (N,)
    total_var_empirical = float(np.var(total_pred))

    correlative_first_order = {}
    if total_var_empirical > 0:
        for i in range(D):
            # Contribution of component i including cross-covariances
            cov_with_total = float(np.cov(component_outputs[i], total_pred)[0, 1])
            correlative_first_order[i] = cov_with_total / total_var_empirical
    else:
        for i in range(D):
            correlative_first_order[i] = 0.0

    # Diagnose correlation level
    if max_abs_cross < 0.1:
        level = 'clean'
    elif max_abs_cross < 0.3:
        level = 'mild'
    else:
        level = 'strong'

    return {
        'first_order': correlative_first_order,
        'cross_correlation_matrix': corr_matrix,
        'covariance_matrix': cov_matrix,
        'max_abs_cross_correlation': max_abs_cross,
        'correlation_level': level,
        'sum_of_correlative_indices': sum(correlative_first_order.values()),
    }
