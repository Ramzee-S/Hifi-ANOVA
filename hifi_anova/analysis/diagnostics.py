"""Variance accounting, calibration, and correlation diagnostics."""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Optional

from .sobol import compute_sobol_indices, compute_correlative_sobol


def variance_accounting_report(model, x_data: jnp.ndarray,
                               y_data: jnp.ndarray) -> dict:
    """Complete hierarchical variance accounting.

    Computes:
    - Per-variable first-order variance (analytic)
    - Per-pair second-order variance (analytic)
    - Residual NN variance (empirical)
    - Total Var(y) vs sum of components (additivity check)
    """
    sobol_results = compute_sobol_indices(model, x_data)
    va = sobol_results['variance_accounting']

    # Empirical total variance of y
    total_var_y = float(jnp.var(y_data))

    # Model predictions
    mean_pred, var_pred = model.predict(x_data)
    residuals = y_data - mean_pred
    empirical_residual_var = float(jnp.var(residuals))

    va['total_var_y'] = total_var_y
    va['empirical_residual_var'] = empirical_residual_var
    va['R_squared'] = 1.0 - empirical_residual_var / total_var_y if total_var_y > 0 else 0.0
    va['additivity_gap'] = abs(total_var_y - va['total_model_variance'] - empirical_residual_var) / total_var_y if total_var_y > 0 else 0.0

    return va


def calibration_report(model, x_data: jnp.ndarray,
                       y_data: jnp.ndarray) -> dict:
    """Calibration check for the heteroscedastic model.

    Computes standardized residuals z_n = (y_n - f_hat(x_n)) / sigma_hat(x_n).
    Checks:
      - mean(z) approx 0
      - var(z) approx 1
    """
    mean_pred, var_pred = model.predict(x_data)
    sigma_pred = jnp.sqrt(var_pred)

    residuals = y_data - mean_pred
    standardized = residuals / sigma_pred

    z = np.array(standardized)

    report = {
        'mean_standardized_residual': float(np.mean(z)),
        'var_standardized_residual': float(np.var(z)),
        'std_standardized_residual': float(np.std(z)),
        'skewness': float(np.mean((z - np.mean(z))**3) / np.std(z)**3),
        'kurtosis': float(np.mean((z - np.mean(z))**4) / np.std(z)**4 - 3.0),
    }

    # Coverage at various levels
    for alpha in [0.5, 0.9, 0.95, 0.99]:
        from scipy.stats import norm
        z_crit = norm.ppf((1 + alpha) / 2)
        coverage = float(np.mean(np.abs(z) <= z_crit))
        report[f'coverage_{alpha}'] = coverage

    return report


def correlation_diagnostic(model, x_data: jnp.ndarray,
                          variable_names: Optional[list] = None) -> dict:
    """Diagnose the impact of input correlations on the Sobol decomposition.

    Compares structural (analytic, independence-assuming) indices against
    correlative (empirical, correlation-aware) indices. The divergence
    between them quantifies how much input correlations affect attribution.

    Args:
        model: fitted HiFiANOVA
        x_data: (N, D) input data (transformed)
        variable_names: optional names for reporting

    Returns:
        dict with:
          structural_indices: {i: S_i^struct}
          correlative_indices: {i: S_i^corr}
          divergence: {i: |S_i^struct - S_i^corr|}
          max_divergence: scalar
          cross_correlation_matrix: (D, D)
          max_abs_cross_correlation: scalar
          recommendation: string
    """
    D = model.D
    if variable_names is None:
        variable_names = [f"x{i+1}" for i in range(D)]

    # Structural indices (analytic G, assumes independence)
    struct_results = compute_sobol_indices(model)
    structural = struct_results['mean_sobol']['first_order']

    # Correlative indices (empirical, respects data correlations)
    corr_results = compute_correlative_sobol(model, x_data)
    correlative = corr_results['first_order']

    # Divergence
    divergence = {}
    for i in range(D):
        divergence[i] = abs(structural.get(i, 0) - correlative.get(i, 0))

    max_div = max(divergence.values()) if divergence else 0.0
    max_cross = corr_results['max_abs_cross_correlation']

    # Recommendation
    if max_cross < 0.1 and max_div < 0.05:
        recommendation = (
            "Input correlations are negligible. Structural (analytic) "
            "indices are reliable and sum to 1."
        )
    elif max_cross < 0.3 and max_div < 0.15:
        recommendation = (
            "Mild input correlations detected. Structural indices are "
            "approximate. Report both types for transparency."
        )
    else:
        recommendation = (
            "Strong input correlations detected. Structural indices may "
            "be misleading. Correlative indices better reflect the data "
            "distribution but do not sum to 1. Consider the interpretive "
            "implications."
        )

    return {
        'structural_indices': structural,
        'correlative_indices': correlative,
        'divergence': divergence,
        'max_divergence': max_div,
        'cross_correlation_matrix': corr_results['cross_correlation_matrix'],
        'max_abs_cross_correlation': max_cross,
        'correlation_level': corr_results['correlation_level'],
        'sum_structural': sum(structural.values()),
        'sum_correlative': corr_results['sum_of_correlative_indices'],
        'recommendation': recommendation,
        'variable_names': variable_names,
    }
