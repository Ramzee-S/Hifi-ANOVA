"""Haar wavelet residual diagnostic.

After fitting with a smooth basis (Fourier or Legendre), project the residuals
onto Haar wavelets to detect localized features the smooth basis missed —
step changes, threshold effects, regime boundaries.

This is a DIAGNOSTIC: it does not fit a model. It answers:
"Does the residual contain localized features that the smooth basis missed?"

Usage:
    from hifi_anova.analysis.haar_diagnostic import haar_residual_analysis

    residual = y - model.predict_mean_only(x_test)
    result = haar_residual_analysis(residual, x_test, max_scale=4)
    for i in result['summary']['variables_with_localized']:
        print(f"x_{i}: localized features at scale "
              f"{result['per_variable'][i]['dominant_scale']}")
"""

import jax.numpy as jnp
import numpy as np
from typing import Optional

from ..core.haar import HaarBasis


def haar_residual_analysis(
    residual: jnp.ndarray,
    x_data: jnp.ndarray,
    max_scale: int = 4,
    significance_threshold: float = 0.01,
) -> dict:
    """Analyze residuals for localized features using Haar wavelets.

    For each variable, projects the residual onto the Haar basis and
    measures how much variance is captured at each scale.

    Args:
        residual: (N,) residual from the primary model (y - y_hat).
        x_data: (N, D) inputs in [0, 1].
        max_scale: J, the maximum wavelet scale. J=4 gives 15 basis functions
            per variable (quarter-domain resolution).
        significance_threshold: minimum fraction of residual variance for a
            variable to be flagged as having localized features.

    Returns:
        dict with:
          per_variable: {i: {
              'total_haar_variance': float,
              'fraction_of_residual': float,
              'per_scale_variance': {j: float},
              'dominant_scale': int or None,
              'significant_coefficients': [dict, ...],
              'has_localized_features': bool,
          }}
          summary: {
              'any_localized': bool,
              'variables_with_localized': [int, ...],
              'total_haar_fraction': float,
              'recommendation': str,
          }
    """
    N, D = x_data.shape
    haar = HaarBasis(max_scale)
    residual = jnp.asarray(residual).ravel()
    residual_var = float(jnp.var(residual))

    results = {'per_variable': {}, 'summary': {}}
    localized_vars = []
    total_haar_var = 0.0

    for i in range(D):
        Phi_haar_i = haar.evaluate(x_data[:, i])  # (N, n_basis)

        # Orthonormal basis: coefficients via least-squares projection.
        # For truly orthonormal bases on the empirical distribution,
        # (Phi^T Phi / N) ≈ I, so coeffs ≈ Phi^T r / N.
        # Use the exact projection for correctness:
        coeffs = jnp.linalg.lstsq(Phi_haar_i, residual, rcond=None)[0]

        # Variance captured per scale
        per_scale = {}
        for j in range(1, max_scale + 1):
            s = haar.scale_slice(j)
            scale_coeffs = coeffs[s]
            per_scale[j] = float(jnp.sum(scale_coeffs ** 2))

        var_i = sum(per_scale.values())
        fraction = var_i / (residual_var + 1e-10)
        total_haar_var += var_i

        # Find the dominant scale (scale with most variance)
        dominant_scale = None
        if var_i > 0:
            dominant_scale = max(per_scale, key=per_scale.get)

        # Find significant individual coefficients
        significant = []
        for idx in range(len(coeffs)):
            c = float(coeffs[idx])
            var_frac = c ** 2 / (residual_var + 1e-10)
            if var_frac > significance_threshold:
                j, k, start, end = haar.position_of_index(idx)
                significant.append({
                    'scale': j,
                    'position': k,
                    'interval': (start, end),
                    'coefficient': c,
                    'variance_fraction': var_frac,
                })

        has_localized = fraction > significance_threshold
        if has_localized:
            localized_vars.append(i)

        results['per_variable'][i] = {
            'total_haar_variance': var_i,
            'fraction_of_residual': fraction,
            'per_scale_variance': per_scale,
            'dominant_scale': dominant_scale,
            'significant_coefficients': significant,
            'has_localized_features': has_localized,
        }

    total_fraction = total_haar_var / (residual_var + 1e-10)

    # Build recommendation
    if not localized_vars:
        recommendation = ("No localized features detected in residuals. "
                          "The smooth basis appears adequate.")
    elif len(localized_vars) <= 3:
        var_names = ', '.join(f'x_{i}' for i in localized_vars)
        recommendation = (f"Localized features detected in {var_names}. "
                          f"Consider Haar basis or combined model for "
                          f"these variables.")
    else:
        recommendation = (f"Localized features detected in {len(localized_vars)} "
                          f"variables ({total_fraction:.1%} of residual). "
                          f"Consider switching to Haar basis.")

    results['summary'] = {
        'any_localized': len(localized_vars) > 0,
        'variables_with_localized': localized_vars,
        'total_haar_fraction': total_fraction,
        'recommendation': recommendation,
    }

    return results


def haar_multi_basis_characterization(
    model,
    x_data: jnp.ndarray,
    y_data: jnp.ndarray,
    max_scale: int = 4,
    significance_threshold: float = 0.01,
) -> dict:
    """Characterize each variable's effect type using multi-basis analysis.

    After fitting the primary model, projects residuals onto both Fourier
    and Haar bases to classify each variable as polynomial, oscillatory,
    or localized.

    This implements Mode 3a from the Haar spec: independent fits with
    cross-residual analysis. The percentages for oscillatory and localized
    are upper bounds (may overlap).

    Args:
        model: fitted HiFiANOVA model (provides predictions and Sobol indices).
        x_data: (N, D) inputs in [0, 1].
        y_data: (N,) targets.
        max_scale: J for Haar analysis.
        significance_threshold: minimum fraction to flag.

    Returns:
        dict with per_variable characterization and summary table.
    """
    from ..analysis.sobol import compute_sobol_indices
    from ..core.features import build_first_order_features
    from ..core.gram import build_gram_matrix

    N, D = x_data.shape
    y_hat = model.predict_mean_only(x_data)
    residual = y_data - y_hat
    residual_var = float(jnp.var(residual))

    # Get primary model's Sobol indices
    sobol = compute_sobol_indices(model, x_data)
    primary_sobol = sobol.get('mean_sobol', {}).get('first_order', {})

    # Haar analysis on residual
    haar = HaarBasis(max_scale)
    haar_result = haar_residual_analysis(residual, x_data, max_scale,
                                         significance_threshold)

    # Fourier analysis on residual (per variable)
    characterization = {}
    for i in range(D):
        primary_frac = float(primary_sobol.get(i, 0.0))

        # Haar capture for this variable
        haar_frac = haar_result['per_variable'][i]['fraction_of_residual']

        # Fourier capture: project residual onto per-variable Fourier features
        # Use K matching the model's K1
        K_fourier = getattr(model, 'K1', 5)
        x_i = x_data[:, i:i+1]  # (N, 1)
        phi_f_i = build_first_order_features(x_i, K_fourier,
                                              basis_name='fourier')  # (N, 2K+1)
        coeffs_f = jnp.linalg.lstsq(phi_f_i, residual, rcond=None)[0]
        # Use full Gram matrix (w^T G w) — the diagonal approximation
        # misses the linear term variance (1/12 vs 1/2) and linear-sine
        # cross-terms (-1/(2*pi*k)), which can cause 4x errors.
        G_fourier = build_gram_matrix(K_fourier, include_linear=True,
                                       basis_name='fourier')
        coeffs_f64 = jnp.asarray(coeffs_f, dtype=jnp.float64)
        G_f64 = jnp.asarray(G_fourier, dtype=jnp.float64)
        fourier_var = float(jnp.maximum(0.0, coeffs_f64 @ G_f64 @ coeffs_f64))
        fourier_frac = fourier_var / (residual_var + 1e-10)

        # Classify
        if primary_frac > 0.8:
            character = 'polynomial'
        elif haar_frac > fourier_frac and haar_frac > significance_threshold:
            character = 'localized'
        elif fourier_frac > significance_threshold:
            character = 'oscillatory'
        elif primary_frac > significance_threshold:
            character = 'polynomial'
        else:
            character = 'negligible'

        characterization[i] = {
            'primary_sobol': primary_frac,
            'residual_fourier_fraction': fourier_frac,
            'residual_haar_fraction': haar_frac,
            'character': character,
            'dominant_haar_scale': haar_result['per_variable'][i].get('dominant_scale'),
        }

    return {
        'per_variable': characterization,
        'residual_variance': residual_var,
        'haar_details': haar_result,
    }
