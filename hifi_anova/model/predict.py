"""Prediction intervals combining aleatoric and epistemic uncertainty.

Three sources of prediction uncertainty:
  1. Aleatoric: sigma^2(x) from the variance model (irreducible noise)
  2. Fourier epistemic: phi(x)^T Sigma_w phi(x) from ridge posterior
  3. Residual epistemic: from Nystrom GP posterior (if available)

The total predictive variance is the sum of all three.
Prediction intervals: mean +/- z_{alpha/2} * sqrt(var_total).

Usage:
    from hifi_anova.model.predict import predict_intervals, prediction_summary

    # Prediction intervals
    result = predict_intervals(model, x_new, Phi_train, reg_diag, sigma2_hat)
    y_lo, y_hi = result['lower'], result['upper']  # 95% by default

    # Full summary at a single point
    summary = prediction_summary(model, x_new[0:1], ...)
"""

import jax.numpy as jnp
import numpy as np
from typing import Dict, Optional
from scipy.stats import norm as sp_norm


def predict_intervals(
    model,
    x_new: jnp.ndarray,
    Phi_train: Optional[np.ndarray] = None,
    reg_diag: Optional[np.ndarray] = None,
    sigma2_hat: Optional[float] = None,
    alpha: float = 0.05,
    include_epistemic: bool = True,
) -> Dict:
    """Compute prediction intervals combining all uncertainty sources.

    Args:
        model: fitted HiFiANOVA
        x_new: (M, D) new inputs in [0,1]
        Phi_train: (N, F) training feature matrix — needed for epistemic uncertainty.
            If None, only aleatoric uncertainty is used.
        reg_diag: (F,) regularization diagonal — needed for epistemic uncertainty.
        sigma2_hat: estimated noise variance (from ridge_analytics). If None and
            model has no variance model, defaults to 1.0.
        alpha: significance level (0.05 = 95% intervals)
        include_epistemic: if False, intervals use only aleatoric variance

    Returns:
        dict with:
          mean: (M,) point predictions
          var_aleatoric: (M,) aleatoric variance (from variance model or constant)
          var_epistemic: (M,) epistemic variance (from Fourier posterior)
          var_total: (M,) total predictive variance
          lower: (M,) lower bound of prediction interval
          upper: (M,) upper bound of prediction interval
          alpha: significance level used
    """
    x_new = jnp.asarray(x_new)
    M = x_new.shape[0]

    # Point prediction + aleatoric variance
    mean_pred, var_aleatoric = model.predict(x_new)
    mean_np = np.asarray(mean_pred)
    var_al_np = np.asarray(var_aleatoric)

    # Epistemic variance from Fourier posterior
    var_ep_np = np.zeros(M)
    if include_epistemic and Phi_train is not None and reg_diag is not None:
        Phi_train = np.asarray(Phi_train, dtype=np.float64)
        reg_diag = np.asarray(reg_diag, dtype=np.float64)

        # Posterior covariance: Sigma_w = (Phi^T Phi + R)^{-1}
        A = Phi_train.T @ Phi_train + np.diag(reg_diag)
        A_inv = np.linalg.inv(A)

        # Features at new points
        Phi_new = np.asarray(model.build_phi_all(x_new), dtype=np.float64)

        # Epistemic variance: sigma^2 * phi^T Sigma_w phi
        s2 = sigma2_hat if sigma2_hat is not None else 1.0
        Phi_Ainv = Phi_new @ A_inv  # (M, F)
        var_ep_np = s2 * np.sum(Phi_Ainv * Phi_new, axis=1)  # (M,)

    # Total variance
    var_total = var_al_np + var_ep_np

    # Prediction interval
    z_crit = sp_norm.ppf(1.0 - alpha / 2)
    std_total = np.sqrt(np.maximum(var_total, 0.0))
    lower = mean_np - z_crit * std_total
    upper = mean_np + z_crit * std_total

    return {
        'mean': mean_np,
        'var_aleatoric': var_al_np,
        'var_epistemic': var_ep_np,
        'var_total': var_total,
        'std_total': std_total,
        'lower': lower,
        'upper': upper,
        'alpha': alpha,
    }


def prediction_summary(
    model,
    x_single: jnp.ndarray,
    Phi_train: Optional[np.ndarray] = None,
    reg_diag: Optional[np.ndarray] = None,
    sigma2_hat: Optional[float] = None,
    feature_names: Optional[list] = None,
) -> Dict:
    """Detailed prediction summary for a single input point.

    Returns the point prediction, all uncertainty components,
    prediction interval, and per-variable contribution breakdown.

    Args:
        model: fitted HiFiANOVA
        x_single: (1, D) or (D,) single input
        Phi_train, reg_diag, sigma2_hat: for epistemic uncertainty
        feature_names: variable names for the contribution breakdown

    Returns:
        dict with prediction, intervals, and per-variable contributions
    """
    from ..analysis.component_eval import evaluate_all_first_order

    x = jnp.asarray(x_single)
    if x.ndim == 1:
        x = x[None, :]  # (1, D)

    # Intervals
    intervals = predict_intervals(
        model, x, Phi_train, reg_diag, sigma2_hat)

    # Per-variable contributions
    D = model.D
    if feature_names is None:
        feature_names = [f'x{i+1}' for i in range(D)]

    components = evaluate_all_first_order(model, x)
    contributions = {}
    for i in range(D):
        contributions[feature_names[i]] = float(components[i][0])

    return {
        'prediction': float(intervals['mean'][0]),
        'interval_95': (float(intervals['lower'][0]), float(intervals['upper'][0])),
        'std_total': float(intervals['std_total'][0]),
        'var_aleatoric': float(intervals['var_aleatoric'][0]),
        'var_epistemic': float(intervals['var_epistemic'][0]),
        'contributions': contributions,
    }
