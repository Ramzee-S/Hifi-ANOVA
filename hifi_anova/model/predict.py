"""Prediction intervals combining aleatoric and epistemic uncertainty.

Two sources of prediction uncertainty are combined here:
  1. Aleatoric: sigma^2(x) from the variance model (irreducible noise)
  2. Fourier epistemic: phi(x)^T Sigma_w phi(x) from the ridge posterior

The total predictive variance is their sum:
    var_total = var_aleatoric + var_epistemic.
Prediction intervals: mean +/- z_{alpha/2} * sqrt(var_total).

(A third source — residual-epistemic uncertainty from a Nystrom GP posterior —
is NOT wired in and contributes nothing to ``var_total`` today. The machinery
lives in ``hifi_anova.model.bayesian_nn`` as a placeholder for a future release;
``predict_intervals`` deliberately computes only the two terms above.)

Usage:
    from hifi_anova.model.predict import predict_intervals, prediction_summary

    # Prediction intervals
    result = predict_intervals(model, x_new, Phi_train, reg_diag, sigma2_hat)
    y_lo, y_hi = result['lower'], result['upper']  # 95% by default

    # Full summary at a single point
    summary = prediction_summary(model, x_new[0:1], ...)
"""

from ..array_backend import xp as jnp  # switchable array backend (numpy exact core)
import numpy as np
from typing import Dict, Optional
from scipy.stats import norm as sp_norm

from ..linalg import spd_inverse


def predict_intervals(
    model,
    x_new: jnp.ndarray,
    Phi_train: Optional[np.ndarray] = None,
    reg_diag: Optional[np.ndarray] = None,
    sigma2_hat: Optional[float] = None,
    alpha: float = 0.05,
    include_epistemic: bool = True,
    weights: Optional[np.ndarray] = None,
    profile_intercept: bool = False,
    df_residual: Optional[float] = None,
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
        weights: (N,) GLS precision weights W = diag(1/sigma^2(x_n)) of the mean
            fit (Stage D). When given, the epistemic term uses the weighted
            posterior A_w = Phi^T W Phi + R and var_ep = phi^T A_w^{-1} phi with NO
            extra sigma^2 factor (W already carries 1/sigma^2). None ⇒ the
            homoscedastic sigma^2 * phi^T A^{-1} phi form (unchanged).
        profile_intercept: if True (Stage-D profiled joint-GLS mean), the weighted
            epistemic posterior is the AUGMENTED one A_aug = Z^T W Z + diag(0, R)
            with Z=[1, Phi], and var_ep(x) = z^T A_aug^{-1} z, z=[1, phi(x)] — the
            intercept is itself uncertain, so holding it fixed under-counts the
            epistemic variance (Remark rem:intercept). Ignored when weights is None.
        df_residual: residual degrees of freedom (N - 2 tr(H) + tr(H^2), from
            ridge_analytics). When given, interval quantiles use Student-t with
            this df instead of the normal — sigma2_hat is itself estimated, and
            the z quantile is anti-conservative at small N. None keeps z
            (asymptotic behavior unchanged; t -> z as df grows).

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

    # Point prediction + aleatoric variance. ``HiFiANOVA.predict`` returns a
    # neutral variance of one when the model has neither a fitted variance
    # model nor the Stage-D constant-variance fallback. In that ordinary
    # homoscedastic case the actual noise estimate is ``sigma2_hat`` from the
    # ridge analytics; using the neutral one would make intervals independent
    # of the fitted noise level (and usually far too wide).
    mean_pred, var_aleatoric = model.predict(x_new)
    mean_np = np.asarray(mean_pred)
    if (model.variance_model is None
            and getattr(model, 'constant_log_var', None) is None
            and sigma2_hat is not None):
        var_al_np = np.full(M, float(sigma2_hat), dtype=np.float64)
    else:
        var_al_np = np.asarray(var_aleatoric)

    # Epistemic variance from Fourier posterior
    var_ep_np = np.zeros(M)
    if include_epistemic and Phi_train is not None and reg_diag is not None:
        Phi_train = np.asarray(Phi_train, dtype=np.float64)
        reg_diag = np.asarray(reg_diag, dtype=np.float64)

        # Features at new points, in the FITTED-DESIGN layout so the columns
        # line up with ``Phi_train`` (= record.Phi). For an order-selective
        # first-order fit (BR-06) that layout drops the excluded variables'
        # first-order columns; ``build_phi_all_fit`` falls back to the uniform
        # ``build_phi_all`` for every ordinary model (and older pickles).
        _build_fit = getattr(model, 'build_phi_all_fit', None)
        Phi_new = np.asarray((_build_fit(x_new) if _build_fit is not None
                              else model.build_phi_all(x_new)),
                             dtype=np.float64)

        if weights is not None:
            # Weighted (GLS) posterior: A_w = Phi^T W Phi + R. The W already
            # carries 1/sigma^2(x_n), so the epistemic quadratic form takes NO
            # extra sigma^2 factor: var_ep = phi_new^T A_w^{-1} phi_new.
            W = np.asarray(weights, dtype=np.float64)
            if profile_intercept:
                # Augmented posterior for the profiled joint-GLS mean: the
                # intercept is a fitted (unpenalized) coordinate, so it enters
                # the posterior via Z=[1, Phi] with a zero-penalty column.
                N_tr = Phi_train.shape[0]
                Z_train = np.concatenate(
                    [np.ones((N_tr, 1), dtype=np.float64), Phi_train], axis=1)
                reg_aug = np.concatenate([[0.0], reg_diag])
                A = Z_train.T @ (W[:, None] * Z_train) + np.diag(reg_aug)
                A_inv = spd_inverse(A)
                Z_new = np.concatenate(
                    [np.ones((M, 1), dtype=np.float64), Phi_new], axis=1)
                Z_Ainv = Z_new @ A_inv  # (M, F+1)
                var_ep_np = np.sum(Z_Ainv * Z_new, axis=1)  # (M,)
            else:
                A = Phi_train.T @ (W[:, None] * Phi_train) + np.diag(reg_diag)
                A_inv = spd_inverse(A)
                Phi_Ainv = Phi_new @ A_inv  # (M, F)
                var_ep_np = np.sum(Phi_Ainv * Phi_new, axis=1)  # (M,)
        else:
            # Homoscedastic posterior: Sigma_w = (Phi^T Phi + R)^{-1};
            # epistemic variance sigma^2 * phi^T Sigma_w phi.
            A = Phi_train.T @ Phi_train + np.diag(reg_diag)
            A_inv = spd_inverse(A)
            s2 = sigma2_hat if sigma2_hat is not None else 1.0
            Phi_Ainv = Phi_new @ A_inv  # (M, F)
            var_ep_np = s2 * np.sum(Phi_Ainv * Phi_new, axis=1)  # (M,)

    # Total variance
    var_total = var_al_np + var_ep_np

    # Prediction interval. With df_residual the quantile is Student-t: the
    # noise scale is estimated, not known, and z undercovers at small N.
    if df_residual is not None and np.isfinite(df_residual) and df_residual > 0:
        from scipy.stats import t as sp_t
        z_crit = float(sp_t.ppf(1.0 - alpha / 2, df_residual))
    else:
        z_crit = float(sp_norm.ppf(1.0 - alpha / 2))
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
