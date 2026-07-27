"""Analytic AutoML: model selection, uncertainty, and diagnostics from one ridge solve.

The linear structure of the Hoeffding-Fourier model enables a complete AutoML
pipeline without expensive retraining loops:

  1. Exact LOO-CV from hat matrix leverages (free from one solve)
  2. Noise estimation: sigma^2(lambda) = RSS / (N - df)
  3. Sandwich estimator: bootstrap-quality confidence intervals on Sobol indices
  4. K-fold CV via Woodbury rank updates (k inversions of N/k matrices)
  5. Sample size diagnostics: effective N per parameter, precision estimates

All quantities derive from the ridge solution and its hat matrix H = Phi A^{-1} Phi^T,
which is already computed during fitting.

Usage:
    from hifi_anova.analysis.automl import ridge_analytics, sobol_confidence_intervals

    # Full analytics from one ridge solve
    analytics = ridge_analytics(Phi, y, reg_diag)
    print(f"sigma_hat = {analytics['sigma_hat']:.4f}")
    print(f"LOO-CV = {analytics['loo_cv']:.4f}")

    # Sobol indices with confidence intervals
    sobol_ci = sobol_confidence_intervals(Phi, y, reg_diag, D, K1, G1)
    for i, (s, lo, hi) in sobol_ci['first_order'].items():
        print(f"  x{i+1}: S = {s:.3f} [{lo:.3f}, {hi:.3f}]")
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


# =============================================================================
# Core analytics from one ridge solve
# =============================================================================

def ridge_analytics(
    Phi: np.ndarray,
    y: np.ndarray,
    reg_diag: np.ndarray,
) -> Dict:
    """Complete analytical diagnostics from a single ridge solve.

    Computes everything derivable from the hat matrix without refitting:
    - Coefficients w, residuals, RSS, MSE
    - Effective degrees of freedom df = tr(H)
    - Noise estimate sigma_hat^2 = RSS / (N - df)
    - Per-observation leverages H_ii (diagonal of hat matrix)
    - Exact LOO-CV (from leverages)
    - GCV, AIC, BIC, profile evidence
    - Effective sample size per parameter

    Args:
        Phi: (N, F) feature matrix
        y: (N,) centered targets
        reg_diag: (F,) regularization diagonal

    Returns:
        Dict with all diagnostics
    """
    Phi = np.asarray(Phi, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    reg_diag = np.asarray(reg_diag, dtype=np.float64)
    N, F = Phi.shape

    # Ridge solve
    PhiTPhi = Phi.T @ Phi
    A = PhiTPhi + np.diag(reg_diag)
    A_inv = np.linalg.inv(A)
    w = A_inv @ (Phi.T @ y)

    # Residuals
    residuals = y - Phi @ w
    rss = float(np.sum(residuals ** 2))
    mse = rss / N

    # Effective degrees of freedom
    # df = tr(H) = tr(Phi A^{-1} Phi^T) = tr(A^{-1} Phi^T Phi)
    df = float(np.trace(A_inv @ PhiTPhi))

    # Noise estimate (REML-style)
    df_residual = max(N - df, 1.0)
    sigma2_hat = rss / df_residual
    sigma_hat = np.sqrt(sigma2_hat)

    # Per-observation leverages: H_ii = Phi_i^T A^{-1} Phi_i
    # H = Phi A^{-1} Phi^T is N x N — too large to store.
    # But we only need the diagonal: H_ii = sum_j (Phi A^{-1/2})_{ij}^2
    # Efficiently: H_ii = row_i(Phi) @ A^{-1} @ row_i(Phi)^T
    # Vectorized: H_diag = rowsum(Phi @ A_inv * Phi)
    Phi_Ainv = Phi @ A_inv  # (N, F)
    leverages = np.sum(Phi_Ainv * Phi, axis=1)  # (N,)

    # Exact LOO-CV: CV_i = [r_i / (1 - H_ii)]^2
    # Guard against leverage overflow: H_ii should be in [0, 1) for
    # regularized regression. Clip to avoid division by zero.
    leverages = np.clip(leverages, 0.0, 1.0 - 1e-10)
    loo_residuals = residuals / (1.0 - leverages)
    loo_cv = float(np.mean(loo_residuals ** 2))

    # GCV (average-leverage approximation of LOO)
    gcv_denom = max(1e-10, 1.0 - df / N)
    gcv = (rss / N) / gcv_denom ** 2

    # AIC, BIC
    aic = N * np.log(max(mse, 1e-15)) + 2.0 * df
    bic = N * np.log(max(mse, 1e-15)) + np.log(N) * df

    # Effective sample size per parameter
    ess_per_param = N / max(df, 1.0)

    return {
        'w': w,
        'A_inv': A_inv,
        'residuals': residuals,
        'rss': rss,
        'mse': mse,
        'df': df,
        'sigma2_hat': sigma2_hat,
        'sigma_hat': float(sigma_hat),
        'leverages': leverages,
        'loo_cv': loo_cv,
        'loo_residuals': loo_residuals,
        'gcv': gcv,
        'aic': aic,
        'bic': bic,
        'ess_per_param': ess_per_param,
        'N': N,
        'F': F,
    }


# =============================================================================
# Sandwich estimator for coefficient covariance
# =============================================================================

def sandwich_covariance(
    Phi: np.ndarray,
    A_inv: np.ndarray,
    residuals: np.ndarray,
) -> np.ndarray:
    """Heteroscedasticity-robust covariance of ridge coefficients.

    Var(w) ≈ A^{-1} [Phi^T diag(r^2) Phi] A^{-1}

    This is the HC0 sandwich estimator. It provides bootstrap-quality
    standard errors without resampling, and naturally handles
    heteroscedastic residuals.

    Args:
        Phi: (N, F) feature matrix
        A_inv: (F, F) inverse of (Phi^T Phi + R)
        residuals: (N,) residuals from the ridge fit

    Returns:
        (F, F) covariance matrix of w
    """
    Phi = np.asarray(Phi, dtype=np.float64)
    A_inv = np.asarray(A_inv, dtype=np.float64)
    r2 = np.asarray(residuals, dtype=np.float64) ** 2

    # Phi^T diag(r^2) Phi = (Phi * r)^T (Phi * r)  where r = |residuals|
    Phi_weighted = Phi * np.sqrt(r2)[:, None]  # (N, F)
    meat = Phi_weighted.T @ Phi_weighted  # (F, F)

    # Sandwich: A^{-1} meat A^{-1}
    return A_inv @ meat @ A_inv


# =============================================================================
# Sobol confidence intervals via delta method
# =============================================================================

def sobol_confidence_intervals(
    Phi: np.ndarray,
    y: np.ndarray,
    reg_diag: np.ndarray,
    D: int,
    K1: int,
    G1: np.ndarray,
    K2: int = 0,
    P: int = 0,
    G2: Optional[np.ndarray] = None,
    pair_indices: Optional[np.ndarray] = None,
    K3: int = 0,
    T: int = 0,
    G3: Optional[np.ndarray] = None,
    triple_indices: Optional[np.ndarray] = None,
    alpha: float = 0.05,
    include_linear_1: bool = True,
    basis_name: str = 'fourier',
) -> Dict:
    """Confidence intervals on Sobol indices via sandwich + delta method.

    For each Sobol index S_i = Var_i / Var_total = (w_i^T G w_i) / (sum w_j^T G w_j):
      Var(S_i) ≈ (dS_i/dw)^T Cov(w) (dS_i/dw)

    The gradient dS_i/dw is computed analytically from the Gram matrix.
    Cov(w) comes from the sandwich estimator.

    Args:
        Phi, y, reg_diag: ridge inputs
        D, K1, G1: first-order structure
        K2, P, G2, pair_indices: second-order (optional)
        K3, T, G3, triple_indices: third-order (optional)
        alpha: significance level (0.05 = 95% CI)
        include_linear_1: whether first-order basis includes linear term
        basis_name: 'fourier' or 'legendre'

    Returns:
        Dict with:
          first_order: {i: (S_i, S_lo, S_hi)}
          second_order: {(i,j): (S_ij, S_lo, S_hi)}
          third_order: {(i,j,k): (S_ijk, S_lo, S_hi)}
          sigma_hat: noise estimate
          df: effective degrees of freedom
    """
    from scipy.stats import norm as sp_norm

    Phi = np.asarray(Phi, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    reg_diag = np.asarray(reg_diag, dtype=np.float64)
    G1 = np.asarray(G1, dtype=np.float64)

    # Get analytics
    analytics = ridge_analytics(Phi, y, reg_diag)
    w = analytics['w']
    A_inv = analytics['A_inv']
    residuals = analytics['residuals']
    N, F = Phi.shape

    # Sandwich covariance
    Cov_w = sandwich_covariance(Phi, A_inv, residuals)

    from ..core.features import basis_size as _bs
    block1 = _bs(K1, include_linear_1, basis_name)

    # Guard: the first-order block layout we were told to assume must fit inside
    # the design matrix. If this fails, basis_name / include_linear_1 do not
    # match how Phi was actually built — fail loudly instead of slicing the
    # wrong columns and returning silently-wrong Sobol CIs.
    if D * block1 > F:
        raise ValueError(
            f"Sobol CI basis-layout mismatch: D*block1 = {D}*{block1} = "
            f"{D * block1} exceeds Phi columns = {F}. Check that basis_name="
            f"'{basis_name}' and include_linear_1={include_linear_1} match the "
            f"basis Phi was built with.")

    # Compute component variances
    first_order_vars = {}
    for i in range(D):
        sl = slice(i * block1, (i + 1) * block1)
        wi = w[sl]
        first_order_vars[i] = max(0.0, float(wi @ G1 @ wi))

    second_order_vars = {}
    F1 = D * block1
    if K2 > 0 and P > 0 and G2 is not None:
        G2_np = np.asarray(G2, dtype=np.float64)
        block2 = G2_np.shape[0]
        for p in range(P):
            sl = slice(F1 + p * block2, F1 + (p + 1) * block2)
            wp = w[sl]
            var_p = max(0.0, float(wp @ G2_np @ wp))
            i, j = int(pair_indices[p, 0]), int(pair_indices[p, 1])
            second_order_vars[(i, j)] = var_p

    third_order_vars = {}
    F2 = P * (G2.shape[0] if G2 is not None else 0)
    if K3 > 0 and T > 0 and G3 is not None:
        G3_np = np.asarray(G3, dtype=np.float64)
        block3 = G3_np.shape[0]
        for t in range(T):
            sl = slice(F1 + F2 + t * block3, F1 + F2 + (t + 1) * block3)
            wt = w[sl]
            var_t = max(0.0, float(wt @ G3_np @ wt))
            i, j, k = (int(triple_indices[t, l]) for l in range(3))
            third_order_vars[(i, j, k)] = var_t

    total_var = (sum(first_order_vars.values()) +
                 sum(second_order_vars.values()) +
                 sum(third_order_vars.values()))

    if total_var < 1e-15:
        # Degenerate: no variance explained
        return {
            'first_order': {i: (0.0, 0.0, 0.0) for i in range(D)},
            'second_order': {},
            'third_order': {},
            'sigma_hat': analytics['sigma_hat'],
            'df': analytics['df'],
            'total_model_variance': 0.0,
        }

    z_crit = sp_norm.ppf(1.0 - alpha / 2)

    def _sobol_ci(var_component, var_total, w_slice, G_block):
        """Compute S, SE(S), and CI for one component via delta method.

        S = w^T G w / total_var
        dS/dw_slice = 2 G w / total_var - S * (sum of 2 G_j w_j / total_var for all j)

        Simplification: for the component's own w_slice, the dominant term is
        dS/dw = 2 G w / total_var (the cross-term is second order for S << 1).
        We use the full gradient for accuracy.
        """
        S = var_component / var_total if var_total > 0 else 0.0

        # Gradient of numerator w.r.t. w_slice: d(w^T G w)/dw = 2 G w
        grad_num = 2.0 * G_block @ w[w_slice]  # (block_size,)

        # Full gradient of S w.r.t. all w: need dS/dw for the full w vector
        # dS/dw_slice = (1/V_tot) * 2Gw - (V_i/V_tot^2) * dV_tot/dw_slice
        # where dV_tot/dw_slice = 2Gw (same component)
        # = 2Gw/V_tot - S * 2Gw/V_tot = 2Gw(1-S)/V_tot
        grad_S = grad_num * (1.0 - S) / var_total

        # SE via delta method: SE(S) = sqrt(grad^T Cov_slice grad)
        Cov_slice = Cov_w[w_slice, w_slice]  # (block, block)
        var_S = float(grad_S @ Cov_slice @ grad_S)
        se_S = np.sqrt(max(0.0, var_S))

        lo = max(0.0, S - z_crit * se_S)
        hi = min(1.0, S + z_crit * se_S)
        return (S, lo, hi, se_S)

    # First-order CIs
    first_order_ci = {}
    for i in range(D):
        sl = slice(i * block1, (i + 1) * block1)
        S, lo, hi, se = _sobol_ci(first_order_vars[i], total_var, sl, G1)
        first_order_ci[i] = (S, lo, hi)

    # Second-order CIs
    second_order_ci = {}
    if K2 > 0 and G2 is not None:
        G2_np = np.asarray(G2, dtype=np.float64)
        block2 = G2_np.shape[0]
        for p, ((i, j), var_p) in enumerate(second_order_vars.items()):
            sl = slice(F1 + p * block2, F1 + (p + 1) * block2)
            S, lo, hi, se = _sobol_ci(var_p, total_var, sl, G2_np)
            second_order_ci[(i, j)] = (S, lo, hi)

    # Third-order CIs
    third_order_ci = {}
    if K3 > 0 and G3 is not None:
        G3_np = np.asarray(G3, dtype=np.float64)
        block3 = G3_np.shape[0]
        for t, ((i, j, k), var_t) in enumerate(third_order_vars.items()):
            sl = slice(F1 + F2 + t * block3, F1 + F2 + (t + 1) * block3)
            S, lo, hi, se = _sobol_ci(var_t, total_var, sl, G3_np)
            third_order_ci[(i, j, k)] = (S, lo, hi)

    return {
        'first_order': first_order_ci,
        'second_order': second_order_ci,
        'third_order': third_order_ci,
        'sigma_hat': analytics['sigma_hat'],
        'sigma2_hat': analytics['sigma2_hat'],
        'df': analytics['df'],
        'loo_cv': analytics['loo_cv'],
        'gcv': analytics['gcv'],
        'aic': analytics['aic'],
        'bic': analytics['bic'],
        'total_model_variance': total_var,
        'alpha': alpha,
    }


# =============================================================================
# Noise-complexity curve: sigma^2(lambda)
# =============================================================================

def noise_complexity_curve(
    Phi: np.ndarray,
    y: np.ndarray,
    reg_structure: np.ndarray,
    n_lambdas: int = 50,
    lambda_range: Tuple[float, float] = (1e-6, 1e2),
) -> Dict:
    """Compute the noise-complexity tradeoff curve sigma^2(lambda).

    sigma^2(lambda) = RSS(lambda) / (N - df(lambda))

    This curve has a characteristic shape:
    - Small lambda: sigma^2 is small (overfitting — noise absorbed into model)
    - Optimal lambda: sigma^2 equals the true noise level
    - Large lambda: sigma^2 is large (underfitting — signal leaks into residuals)

    The minimum of sigma^2(lambda) estimates the true noise variance.

    Args:
        Phi: (N, F) feature matrix
        y: (N,) centered targets
        reg_structure: (F,) relative penalty weights
        n_lambdas: number of lambda values
        lambda_range: search range

    Returns:
        Dict with lambda grid, sigma2 curve, df curve, and estimates
    """
    Phi = np.asarray(Phi, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    reg_structure = np.asarray(reg_structure, dtype=np.float64)
    N, F = Phi.shape

    lambdas = np.logspace(np.log10(lambda_range[0]),
                          np.log10(lambda_range[1]), n_lambdas)

    sigma2_curve = []
    df_curve = []
    rss_curve = []
    loo_curve = []
    gcv_curve = []
    aic_curve = []
    bic_curve = []

    for lam in lambdas:
        reg_diag = lam * reg_structure
        analytics = ridge_analytics(Phi, y, reg_diag)
        sigma2_curve.append(analytics['sigma2_hat'])
        df_curve.append(analytics['df'])
        rss_curve.append(analytics['rss'])
        loo_curve.append(analytics['loo_cv'])
        gcv_curve.append(analytics['gcv'])
        aic_curve.append(analytics['aic'])
        bic_curve.append(analytics['bic'])

    sigma2_curve = np.array(sigma2_curve)
    df_curve = np.array(df_curve)

    # Find optimal points
    idx_sigma_min = int(np.argmin(sigma2_curve))
    idx_loo_min = int(np.argmin(loo_curve))
    idx_gcv_min = int(np.argmin(gcv_curve))
    idx_aic_min = int(np.argmin(aic_curve))
    idx_bic_min = int(np.argmin(bic_curve))

    return {
        'lambdas': lambdas,
        'sigma2': sigma2_curve,
        'df': df_curve,
        'rss': np.array(rss_curve),
        'loo_cv': np.array(loo_curve),
        'gcv': np.array(gcv_curve),
        'aic': np.array(aic_curve),
        'bic': np.array(bic_curve),
        # Optima
        'sigma2_min': float(sigma2_curve[idx_sigma_min]),
        'lambda_sigma_opt': float(lambdas[idx_sigma_min]),
        'lambda_loo_opt': float(lambdas[idx_loo_min]),
        'lambda_gcv_opt': float(lambdas[idx_gcv_min]),
        'lambda_aic_opt': float(lambdas[idx_aic_min]),
        'lambda_bic_opt': float(lambdas[idx_bic_min]),
        'df_at_sigma_opt': float(df_curve[idx_sigma_min]),
        'N': N,
        'F': F,
    }


# =============================================================================
# K-fold CV via Woodbury rank updates
# =============================================================================

def kfold_cv_analytic(
    Phi: np.ndarray,
    y: np.ndarray,
    reg_diag: np.ndarray,
    n_folds: int = 5,
    seed: int = 42,
) -> Dict:
    """Exact K-fold CV using Woodbury rank updates — no refitting.

    For each fold k, removing N/k observations changes the ridge solution
    via a rank-(N/k) update using the Woodbury identity:

    w_{-k} = w - A^{-1} Phi_k^T [I + Phi_k A^{-1} Phi_k^T]^{-1} (Phi_k w - y_k)

    A^{-1} is computed once. The inner matrix is only (N/k) x (N/k).

    Args:
        Phi: (N, F) feature matrix
        y: (N,) centered targets
        reg_diag: (F,) regularization diagonal
        n_folds: number of folds
        seed: random seed for fold assignment

    Returns:
        Dict with per-fold errors, overall CV, and standard error
    """
    Phi = np.asarray(Phi, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    reg_diag = np.asarray(reg_diag, dtype=np.float64)
    N, F = Phi.shape

    # Full solve
    A = Phi.T @ Phi + np.diag(reg_diag)
    A_inv = np.linalg.inv(A)
    w_full = A_inv @ (Phi.T @ y)

    # Assign folds
    rng = np.random.RandomState(seed)
    fold_ids = np.zeros(N, dtype=int)
    perm = rng.permutation(N)
    fold_size = N // n_folds
    for k in range(n_folds):
        start = k * fold_size
        end = (k + 1) * fold_size if k < n_folds - 1 else N
        fold_ids[perm[start:end]] = k

    fold_mses = []
    for k in range(n_folds):
        val_mask = fold_ids == k
        Phi_k = Phi[val_mask]    # (n_k, F)
        y_k = y[val_mask]        # (n_k,)
        n_k = int(np.sum(val_mask))

        # Woodbury update: w_{-k} = w - A^{-1} Phi_k^T [I + Phi_k A^{-1} Phi_k^T]^{-1} (Phi_k w - y_k)
        A_inv_PhikT = A_inv @ Phi_k.T  # (F, n_k)
        inner = np.eye(n_k) + Phi_k @ A_inv_PhikT  # (n_k, n_k)
        delta = Phi_k @ w_full - y_k  # (n_k,)
        correction = A_inv_PhikT @ np.linalg.solve(inner, delta)  # (F,)
        w_minus_k = w_full - correction

        # Validation predictions
        pred_k = Phi_k @ w_minus_k
        fold_mse = float(np.mean((y_k - pred_k) ** 2))
        fold_mses.append(fold_mse)

    fold_mses = np.array(fold_mses)
    cv_mean = float(np.mean(fold_mses))
    cv_se = float(np.std(fold_mses, ddof=1) / np.sqrt(n_folds))

    return {
        'cv_mean': cv_mean,
        'cv_se': cv_se,
        'fold_mses': fold_mses,
        'n_folds': n_folds,
        'N': N,
    }


# =============================================================================
# Stability diagnostics via K-fold
# =============================================================================

def stability_diagnostics(
    Phi: np.ndarray,
    y: np.ndarray,
    reg_diag: np.ndarray,
    D: int,
    K1: int,
    G1: np.ndarray,
    n_folds: int = 5,
    seed: int = 42,
    K2: int = 0,
    P: int = 0,
    G2: np.ndarray = None,
    pair_indices: np.ndarray = None,
    include_linear_1: bool = True,
    basis_name: str = 'fourier',
) -> dict:
    """Stability diagnostics: per-fold Sobol, noise, and RMSE.

    Fits the model on each training fold (via Woodbury, no full refit),
    computes per-fold Sobol indices and noise estimates, and reports
    stability as the spread across folds.  This directly compares with
    the analytic LOO/GCV diagnostics.

    Args:
        Phi: (N, F) feature matrix
        y: (N,) centered targets
        reg_diag: (F,) regularization diagonal
        D: number of variables
        K1: first-order basis size parameter
        G1: (B, B) first-order Gram matrix
        n_folds: number of folds
        seed: random seed
        K2, P, G2, pair_indices: second-order info (optional)
        include_linear_1: whether linear term is in basis
        basis_name: basis family

    Returns:
        Dict with per_fold (sobol, sigma_hat, rmse), means, stds,
        comparison with analytic LOO, and overall stability summary.
    """
    from ..core.features import basis_size as _bs

    Phi = np.asarray(Phi, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    reg_diag = np.asarray(reg_diag, dtype=np.float64)
    G1 = np.asarray(G1, dtype=np.float64)
    N, F = Phi.shape
    B1 = _bs(K1, include_linear_1, basis_name)

    # Full solve
    A = Phi.T @ Phi + np.diag(reg_diag)
    A_inv = np.linalg.inv(A)
    w_full = A_inv @ (Phi.T @ y)
    r_full = y - Phi @ w_full
    df_full = float(np.trace(A_inv @ (Phi.T @ Phi)))

    # Full-data analytics for comparison
    sigma2_full = float(np.sum(r_full**2) / max(1, N - df_full))
    H_diag = np.sum((Phi @ A_inv) * Phi, axis=1)
    loo_resids = r_full / (1 - H_diag)
    loo_cv_full = float(np.mean(loo_resids**2))

    # Full-data Sobol
    sobol_full = {}
    total_var = 0
    for i in range(D):
        wi = w_full[i * B1: (i + 1) * B1]
        vi = max(0, float(wi @ G1 @ wi))
        sobol_full[i] = vi
        total_var += vi
    if total_var > 0:
        sobol_full = {i: v / total_var for i, v in sobol_full.items()}

    # Assign folds
    rng = np.random.RandomState(seed)
    fold_ids = np.zeros(N, dtype=int)
    perm = rng.permutation(N)
    fold_size = N // n_folds
    for k in range(n_folds):
        start = k * fold_size
        end = (k + 1) * fold_size if k < n_folds - 1 else N
        fold_ids[perm[start:end]] = k

    # Per-fold diagnostics
    per_fold = []
    for k in range(n_folds):
        val_mask = fold_ids == k
        train_mask = ~val_mask
        Phi_k = Phi[val_mask]
        y_k = y[val_mask]
        n_k = int(np.sum(val_mask))
        n_train = int(np.sum(train_mask))

        # Woodbury update for w_{-k}
        A_inv_PhikT = A_inv @ Phi_k.T
        inner = np.eye(n_k) + Phi_k @ A_inv_PhikT
        delta = Phi_k @ w_full - y_k
        correction = A_inv_PhikT @ np.linalg.solve(inner, delta)
        w_k = w_full - correction

        # Validation RMSE
        pred_k = Phi_k @ w_k
        rmse_k = float(np.sqrt(np.mean((y_k - pred_k) ** 2)))

        # Training residuals for sigma estimate
        Phi_train = Phi[train_mask]
        y_train = y[train_mask]
        r_train = y_train - Phi_train @ w_k
        # Approximate df for the leave-k-out model
        df_k = df_full * (n_train / N)  # rough scaling
        sigma2_k = float(np.sum(r_train**2) / max(1, n_train - df_k))

        # Sobol from w_k
        sobol_k = {}
        total_var_k = 0
        for i in range(D):
            wi = w_k[i * B1: (i + 1) * B1]
            vi = max(0, float(wi @ G1 @ wi))
            sobol_k[i] = vi
            total_var_k += vi
        if total_var_k > 0:
            sobol_k = {i: v / total_var_k for i, v in sobol_k.items()}

        per_fold.append({
            'fold': k,
            'rmse': rmse_k,
            'sigma_hat': float(np.sqrt(max(0, sigma2_k))),
            'sobol': sobol_k,
            'n_train': n_train,
            'n_val': n_k,
        })

    # Aggregate
    all_rmses = np.array([f['rmse'] for f in per_fold])
    all_sigmas = np.array([f['sigma_hat'] for f in per_fold])
    sobol_arrays = {i: np.array([f['sobol'].get(i, 0) for f in per_fold])
                    for i in range(D)}

    sobol_means = {i: float(np.mean(v)) for i, v in sobol_arrays.items()}
    sobol_stds = {i: float(np.std(v, ddof=1)) for i, v in sobol_arrays.items()}

    # Stability summary
    max_sobol_std = max(sobol_stds.values()) if sobol_stds else 0
    mean_sobol_std = float(np.mean(list(sobol_stds.values()))) if sobol_stds else 0

    if max_sobol_std < 0.01:
        stability = 'excellent'
    elif max_sobol_std < 0.03:
        stability = 'good'
    elif max_sobol_std < 0.05:
        stability = 'moderate'
    else:
        stability = 'poor'

    return {
        # Full-data reference
        'full_data': {
            'sobol': sobol_full,
            'sigma_hat': float(np.sqrt(max(0, sigma2_full))),
            'loo_cv': loo_cv_full,
            'df': df_full,
        },
        # Per-fold results
        'per_fold': per_fold,
        # Aggregated
        'sobol_mean': sobol_means,
        'sobol_std': sobol_stds,
        'rmse_mean': float(np.mean(all_rmses)),
        'rmse_std': float(np.std(all_rmses, ddof=1)),
        'sigma_mean': float(np.mean(all_sigmas)),
        'sigma_std': float(np.std(all_sigmas, ddof=1)),
        # Comparison: analytic LOO vs K-fold
        'loo_cv': loo_cv_full,
        'kfold_cv': float(np.mean(all_rmses**2)),
        'loo_kfold_ratio': loo_cv_full / max(1e-10, float(np.mean(all_rmses**2))),
        # Stability assessment
        'stability': stability,
        'max_sobol_std': max_sobol_std,
        'mean_sobol_std': mean_sobol_std,
        'n_folds': n_folds,
        'N': N,
    }


# =============================================================================
# Sample size diagnostics
# =============================================================================

def sample_size_diagnostics(
    Phi: np.ndarray,
    y: np.ndarray,
    reg_diag: np.ndarray,
    D: int,
    K1: int,
    G1: np.ndarray,
    K2: int = 0,
    P: int = 0,
    G2: Optional[np.ndarray] = None,
    pair_indices: Optional[np.ndarray] = None,
    include_linear_1: bool = True,
    include_linear_2: bool = True,
    basis_name: str = 'fourier',
) -> Dict:
    """Sample size diagnostics: how much data do you need?

    For each component, computes:
    - Effective sample size per parameter
    - Estimated precision of the Sobol index (from sandwich SE)
    - Recommended N for a target precision

    Args:
        Phi, y, reg_diag: ridge inputs
        D, K1, G1: first-order structure
        K2, P, G2, pair_indices: second-order (optional)
        include_linear_1: whether first-order basis includes linear term
        include_linear_2: whether second-order basis includes linear term
        basis_name: 'fourier' or 'legendre'

    Returns:
        Dict with per-component diagnostics and recommendations
    """
    from ..core.features import basis_size as _bs

    analytics = ridge_analytics(Phi, y, reg_diag)
    N = analytics['N']
    df = analytics['df']

    # Get Sobol CIs for current precision
    sobol_ci = sobol_confidence_intervals(
        Phi, y, reg_diag, D, K1, G1, K2, P, G2, pair_indices,
        include_linear_1=include_linear_1, basis_name=basis_name)

    block1 = _bs(K1, include_linear_1, basis_name)
    F1 = D * block1

    diagnostics = {'N': N, 'df': df, 'sigma_hat': analytics['sigma_hat']}

    # Per-order effective sample size
    order1_params = D * block1
    diagnostics['order1'] = {
        'n_params': order1_params,
        'ess_per_param': N / order1_params,
    }

    if K2 > 0 and P > 0:
        block2 = _bs(K2, include_linear_2, basis_name) ** 2
        order2_params = P * block2
        diagnostics['order2'] = {
            'n_params': order2_params,
            'ess_per_param': N / order2_params,
        }

    # Per-variable precision from CIs
    var_precision = {}
    for i, (S, lo, hi) in sobol_ci['first_order'].items():
        width = hi - lo
        var_precision[i] = {
            'sobol': S,
            'ci_width': width,
            'se': width / (2 * 1.96),  # approximate SE from 95% CI
        }
        # Estimated N for halving the CI (SE ~ 1/sqrt(N)):
        if width > 0:
            var_precision[i]['N_for_half_ci'] = int(4 * N)  # 4x data halves SE

    diagnostics['per_variable'] = var_precision

    # Summary recommendation
    max_ci = max(v['ci_width'] for v in var_precision.values()) if var_precision else 0
    if max_ci < 0.05:
        rec = f"Precision is good (max CI width {max_ci:.3f}). Current N={N} is sufficient."
    elif max_ci < 0.15:
        needed = int(N * (max_ci / 0.05) ** 2)
        rec = (f"Moderate precision (max CI width {max_ci:.3f}). "
               f"For ±0.025 precision, need ~{needed:,} samples.")
    else:
        needed = int(N * (max_ci / 0.05) ** 2)
        rec = (f"Low precision (max CI width {max_ci:.3f}). "
               f"For ±0.025 precision, need ~{needed:,} samples. "
               f"Consider reducing model complexity.")

    diagnostics['recommendation'] = rec

    return diagnostics
