"""Analytic hyperparameter optimization for ridge regression.

Provides GCV, marginal likelihood (evidence), AIC/BIC, and gradient-based
optimization — all closed-form for ridge regression.

For structured regularization R = diag(reg_vector), we use direct
computation of the hat matrix trace for effective degrees of freedom.

When F > N (overparameterized), the evidence uses the dual form
(N x N matrix instead of F x F) for numerical stability.
GCV is recommended as the default when F > N.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from scipy.optimize import minimize_scalar, minimize


def _log_profile_evidence_dual(Phi: np.ndarray, y: np.ndarray,
                               reg_diag: np.ndarray) -> float:
    """Profile log marginal likelihood using the dual form (N x N).

    Profiles out sigma^2 analytically, giving a clean function of lambda only.

    K(lambda) = Phi R^{-1} Phi^T + I   (N x N, with R = diag(reg_diag))
    sigma^2_profile = y^T K^{-1} y / N
    log p(y|lambda) = -N/2 log(sigma^2) - 1/2 log|K| - N/2 (1 + log 2pi)

    Used when F > N for numerical stability.
    """
    N, F = Phi.shape

    # Build R^{-1}: for near-zero entries, cap at large finite value
    reg_inv = np.where(reg_diag > 1e-12, 1.0 / reg_diag, 1e12)

    # K = Phi diag(reg_inv) Phi^T + I   (N x N)
    Phi_scaled = Phi * np.sqrt(reg_inv)[None, :]
    K = Phi_scaled @ Phi_scaled.T + np.eye(N)

    sign, logdet_K = np.linalg.slogdet(K)
    if sign <= 0:
        return -np.inf

    # Profile sigma^2 = y^T K^{-1} y / N
    K_inv_y = np.linalg.solve(K, y)
    sigma2_profile = max(1e-15, float(y @ K_inv_y) / N)

    log_ev = (-N / 2.0 * np.log(sigma2_profile)
              - 0.5 * logdet_K
              - N / 2.0 * (1.0 + np.log(2 * np.pi)))
    return float(log_ev)


def _log_profile_evidence_primal(Phi: np.ndarray, y: np.ndarray,
                                 reg_diag: np.ndarray,
                                 w: np.ndarray) -> float:
    """Profile log marginal likelihood using the primal form (F x F).

    Profiles out sigma^2 analytically. Uses Sylvester's determinant
    theorem and the Woodbury identity for efficient computation:

    sigma^2_profile = y^T r / N    where r = y - Phi w*
                                   (NOT RSS/N = r^T r / N)
    log|K| = log|Phi^T Phi + R| - log|R|    (Sylvester)
    log p(y|lambda) = -N/2 log(sigma^2) - 1/2 log|K| - N/2 (1 + log 2pi)

    Used when F <= N.
    """
    N, F = Phi.shape
    r = y - Phi @ w

    # Profile sigma^2 = y^T K^{-1} y / N = y^T r / N
    # (via Woodbury: K^{-1}y = y - Phi(Phi^T Phi + R)^{-1} Phi^T y = y - Phi w* = r)
    sigma2_profile = max(1e-15, float(y @ r) / N)

    # log|K| via Sylvester: |I + Phi R^{-1} Phi^T| = |Phi^T Phi + R| / |R|
    A = Phi.T @ Phi + np.diag(reg_diag)
    sign, logdet_A = np.linalg.slogdet(A)
    pos_reg = reg_diag[reg_diag > 1e-15]
    logdet_R = np.sum(np.log(pos_reg)) if len(pos_reg) > 0 else 0.0
    logdet_K = logdet_A - logdet_R

    log_ev = (-N / 2.0 * np.log(sigma2_profile)
              - 0.5 * logdet_K
              - N / 2.0 * (1.0 + np.log(2 * np.pi)))
    return float(log_ev)


def ridge_solve_with_diagnostics(Phi: np.ndarray, y: np.ndarray,
                                 reg_diag: np.ndarray) -> Dict:
    """Solve ridge regression and return all diagnostics.

    Solves w* = (Phi^T Phi + R)^{-1} Phi^T y  where R = diag(reg_diag)

    Returns w*, effective df, RSS, MSE, GCV, AIC, BIC, log-evidence.
    Automatically selects dual-form evidence when F > N.
    """
    Phi = np.asarray(Phi, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    reg_diag = np.asarray(reg_diag, dtype=np.float64)
    N, F = Phi.shape

    # Form the normal equations matrix
    PhiTPhi = Phi.T @ Phi
    A = PhiTPhi + np.diag(reg_diag)

    # Solve
    PhiTy = Phi.T @ y
    w = np.linalg.solve(A, PhiTy)

    # Residuals
    residuals = y - Phi @ w
    rss_val = float(np.sum(residuals**2))
    mse_val = rss_val / N

    # Effective degrees of freedom: df = tr((Phi^T Phi + R)^{-1} Phi^T Phi)
    A_inv = np.linalg.inv(A)
    df = float(np.trace(A_inv @ PhiTPhi))

    # GCV
    gcv_denom = max(1e-10, 1.0 - df / N)
    gcv_val = (rss_val / N) / gcv_denom ** 2

    # AIC, BIC
    aic_val = N * np.log(max(mse_val, 1e-15)) + 2.0 * df
    bic_val = N * np.log(max(mse_val, 1e-15)) + np.log(N) * df

    # Profile log marginal likelihood — sigma^2 profiled out analytically.
    # Uses dual form (N x N) when F > N for numerical stability.
    sigma2_ml = max(mse_val, 1e-15)
    if F > N:
        log_evidence = _log_profile_evidence_dual(Phi, y, reg_diag)
    else:
        log_evidence = _log_profile_evidence_primal(Phi, y, reg_diag, w)

    return {
        'w': w,
        'rss': rss_val,
        'mse': mse_val,
        'df': df,
        'gcv': gcv_val,
        'aic': aic_val,
        'bic': bic_val,
        'log_evidence': log_evidence,
        'sigma2_ml': sigma2_ml,
    }


def optimize_single_lambda(Phi: np.ndarray, y: np.ndarray,
                           reg_structure: np.ndarray,
                           method: str = 'gcv',
                           bounds: Tuple[float, float] = (1e-6, 1e2),
                           n_grid: int = 50) -> Dict:
    """Find optimal scalar lambda for R = lambda * diag(reg_structure).

    Uses grid search on log-scale followed by refinement.

    Note: when F > N, 'gcv' is recommended over 'evidence' because
    the evidence can over-regularize in the overparameterized regime.

    Args:
        Phi: (N, F) feature matrix
        y: (N,) centered targets
        reg_structure: (F,) relative penalty weights
        method: 'gcv', 'evidence', 'aic', or 'bic'
        bounds: search range for lambda
        n_grid: number of grid points

    Returns:
        Dict with lambda_opt and diagnostics at the optimum.
    """
    Phi = np.asarray(Phi, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    reg_structure = np.asarray(reg_structure, dtype=np.float64)
    N, F = Phi.shape

    # When F > N, evidence optimization is expensive (N x N system per lambda)
    # and can over-regularize. Fall back to GCV with a warning.
    if method == 'evidence' and F > N:
        import warnings
        warnings.warn(
            f"Evidence optimization with F={F} > N={N} is expensive and may "
            f"over-regularize. Falling back to GCV. Use ridge_solve_with_diagnostics "
            f"for single-point evidence evaluation.",
            stacklevel=2
        )
        method = 'gcv'

    # Precompute
    PhiTPhi = Phi.T @ Phi
    PhiTy = Phi.T @ y

    def evaluate(lam):
        reg_diag = lam * reg_structure
        diag = ridge_solve_with_diagnostics(Phi, y, reg_diag)
        return diag

    def score(diag):
        if method == 'gcv':
            return diag['gcv']
        elif method == 'evidence':
            return -diag['log_evidence']
        elif method == 'aic':
            return diag['aic']
        elif method == 'bic':
            return diag['bic']
        return diag['gcv']

    # Grid search
    lambdas = np.logspace(np.log10(bounds[0]), np.log10(bounds[1]), n_grid)
    scores = []
    evals = []
    for lam in lambdas:
        r = evaluate(lam)
        evals.append(r)
        scores.append(score(r))

    best_idx = int(np.argmin(scores))

    # Refine with bounded scalar optimization
    log_lo = np.log10(lambdas[max(0, best_idx - 2)])
    log_hi = np.log10(lambdas[min(n_grid - 1, best_idx + 2)])

    result = minimize_scalar(
        lambda log_lam: score(evaluate(10**log_lam)),
        bounds=(log_lo, log_hi), method='bounded'
    )
    lam_opt = 10**result.x

    # Get full diagnostics at optimum
    final = evaluate(lam_opt)
    final['lambda_opt'] = lam_opt
    final['converged'] = True
    return final


def optimize_multi_lambda(Phi: np.ndarray, y: np.ndarray,
                          D: int, K1: int, K2: int = 0, P: int = 0,
                          strategy: str = 'variance',
                          method: str = 'gcv',
                          bounds: Tuple[float, float] = (1e-6, 1e2)) -> Dict:
    """Optimize (lambda_1, lambda_2) jointly via GCV or evidence.

    Args:
        Phi: (N, F) full feature matrix
        y: (N,) centered targets
        D, K1, K2, P: model structure parameters
        strategy: regularization strategy
        method: 'gcv' or 'evidence'
        bounds: search range for each lambda

    Returns:
        Dict with optimal lambdas and diagnostics.
    """
    from .regularization import build_regularization_vector

    Phi_np = np.asarray(Phi, dtype=np.float64)
    y_np = np.asarray(y, dtype=np.float64)
    N, F = Phi_np.shape

    if K2 == 0 or P == 0:
        reg_struct = np.asarray(
            build_regularization_vector(D, K1, 0, 0, strategy, 1.0, 1.0),
            dtype=np.float64
        )
        return optimize_single_lambda(Phi_np, y_np, reg_struct, method, bounds)

    def evaluate(lam1, lam2):
        reg = np.asarray(
            build_regularization_vector(D, K1, K2, P, strategy, lam1, lam2),
            dtype=np.float64
        )
        return ridge_solve_with_diagnostics(Phi_np, y_np, reg)

    def objective(log_lams):
        lam1, lam2 = 10**log_lams[0], 10**log_lams[1]
        r = evaluate(lam1, lam2)
        if method == 'gcv':
            return r['gcv']
        else:
            return -r['log_evidence']

    x0 = np.array([np.log10(0.001), np.log10(0.01)])
    log_bounds = [(np.log10(bounds[0]), np.log10(bounds[1]))] * 2
    result = minimize(objective, x0, method='L-BFGS-B', bounds=log_bounds)

    lam1_opt, lam2_opt = 10**result.x[0], 10**result.x[1]
    final = evaluate(lam1_opt, lam2_opt)
    final['lambda_order1'] = lam1_opt
    final['lambda_order2'] = lam2_opt
    final['converged'] = result.success
    return final


def optimize_multi_lambda_extended(
    Phi: np.ndarray,
    y: np.ndarray,
    D: int, K1: int,
    K2: int = 0, P: int = 0,
    K3: int = 0, T: int = 0,
    M_residual: int = 0,
    strategy: str = 'variance',
    method: str = 'gcv',
    bounds: Tuple[float, float] = (1e-6, 1e2),
    verbose: bool = True,
) -> Dict:
    """Optimize (lambda1, lambda2, lambda3, lambda_res) jointly via GCV.

    Automatically determines which lambdas are active based on K2, K3,
    M_residual. Optimizes only over active lambdas.

    For models with projected residual features (Phi^T Z_proj = 0),
    the Fourier and residual lambdas decouple. This function still
    optimizes jointly (correct answer either way), but could be split
    for efficiency.

    Args:
        Phi: (N, F) full feature matrix [Phi1 | Phi2 | Phi3 | Z_proj]
        y: (N,) centered targets
        D, K1, K2, P, K3, T, M_residual: model structure
        strategy: regularization strategy
        method: 'gcv', 'aic', 'bic', or 'evidence'
        bounds: search range per lambda
        verbose: print optimization progress

    Returns:
        Dict with optimal lambdas and diagnostics at the optimum.
    """
    from .regularization import build_regularization_vector

    Phi_np = np.asarray(Phi, dtype=np.float64)
    y_np = np.asarray(y, dtype=np.float64)
    N, F = Phi_np.shape

    # Determine which lambdas are active
    active_names = ['lambda_order1']
    if K2 > 0 and P > 0:
        active_names.append('lambda_order2')
    if K3 > 0 and T > 0:
        active_names.append('lambda_order3')
    if M_residual > 0:
        active_names.append('lambda_residual')

    n_active = len(active_names)

    # Fallback to simpler optimizers for 1-2 lambdas
    if n_active == 1:
        reg_struct = np.asarray(
            build_regularization_vector(D, K1, 0, 0, strategy, 1.0, 1.0),
            dtype=np.float64
        )
        result = optimize_single_lambda(Phi_np, y_np, reg_struct, method, bounds)
        result['lambda_order1'] = result['lambda_opt']
        return result

    if n_active == 2 and 'lambda_residual' not in active_names:
        return optimize_multi_lambda(Phi_np, y_np, D, K1, K2, P,
                                     strategy, method, bounds)

    # --- General n-lambda optimization ---
    # Default initial values (log10 scale)
    defaults = {
        'lambda_order1': -3.0,   # 0.001
        'lambda_order2': -2.0,   # 0.01
        'lambda_order3': -1.0,   # 0.1
        'lambda_residual': 0.0,  # 1.0
    }

    x0 = np.array([defaults[name] for name in active_names])
    log_bounds = [(np.log10(bounds[0]), np.log10(bounds[1]))] * n_active

    def score(diag):
        if method == 'gcv':
            return diag['gcv']
        elif method == 'aic':
            return diag['aic']
        elif method == 'bic':
            return diag['bic']
        elif method == 'evidence':
            return -diag['log_evidence']
        return diag['gcv']

    def objective(log_lams):
        lam_dict = {}
        for i, name in enumerate(active_names):
            lam_dict[name] = 10 ** log_lams[i]

        reg = np.asarray(build_regularization_vector(
            D, K1, K2, P, strategy,
            lam_dict.get('lambda_order1', 0.001),
            lam_dict.get('lambda_order2', 0.01),
            K3=K3, T=T,
            lambda_order3=lam_dict.get('lambda_order3', 0.1),
            M_residual=M_residual,
            lambda_residual=lam_dict.get('lambda_residual', 1.0),
        ), dtype=np.float64)

        diag = ridge_solve_with_diagnostics(Phi_np, y_np, reg)
        return score(diag)

    # Grid search for initial point (coarse)
    if n_active <= 3:
        n_grid_per = 8
        from itertools import product as iterproduct
        grid_1d = np.linspace(np.log10(bounds[0]), np.log10(bounds[1]), n_grid_per)
        best_score = float('inf')
        best_x = x0.copy()

        for combo in iterproduct(*[grid_1d] * n_active):
            s = objective(np.array(combo))
            if s < best_score:
                best_score = s
                best_x = np.array(combo)
        x0 = best_x
    else:
        # For 4+ lambdas, grid search is too expensive; use defaults
        pass

    # Refine with L-BFGS-B
    result = minimize(objective, x0, method='L-BFGS-B', bounds=log_bounds,
                      options={'maxiter': 100, 'ftol': 1e-8})

    # Evaluate at optimum
    opt_lams = {}
    for i, name in enumerate(active_names):
        opt_lams[name] = 10 ** result.x[i]

    reg_opt = np.asarray(build_regularization_vector(
        D, K1, K2, P, strategy,
        opt_lams.get('lambda_order1', 0.001),
        opt_lams.get('lambda_order2', 0.01),
        K3=K3, T=T,
        lambda_order3=opt_lams.get('lambda_order3', 0.1),
        M_residual=M_residual,
        lambda_residual=opt_lams.get('lambda_residual', 1.0),
    ), dtype=np.float64)

    final = ridge_solve_with_diagnostics(Phi_np, y_np, reg_opt)
    final.update(opt_lams)
    final['converged'] = result.success
    final['n_lambdas_optimized'] = n_active
    final['lambda_names'] = active_names

    if verbose:
        parts = [f"{name}={opt_lams[name]:.2e}" for name in active_names]
        print(f"  Multi-lambda GCV: {', '.join(parts)} "
              f"({method}={score(final):.6f}, df={final['df']:.1f})")

    return final
