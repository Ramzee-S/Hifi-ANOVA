"""Fixed-configuration ridge analytics, uncertainty, and diagnostics.

The linear-in-parameters structure of the HiFi-ANOVA model enables a complete AutoML
pipeline without expensive retraining loops:

  1. Exact LOO-CV from hat matrix leverages (exact *at the fixed, finalized
     lambda*: when lambda was itself selected on the same data, e.g. by GCV,
     the per-fold re-tuning is not replayed, so LOO is mildly optimistic)
  2. Noise estimation: sigma^2 = RSS / (N - 2 tr(H) + tr(H^2))  (residual df)
  3. Sandwich estimator: bootstrap-quality confidence intervals on Sobol indices
  4. K-fold CV via Woodbury rank updates (k inversions of N/k matrices)
  5. Sample size diagnostics: effective N per parameter, precision estimates

For a fixed design, admitted structure, penalties, and weights, these quantities
reuse the finalized ridge factorization and its hat matrix. Selection, penalty
optimization, or changing Stage-D weights requires earlier solves.

Usage:
    from hifi_anova.analysis.automl import ridge_analytics, sobol_confidence_intervals

    # Full analytics from one fixed-configuration ridge factorization
    analytics = ridge_analytics(Phi, y, reg_diag)
    print(f"sigma_hat = {analytics['sigma_hat']:.4f}")
    print(f"LOO-CV = {analytics['loo_cv']:.4f}")

    # Sobol indices with confidence intervals
    sobol_ci = sobol_confidence_intervals(Phi, y, reg_diag, D, K1, G1)
    for i, (s, lo, hi) in sobol_ci['first_order'].items():
        print(f"  x{i+1}: S = {s:.3f} [{lo:.3f}, {hi:.3f}]")
"""

import numpy as np
from typing import Dict, Optional, Tuple

from ..model.variance_model import LOG_VAR_CLIP
from ..linalg import spd_inverse
from .._result_aliases import ResultAliasDict

_LOG2PI = float(np.log(2.0 * np.pi))


# =============================================================================
# Shared Woodbury k-fold kernel
# =============================================================================

def _woodbury_downdate(A_inv: np.ndarray, Phi_k: np.ndarray,
                       w_full: np.ndarray, y_k: np.ndarray):
    """Leave-fold-``k``-out ridge coefficients without a refit (Woodbury).

    Removing fold ``k`` is a rank-``n_k`` downdate of ``A = Phi^T Phi + R``:

        w_{-k} = w + A^{-1} Phi_k^T (I - Phi_k A^{-1} Phi_k^T)^{-1} (Phi_k w - y_k).

    Returns ``(w_minus_k, A_inv_PhikT, inner)`` — the last two are the reusable
    factors ``A^{-1} Phi_k^T`` and ``I - H_k`` so callers can form the fold
    smoother ``S_k = (I - H_k)^{-1} (A^{-1} Phi_k^T)^T`` without recomputing them.
    Shared by ``kfold_cv_analytic`` and ``stability_diagnostics`` (one kernel; the
    two must not drift).
    """
    A_inv_PhikT = A_inv @ Phi_k.T                         # (F, n_k)
    inner = np.eye(Phi_k.shape[0]) - Phi_k @ A_inv_PhikT  # I - H_k, (n_k, n_k)
    delta = Phi_k @ w_full - y_k                          # (n_k,)
    correction = A_inv_PhikT @ np.linalg.solve(inner, delta)   # (F,)
    return w_full + correction, A_inv_PhikT, inner


# =============================================================================
# Core fixed-configuration ridge analytics
# =============================================================================

def ridge_analytics(
    Phi: np.ndarray,
    y: np.ndarray,
    reg_diag: np.ndarray,
    weights: Optional[np.ndarray] = None,
    profile_intercept: bool = False,
    intercept_df: float = 1.0,
) -> Dict:
    """Complete analytical diagnostics from one fixed ridge factorization.

    Computes everything derivable from the hat matrix without refitting:
    - Coefficients w, residuals, RSS, MSE
    - Effective degrees of freedom df = tr(H)
    - Noise estimate sigma_hat^2 = RSS / (N - 2 tr(H) + tr(H^2))
    - Per-observation leverages H_ii (diagonal of hat matrix)
    - Exact LOO-CV (from leverages; exact conditional on the fixed lambda —
      data-dependent lambda selection is not re-run per fold)
    - GCV, AIC, BIC, profile evidence
    - Effective sample size per parameter

    With ``weights`` (Stage-D GLS precision ``W = diag(1/sigma^2(x_n))``) the
    diagnostics become the *weighted* (GLS) ones the theory manuscript defines
    (``Manuscript_Theoryv05.tex``, App. loo): ``A = Phi^T W Phi + R``, hat
    diagonal ``a_n = w_n phi_n^T A^{-1} phi_n`` (``sum a_n = tr S = df``), weighted
    ``RSS_w = sum w_n r_n^2``, whitened residual df ``N - 2 tr S + tr(S^2)`` with
    ``S`` the whitened smoother, weighted PRESS/GCV, and — importantly —
    ``sigma_hat`` becomes the *whitened calibration scale* ``sqrt(RSS_w/df_res)``
    (≈ 1 when the variance model is calibrated), NOT a homoscedastic noise level
    (see ``noise_scale_is_calibration`` in the returned dict). ``weights=None`` is
    byte-for-byte the original unweighted path.

    ``profile_intercept=True`` fits an *unpenalized* intercept jointly with the
    nonconstant slopes, exactly matching the Stage-D profiled joint-GLS mean
    (Manuscript_Theoryv07 Remark ``rem:intercept``: the mean identities use the
    augmented design ``Z=[1,Phi]``, coefficient ``(f0, w)``, penalty
    ``diag(0, R)``). Every diagnostic — df, leverage, residual df, sigma, PRESS/LOO,
    GCV/AIC/BIC — is then computed from the augmented smoother, so a response
    perturbation or a deleted row correctly RE-PROFILES the intercept rather than
    holding the fitted ``f0`` fixed. The returned ``'w'`` is the NONCONSTANT slopes
    (intercept excluded, so Sobol component slices are unaffected); the profiled
    intercept is returned separately as ``'f0'``, the full coefficient as
    ``'theta' = [f0, *w]``, and ``'A_inv'`` / ``'Z'`` are the AUGMENTED inverse
    ``(F+1, F+1)`` and design ``(N, F+1)`` for a coherent augmented sandwich.
    Pass the *uncentered* response (``y_centered + f0``) so ``'f0'`` recovers the
    fitted intercept; the slopes/residuals/df are invariant to a constant shift in
    ``y``. ``profile_intercept=False`` (default) is byte-for-byte the legacy
    feature-only path.

    Args:
        Phi: (N, F) feature matrix
        y: (N,) targets — centered (feature-only path) or uncentered when
            ``profile_intercept=True`` (the augmented fit profiles ``f0``).
        reg_diag: (F,) regularization diagonal
        weights: (N,) GLS precision weights; ``None`` ⇒ unweighted (homoscedastic)
        profile_intercept: fit an unpenalized intercept on the augmented design
            ``[1, Phi]`` (Stage-D profiled joint-GLS convention).
        intercept_df: df consumed by an intercept profiled OUTSIDE this call
            (the package convention passes y centered by its (weighted) mean).
            Added to tr(H) inside the GCV/AIC/BIC criteria only — the reported
            ``df`` stays tr(H). Set 0.0 if y was not centered. Ignored when
            ``profile_intercept=True`` (the augmented design counts its own
            intercept column).

    Returns:
        Dict with all diagnostics
    """
    if profile_intercept:
        return _augmented_ridge_analytics(Phi, y, reg_diag, weights)
    if weights is not None:
        return _ridge_analytics_weighted(Phi, y, reg_diag, weights,
                                         intercept_df=intercept_df)

    Phi = np.asarray(Phi, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    reg_diag = np.asarray(reg_diag, dtype=np.float64)
    N, F = Phi.shape

    # Ridge solve
    PhiTPhi = Phi.T @ Phi
    A = PhiTPhi + np.diag(reg_diag)
    A_inv = spd_inverse(A)
    w = A_inv @ (Phi.T @ y)

    # Residuals
    residuals = y - Phi @ w
    rss = float(np.sum(residuals ** 2))
    mse = rss / N

    # Effective degrees of freedom.
    #   df = tr(H) = tr(Phi A^{-1} Phi^T) = tr(A^{-1} Phi^T Phi) = tr(M)
    #   tr(H^2) = tr(M^2)                       (H symmetric here)
    # with M = A^{-1} Phi^T Phi (F x F, already cheap from the factors formed).
    M = A_inv @ PhiTPhi
    df = float(np.trace(M))                     # model complexity / effective df
    tr_H2 = float(np.sum(M * M.T))               # tr(H^2) = tr(M^2)

    # Noise estimate. The residual effective degrees of freedom is
    #   E[RSS] = sigma^2 (N - 2 tr(H) + tr(H^2)) + bias^2,
    # so sigma_hat^2 = RSS / (N - 2 tr(H) + tr(H^2)). The familiar N - tr(H) is
    # a shorthand that coincides only for a projection (H^2 = H); we use the exact
    # residual-df form throughout (ridge_analytics, the Sobol CIs, the manuscript)
    # for a single convention. See DEC-021.
    df_residual = max(N - 2.0 * df + tr_H2, 1.0)
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
    # Exact for the ridge fit at THIS fixed lambda. If lambda was selected on
    # the same data (GCV), the identity does not re-tune lambda per fold, so
    # the reported LOO is conditional-on-lambda and mildly optimistic.
    # Guard against leverage overflow: H_ii should be in [0, 1) for
    # regularized regression. Clip to avoid division by zero.
    leverages = np.clip(leverages, 0.0, 1.0 - 1e-10)
    loo_residuals = residuals / (1.0 - leverages)
    loo_cv = float(np.mean(loo_residuals ** 2))

    # Predictive LOO negative log-likelihood on the common Gaussian scale — the
    # cross-model-comparable criterion (Manuscript_Theoryv06 App. C). With a
    # single constant scale sigma_hat^2 the deleted variance is held fixed, so
    #   LOO-NLL = 1/2 log sigma_hat^2 + loo_cv/(2 sigma_hat^2) + 1/2 log 2pi
    # (essentially free from the PRESS). On the homoscedastic path the three LOO
    # tiers coincide (W = I), so this is tier-agnostic — reported as loo_tier = 1.
    # A Stage-D fit upgrades this to the Tier-II one-step jackknife via joint_loo.
    _s2 = max(sigma2_hat, 1e-300)
    loo_nll = float(0.5 * np.log(_s2) + loo_cv / (2.0 * _s2) + 0.5 * _LOG2PI)

    # GCV (average-leverage approximation of LOO) / AIC / BIC — the criteria
    # count the profiled (centered-out) intercept via intercept_df; the
    # reported 'df' stays tr(H).
    df_sel = df + intercept_df
    gcv_denom = max(1e-10, 1.0 - df_sel / N)
    gcv = (rss / N) / gcv_denom ** 2
    aic = N * np.log(max(mse, 1e-15)) + 2.0 * df_sel
    bic = N * np.log(max(mse, 1e-15)) + np.log(N) * df_sel

    # Effective sample size per parameter
    ess_per_param = N / max(df, 1.0)

    return ResultAliasDict({
        'w': w,
        'A_inv': A_inv,
        'residuals': residuals,
        'rss': rss,
        'mse': mse,
        'df': df,
        'tr_H2': tr_H2,
        'df_residual': df_residual,   # N - 2 tr(H) + tr(H^2): sigma denom & t-df
        'sigma2_hat': sigma2_hat,
        'sigma_hat': float(sigma_hat),
        'leverages': leverages,
        'loo_cv': loo_cv,
        'loo_residuals': loo_residuals,
        'loo_nll': loo_nll,
        'loo_tier': 1,
        'gcv': gcv,
        'aic': aic,
        'bic': bic,
        'ess_per_param': ess_per_param,
        'N': N,
        'F': F,
        # sigma_hat is a homoscedastic noise scale here (unit weights).
        'noise_scale_is_calibration': False,
    })


def _ridge_analytics_weighted(Phi, y, reg_diag, weights,
                              intercept_df: float = 1.0) -> Dict:
    """Weighted (GLS) diagnostics — see ``ridge_analytics`` for the semantics.

    Precision weights ``W = diag(w_n)`` (``w_n = 1/sigma^2(x_n)``) enter the
    normal equations, the hat matrix, the residual df, and the residual scale.
    The algebra mirrors the unweighted path with ``Phi^T Phi -> Phi^T W Phi`` and
    ``sum r^2 -> sum w_n r_n^2``; ``M = A^{-1} Phi^T W Phi`` gives ``df = tr S``
    and ``tr(S^2) = tr(M^2)`` for the whitened smoother ``S``. With ``w_n ≡ 1``
    every line reduces to the unweighted formula exactly.
    """
    Phi = np.asarray(Phi, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    reg_diag = np.asarray(reg_diag, dtype=np.float64)
    W = np.asarray(weights, dtype=np.float64)
    N, F = Phi.shape

    # Weighted ridge solve: A = Phi^T W Phi + R, w = A^{-1} Phi^T W y.
    WPhi = W[:, None] * Phi
    PhiTWPhi = Phi.T @ WPhi
    A = PhiTWPhi + np.diag(reg_diag)
    A_inv = spd_inverse(A)
    w = A_inv @ (Phi.T @ (W * y))

    residuals = y - Phi @ w
    rss_w = float(np.sum(W * residuals ** 2))        # weighted RSS = sum w_n r_n^2
    mse = rss_w / N                                  # weighted mean squared residual

    # df = tr S = tr(A^{-1} Phi^T W Phi); tr(S^2) = tr(M^2), M = A^{-1} Phi^T W Phi.
    M = A_inv @ PhiTWPhi
    df = float(np.trace(M))
    tr_H2 = float(np.sum(M * M.T))

    # Whitened residual df and the whitened residual scale (calibration meter).
    df_residual = max(N - 2.0 * df + tr_H2, 1.0)
    sigma2_hat = rss_w / df_residual
    sigma_hat = np.sqrt(sigma2_hat)

    # Weighted hat diagonal a_n = w_n phi_n^T A^{-1} phi_n (sum a_n = tr S = df).
    quad = np.sum((Phi @ A_inv) * Phi, axis=1)       # phi_n^T A^{-1} phi_n
    leverages = np.clip(W * quad, 0.0, 1.0 - 1e-10)

    # Weighted PRESS/LOO: PRESS = sum w_n (r_n/(1-a_n))^2, reported as PRESS/N so
    # it reduces to the unweighted mean LOO squared residual when W = I.
    loo_residuals = residuals / (1.0 - leverages)
    loo_cv = float(np.sum(W * loo_residuals ** 2) / N)

    # Tier-I predictive LOO-NLL (Manuscript_Theoryv06 App. C): the variance model
    # is held at its full-data value, so h_hat(x_n) = -log W_n, e^{-h_hat} = W_n,
    #   LOO-NLL_I = 1/N sum[ 1/2 h_hat(x_n) + 1/2 W_n r_{(-n)}^2 ] + 1/2 log 2pi
    #            = -1/(2N) sum log W_n + 1/2 loo_cv + 1/2 log 2pi.
    # This is the plug-in tier; the Stage-D one-call UPGRADES it to the default
    # Tier II (one-step variance jackknife) via joint_loo, which also carries the
    # variance-floor / conditioning flags. Standalone (no variance design) the
    # Tier-I value is returned and loo_tier = 1.
    loo_nll = float(-0.5 * np.mean(np.log(np.maximum(W, 1e-300)))
                    + 0.5 * loo_cv + 0.5 * _LOG2PI)

    # Criteria count the profiled (weighted-centered-out) intercept via
    # intercept_df; the reported 'df' stays tr(S).
    df_sel = df + intercept_df
    gcv_denom = max(1e-10, 1.0 - df_sel / N)
    gcv = (rss_w / N) / gcv_denom ** 2

    aic = N * np.log(max(mse, 1e-15)) + 2.0 * df_sel
    bic = N * np.log(max(mse, 1e-15)) + np.log(N) * df_sel
    ess_per_param = N / max(df, 1.0)

    return ResultAliasDict({
        'w': w,
        'A_inv': A_inv,
        'residuals': residuals,
        'rss': rss_w,
        'mse': mse,
        'df': df,
        'tr_H2': tr_H2,
        'df_residual': df_residual,
        'sigma2_hat': sigma2_hat,
        'sigma_hat': float(sigma_hat),
        'leverages': leverages,
        'loo_cv': loo_cv,
        'loo_residuals': loo_residuals,
        'loo_nll': loo_nll,
        'loo_tier': 1,
        'gcv': gcv,
        'aic': aic,
        'bic': bic,
        'ess_per_param': ess_per_param,
        'N': N,
        'F': F,
        'weights': W,
        # sigma_hat is the whitened calibration scale (≈1 when calibrated), not a
        # homoscedastic noise level; sigma^2(x) lives in the fitted variance model.
        'noise_scale_is_calibration': True,
    })


def _augmented_ridge_analytics(Phi, y, reg_diag, weights) -> Dict:
    """Profiled-intercept diagnostics on the augmented design ``Z = [1, Phi]``.

    The Stage-D joint-GLS mean profiles an *unpenalized* intercept: it solves
    ``min_(f0,w) sum W_n (y_n - f0 - phi_n^T w)^2 + w^T R w`` (weighted-centering
    both ``y`` and ``Phi`` is the closed-form profiling of that intercept). The
    correct diagnostic instrument is therefore the plain ridge on the augmented
    design with a zero-penalty intercept column (Manuscript_Theoryv07 Remark
    ``rem:intercept``): ``Z = [1, Phi]``, ``R_aug = diag(0, R)``,
    ``A_aug = Z^T W Z + R_aug``, ``H_aug = Z A_aug^{-1} Z^T W``. Because the ones
    column is just another (unpenalized) feature, EVERY hat-matrix identity the
    feature-only path already computes — ``df = tr(H_aug)``, the leverage diagonal
    (which now sums to ``df`` INCLUDING the intercept), whitened residual df,
    sigma, PRESS/LOO, GCV/AIC/BIC — is correct on ``Z`` verbatim. We reuse the
    feature-only kernels on ``Z`` and then split the coefficient back into the
    profiled intercept ``f0 = theta[0]`` and the nonconstant slopes
    ``w = theta[1:]`` (the intercept must NOT enter Sobol component slices or
    energy sums). ``A_inv`` / ``Z`` are surfaced so a coherent augmented sandwich
    can be formed and its slope-slope block extracted after the full calculation.
    """
    Phi = np.asarray(Phi, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    reg_diag = np.asarray(reg_diag, dtype=np.float64)
    N, F = Phi.shape

    Z = np.concatenate([np.ones((N, 1), dtype=np.float64), Phi], axis=1)
    reg_aug = np.concatenate([[0.0], reg_diag])

    # The augmented fit is an ordinary (weighted) ridge on Z with a zero-penalty
    # intercept column — reuse the feature-only kernels so the augmented smoother
    # and the legacy smoother share one implementation (they must not drift).
    # intercept_df=0.0: the augmented design's zero-penalty intercept column is
    # counted by tr(H_aug) itself — adding the profiled-intercept df would
    # double-count it.
    base = (_ridge_analytics_weighted(Z, y, reg_aug, weights, intercept_df=0.0)
            if weights is not None
            else ridge_analytics(Z, y, reg_aug, intercept_df=0.0))

    theta = base['w']                       # (F+1,) = [f0, *slopes]
    base['theta'] = theta
    base['f0'] = float(theta[0])
    base['w'] = theta[1:]                   # nonconstant slopes only (Sobol slicing)
    base['Z'] = Z                           # augmented design for the sandwich
    base['F'] = F                           # nonconstant feature count (not F+1)
    base['intercept_profiled'] = True
    return base


# =============================================================================
# Joint heteroscedastic LOO: Tier-II one-step jackknife + Tier-III oracle
# =============================================================================

def joint_loo(
    analytics: Dict,
    variance,
    *,
    delta_cap: float = 5.0,
    cond_tol: float = 1e-10,
    floor_rtol: float = 1e-6,
) -> Dict:
    """Tier-II one-step-jackknife LOO of the joint heteroscedastic model.

    Upgrades the Tier-I LOO of a *weighted* ``analytics`` (from
    ``ridge_analytics(..., weights=W)``) to the manuscript's default **Tier II**
    (``Manuscript_Theoryv06`` App. C): the variance block is corrected by one
    Newton step per deleted observation,

        theta_hat_{(-n)} ~= theta_hat + H_h^{-1} grad ell_n,

    where ``ell_n = 1/2 h_n + 1/2 r_n^2 e^{-h_n}`` is the per-point NLL **loss**
    the Newton solver MINIMIZES (so the sign is ``+``) and ``H_h`` is the
    penalized variance-block Newton Hessian — it **includes the lambda_h penalty
    block** (check 1). Because ``grad ell_n = 1/2 (1 - rho_n) psi_n`` with
    ``rho_n = r_n^2/sigma_n^2``, the deleted prediction is a one-liner,

        delta h_n = psi_n^T H_h^{-1} grad ell_n = 1/2 (1 - rho_n) q_n,
        q_n = psi_n^T H_h^{-1} psi_n   (a variance-design leverage; O(N F_h^2)),

    which correctly makes a large-residual point (rho_n > 1) *lower* its own
    deleted variance (delta h_n < 0), penalizing the held-out residual more — the
    anti-optimism correction Tier I lacks.

    The mean block is held at its full-data-weight deleted residual
    ``r_{(-n)} = r_n/(1-a_n)`` (the Tier-II *hybrid* of check 2; the
    deleted-weights -> perturbed-mean cross-term is ``O_p(N^-2)`` per point and
    ``O(N^-1)`` at criterion level — validate against ``exact_loo_nll``). Returns
    the Tier-II predictive ``loo_nll`` (cross-model comparable), the
    Tier-II-weighted squared-error ``loo_cv``, ``loo_tier = 2``, and the
    regularity flags (KKT variance-floor test + ``H_h`` conditioning).

    Args:
        analytics: the weighted ``ridge_analytics`` dict (uses ``residuals``,
            ``loo_residuals``, ``weights``).
        variance: a :class:`hifi_anova.training.fitted_design.VarianceDesign`
            (``Psi``, ``reg_var``, ``w_h``, ``h0``).
        delta_cap: cap on ``|delta h_n|`` in log-variance units. A one-step
            extrapolation can blow up ``e^{-h}`` for a high-influence point
            (check 3); larger corrections are clipped and counted — and a large
            correction is exactly where the one-step tier is least trustworthy.
        cond_tol: reciprocal-condition-number floor for ``H_h``. The ``O_p(N^-2)``
            rate assumes a uniformly nonsingular Hessian; below this the Tier-II
            guarantee is flagged as not holding.
        floor_rtol: relative tolerance for "at the log-variance clip".

    Returns:
        dict with ``loo_nll``, ``loo_cv``, ``loo_tier`` (=2),
        ``loo_variance_floor_active``, ``variance_hessian_ill_conditioned``,
        ``loo_tier2_guarantee_holds``, ``loo_nll_correction_clipped``,
        ``n_correction_clipped``, ``h_rcond``.
    """
    r = np.asarray(analytics['residuals'], dtype=np.float64)          # mean-block r_n
    r_loo = np.asarray(analytics['loo_residuals'], dtype=np.float64)  # r_{(-n)}=r_n/(1-a_n)
    W = np.asarray(analytics['weights'], dtype=np.float64)            # W_n = 1/sigma_n^2
    N = r.shape[0]

    Psi = np.asarray(variance.Psi, dtype=np.float64)
    reg_var = np.asarray(variance.reg_var, dtype=np.float64)
    # Augmented variance design [1, Psi] and penalty [0, reg_var]: h0 is an
    # unpenalized fixed effect, exactly as newton_solve_log_variance augments.
    Psi_aug = np.concatenate([np.ones((N, 1)), Psi], axis=1)
    reg_aug = np.concatenate([[0.0], reg_var])

    # Recover the fitted log-variance: h_raw = -log W_n (unclipped, for the floor
    # test); h_c is the prediction-consistent clipped value used for sigma_n^2
    # and the Newton Hessian (the Newton objective clamps h the same way).
    h_raw = -np.log(np.maximum(W, 1e-300))
    h_c = np.clip(h_raw, -LOG_VAR_CLIP, LOG_VAR_CLIP)
    sigma2 = np.exp(h_c)
    ratio = r ** 2 / sigma2                                            # rho_n

    # Penalized variance-block Newton Hessian H_h = sum_n 1/2 rho_n psi psi^T +
    # diag(reg_aug) (the same object as joint_lambda._variance_hessians' H_var).
    hw = 0.5 * ratio
    H = (Psi_aug.T * hw[None, :]) @ Psi_aug + np.diag(reg_aug)

    # H_h conditioning: SPD when penalized; rcond via symmetric eigenvalues.
    evals = np.linalg.eigvalsh(H)
    lam_min, lam_max = float(evals[0]), float(evals[-1])
    h_rcond = (lam_min / lam_max) if lam_max > 0 else 0.0
    ill_conditioned = bool(lam_min <= 0.0 or h_rcond < cond_tol)

    # q_n = psi_n^T H^{-1} psi_n (one back-substitution per observation).
    HinvPsiT = np.linalg.solve(H, Psi_aug.T)                           # (F_h+1, N)
    q = np.maximum(np.einsum('nf,fn->n', Psi_aug, HinvPsiT), 0.0)

    # One-step deleted log-variance prediction and the numerical guard (check 3).
    delta_h = 0.5 * (1.0 - ratio) * q
    clipped = np.abs(delta_h) > delta_cap
    n_clipped = int(np.sum(clipped))
    delta_h = np.clip(delta_h, -delta_cap, delta_cap)

    h_del = np.clip(h_c + delta_h, -LOG_VAR_CLIP, LOG_VAR_CLIP)
    w_del = np.exp(-h_del)                                             # e^{-h_hat_{(-n)}}

    per_point_nll = 0.5 * h_del + 0.5 * r_loo ** 2 * w_del + 0.5 * _LOG2PI
    loo_nll = float(np.mean(per_point_nll))
    loo_cv = float(np.mean(w_del * r_loo ** 2))

    # KKT-style variance-floor detection (M1.3). A fitted h AT the clip does NOT
    # void Tier II by itself — the constraint may be inactive (multiplier ~ 0).
    # It is ACTIVE only if, absent the clip, the point would be pushed FURTHER
    # past the floor: at the lower clip the unconstrained per-point loss gradient
    # in h is 1/2(1 - rho_n), so rho_n < 1 (residual smaller than the floor
    # variance) means the loss still wants h even lower -> genuinely binding.
    clip_tol = floor_rtol * LOG_VAR_CLIP
    at_lo = h_raw <= (-LOG_VAR_CLIP + clip_tol)
    at_hi = h_raw >= (LOG_VAR_CLIP - clip_tol)
    floor_active = bool(np.any(at_lo & (ratio < 1.0))       # lower floor pulls down
                        or np.any(at_hi & (ratio > 1.0)))   # upper clip pulls up
    guarantee_holds = bool((not floor_active) and (not ill_conditioned))

    return {
        'loo_nll': loo_nll,
        'loo_cv': loo_cv,
        'loo_tier': 2,
        'loo_variance_floor_active': floor_active,
        'variance_hessian_ill_conditioned': ill_conditioned,
        'loo_tier2_guarantee_holds': guarantee_holds,
        'loo_nll_correction_clipped': bool(n_clipped > 0),
        'n_correction_clipped': n_clipped,
        'h_rcond': h_rcond,
        # Per-observation arrays (diagnostics / Tier-III rate validation).
        'per_point_nll': per_point_nll,
        'per_point_h_del': h_del,            # one-step deleted log-variance pred
        'per_point_delta_h': delta_h,        # the one-step correction
    }


def exact_loo_nll(
    Phi: np.ndarray,
    y: np.ndarray,
    reg_mean: np.ndarray,
    Psi: np.ndarray,
    reg_var: np.ndarray,
    *,
    leverage_correct: bool = True,
    sigma2_floor: float = 0.0,
    max_outer: int = 12,
    subset: Optional[np.ndarray] = None,
) -> Dict:
    """Tier-III exact nested leave-one-out of the joint mean+variance model.

    The oracle for the cheaper tiers (``Manuscript_Theoryv06`` App. C, and the
    authority whenever a variance floor binds): for each held-out ``n``, refit the
    joint (weighted-ridge mean + Newton log-variance) estimator on the other
    ``N-1`` points to convergence and score the held-out predictive NLL
    (including ``1/2 log 2pi``, matching :func:`joint_loo`). ``O(N)`` full joint
    refits — expensive — so this is opt-in (``result.loo(tier=3)``) and the
    validation oracle. Any Stage-C mean residual is folded into ``y`` upstream
    (``y = y_for_fourier``), so the joint sub-problem here is exactly the design
    Stage D solved.

    Args:
        Phi: (N, F) mean design (no intercept column; added internally).
        y: (N,) target the mean fit sees (residual-net contribution removed).
        reg_mean: (F,) mean ridge penalty.
        Psi: (N, F_h) variance design.
        reg_var: (F_h,) variance ridge penalty.
        leverage_correct: leverage-correct the variance step (as Stage D does).
        sigma2_floor: optional variance floor when forming mean weights.
        max_outer: alternating rounds per refit (to convergence).
        subset: restrict the deleted indices (a cheap partial oracle / testing);
            ``None`` deletes every point.

    Returns:
        dict with ``loo_nll`` (mean held-out NLL over the deleted points),
        ``per_point_nll``, ``deleted_indices``, ``loo_tier`` (=3), ``n_folds``.
    """
    from ..training.joint_lambda import _joint_fit, _augment, _nll_per_point

    Phi = np.asarray(Phi, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    Psi = np.asarray(Psi, dtype=np.float64)
    reg_mean = np.asarray(reg_mean, dtype=np.float64)
    reg_var = np.asarray(reg_var, dtype=np.float64)
    N = Phi.shape[0]

    Phi_aug = _augment(Phi)
    reg_mean_aug = np.concatenate([[0.0], reg_mean])
    idxs = np.arange(N) if subset is None else np.asarray(subset, dtype=int)

    nlls, h_dels, mus = [], [], []
    for n in idxs:
        keep = np.ones(N, dtype=bool)
        keep[n] = False
        fit = _joint_fit(
            Phi_aug[keep], Psi[keep], y[keep], reg_mean_aug, reg_var,
            leverage_correct=leverage_correct, sigma2_floor=sigma2_floor,
            max_outer=max_outer)
        mu_n = float(Phi_aug[n] @ fit.w_aug)
        h_n = float(fit.h0 + Psi[n] @ fit.w_h)
        nlls.append(float(_nll_per_point(
            np.array([y[n]]), np.array([mu_n]), np.array([h_n]))[0]))
        h_dels.append(h_n)
        mus.append(mu_n)

    nlls = np.asarray(nlls, dtype=np.float64)
    return {
        'loo_nll': float(np.mean(nlls)),
        'per_point_nll': nlls,
        'per_point_h_del': np.asarray(h_dels, dtype=np.float64),  # exact deleted log-var
        'per_point_mu': np.asarray(mus, dtype=np.float64),        # exact deleted mean
        'deleted_indices': idxs,
        'loo_tier': 3,
        'n_folds': int(idxs.shape[0]),
    }


# =============================================================================
# Sandwich estimator for coefficient covariance
# =============================================================================

def sandwich_covariance(
    Phi: np.ndarray,
    A_inv: np.ndarray,
    residuals: np.ndarray,
    hc: str = 'HC3',
    leverages: Optional[np.ndarray] = None,
    sample_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Heteroscedasticity-robust covariance of ridge coefficients.

    Var(w) ≈ A^{-1} [Phi^T diag(ω_i r_i^2) Phi] A^{-1}

    with leverage weights ω_i in the sandwich "meat":
      - HC0: ω_i = 1                       (no small-sample correction)
      - HC3: ω_i = 1 / (1 - h_ii)^2        (leverage-corrected; default)

    where h_ii = Phi_i^T A^{-1} Phi_i is the hat-matrix leverage (already used for
    LOO). HC0 is known to *under*-cover in small samples / high leverage — exactly
    the direction of the observed ~0.90-vs-0.95 Sobol-CI gap — so HC3 is the
    default here; the leverage factors are essentially free given the ridge hat
    matrix. Note this is a robust estimator of the sampling *variance* of the
    ridge coefficients; it does not capture ridge *shrinkage bias*, so downstream
    intervals are intervals for the (penalized) ridge estimand, not the
    unpenalized truth. See DEC-021.

    With ``sample_weights`` (the GLS precision weights ``W = diag(1/σ²(x_n))``
    used for a Stage-D fit) the meat becomes the *weighted* sandwich of the theory
    manuscript (``Manuscript_Theoryv06.tex`` §"One factorization, many
    diagnostics"): ``Cov(ŵ) = A^{-1}[Φ^T diag(w_n² r_n²/(1-a_n)²)Φ]A^{-1}`` with
    ``a_n = w_n φ_n^T A^{-1} φ_n`` the weighted leverage. Here ``A_inv`` and
    ``leverages`` must already be the weighted objects (from ``ridge_analytics``
    with ``weights=``). ``sample_weights=None`` is byte-for-byte the unit-weight
    estimator. This is the covariance for the *efficient* fit's own Sobol
    indices; the *interpretable* (attribution) indices are always the unit-weight
    HC3 sandwich (Theorem projection Part ii — the attribution CI is NOT
    reweighted).

    Args:
        Phi: (N, F) feature matrix
        A_inv: (F, F) inverse of (Phi^T W Phi + R)
        residuals: (N,) residuals from the ridge fit
        hc: 'HC3' (default) or 'HC0'
        leverages: (N,) hat-matrix diagonal; computed from Phi, A_inv if None
        sample_weights: (N,) GLS precision weights; ``None`` ⇒ unit weight

    Returns:
        (F, F) covariance matrix of w
    """
    Phi = np.asarray(Phi, dtype=np.float64)
    A_inv = np.asarray(A_inv, dtype=np.float64)
    r = np.asarray(residuals, dtype=np.float64)

    if hc.upper() == 'HC0':
        w_i = np.abs(r)
    elif hc.upper() == 'HC3':
        if leverages is None:
            leverages = np.sum((Phi @ A_inv) * Phi, axis=1)
        h = np.clip(np.asarray(leverages, dtype=np.float64), 0.0, 1.0 - 1e-10)
        w_i = np.abs(r) / (1.0 - h)           # ω_i r_i^2 = (|r_i|/(1-h_ii))^2
    else:
        raise ValueError(f"Unknown sandwich type '{hc}'. Options: 'HC0', 'HC3'.")

    # Precision-weighted meat: multiply each row's factor by w_n, so the squared
    # meat carries w_n² (weighted sandwich). w_n=1 recovers the unit-weight form.
    if sample_weights is not None:
        w_i = w_i * np.asarray(sample_weights, dtype=np.float64)

    # Phi^T diag(w_i^2) Phi = (Phi * w_i)^T (Phi * w_i)
    Phi_weighted = Phi * w_i[:, None]  # (N, F)
    meat = Phi_weighted.T @ Phi_weighted  # (F, F)

    # Sandwich: A^{-1} meat A^{-1}
    return A_inv @ meat @ A_inv


# =============================================================================
# Sobol confidence intervals via delta method
# =============================================================================

def _sobol_component_status(S, own, full_norm2):
    """Classify regularity from the *full* Sobol delta-gradient norm.

    For component ``u``, the numerator of the gradient is
    ``e_u ∘ U - S_u U``.  Because component blocks are disjoint, its squared
    norm is available without allocating an F-vector per component.  This
    detects both the null boundary (S=0, own block zero) and the complete-share
    boundary (S=1, every other block zero).
    """
    own_norm2 = float(own @ own)
    grad_num2 = ((1.0 - 2.0 * S) * own_norm2
                 + S * S * full_norm2)
    grad_num_norm = np.sqrt(max(0.0, grad_num2))
    scale = max(np.sqrt(max(0.0, full_norm2)), np.finfo(np.float64).tiny)
    if grad_num_norm > 1e-14 * scale:
        return 'regular'
    if S <= 1e-14:
        return 'nonregular_null'
    return 'nonregular_boundary'


def _sobol_ci_block_driven(w, Cov_w, analytics, groups, F, alpha):
    """Sobol CIs from an explicit per-group layout (mixed per-variable basis).

    ``groups`` is a list of ``(order, key, columns_slice, gram)`` covering every
    retained Sobol component; group column sizes and Grams may differ (unlike the
    uniform ``[phi1|phi2|phi3]`` path). Uses the same full delta-method gradient
    over all component blocks as the uniform routine — with
    ``g_i = (e_i∘U - S_i U)/V_tot`` and precomputed ``Cov_U``/``UCU`` — so the two
    paths agree exactly on a uniform layout.
    """
    from scipy.stats import t as sp_t

    comps = []  # (order, key, slice, G, var)
    for (order, key, sl, G) in groups:
        G = np.asarray(G, dtype=np.float64)
        wi = w[sl]
        comps.append((order, key, sl, G, max(0.0, float(wi @ G @ wi))))
    total_var = sum(c[4] for c in comps)

    base = {
        'sigma_hat': analytics['sigma_hat'],
        'sigma2_hat': analytics['sigma2_hat'],
        'df': analytics['df'],
        'df_residual': analytics['df_residual'],
        'loo_cv': analytics['loo_cv'],
        'gcv': analytics['gcv'],
        'aic': analytics['aic'],
        'bic': analytics['bic'],
        'total_model_variance': total_var,
        'alpha': alpha,
        'sandwich': 'HC3',
        'crit_dist': 't',
        'conditional_on_residual_variance': True,
        # Delta intervals are fixed-configuration HC3/t intervals.  A zero
        # component block has a degenerate first derivative, so its ordinary
        # delta interval is nonregular; bootstrap/quadratic-form inference for
        # that boundary case is deliberately deferred (X6 Session 3).
        'interval_method': 'HC3_delta_t',
    }

    if total_var < 1e-15:
        base.update(first_order={c[1]: (0.0, 0.0, 0.0)
                                 for c in comps if c[0] == 1},
                    second_order={}, third_order={}, total_model_variance=0.0,
                    component_status={
                        'first_order': {c[1]: 'nonregular_null'
                                        for c in comps if c[0] == 1},
                        'second_order': {c[1]: 'nonregular_null'
                                         for c in comps if c[0] == 2},
                        'third_order': {c[1]: 'nonregular_null'
                                        for c in comps if c[0] == 3},
                    })
        return base

    t_df = max(analytics['df_residual'], 1.0)
    z_crit = float(sp_t.ppf(1.0 - alpha / 2, df=t_df))

    # U holds 2 G_j w_j per block (zero elsewhere); Cov_U / UCU are shared across
    # components via the denominator-coupling term of the delta method.
    U = np.zeros(F, dtype=np.float64)
    for (order, key, sl, G, var) in comps:
        U[sl] = 2.0 * (G @ w[sl])
    Cov_U = Cov_w @ U
    UCU = float(U @ Cov_U)
    U_norm2 = float(U @ U)

    first_order_ci, second_order_ci, third_order_ci = {}, {}, {}
    component_status = {
        'first_order': {}, 'second_order': {}, 'third_order': {}}
    for (order, key, sl, G, var) in comps:
        S = var / total_var
        own = U[sl]
        var_num = (float(own @ Cov_w[sl, sl] @ own)
                   - 2.0 * S * float(own @ Cov_U[sl])
                   + S * S * UCU)
        se_S = np.sqrt(max(0.0, var_num / (total_var * total_var)))
        lo = max(0.0, S - z_crit * se_S)
        hi = min(1.0, S + z_crit * se_S)
        status = _sobol_component_status(S, own, U_norm2)
        if order == 1:
            first_order_ci[key] = (S, lo, hi)
            component_status['first_order'][key] = status
        elif order == 2:
            second_order_ci[key] = (S, lo, hi)
            component_status['second_order'][key] = status
        else:
            third_order_ci[key] = (S, lo, hi)
            component_status['third_order'][key] = status

    base.update(first_order=first_order_ci, second_order=second_order_ci,
                third_order=third_order_ci, component_status=component_status,
                t_df=t_df)
    return base


def sobol_confidence_intervals(
    Phi: np.ndarray,
    y: np.ndarray,
    reg_diag: np.ndarray,
    D: int,
    K1: int = 0,
    G1: Optional[np.ndarray] = None,
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
    weights: Optional[np.ndarray] = None,
    groups: Optional[list] = None,
    fidelity: Optional[float] = None,
    profile_intercept: bool = False,
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
        fidelity: structural fidelity 𝔉 = V_core/(V_core+Var(ĝ)) (M3/DEC-032). When
            given, a ``total`` block of shares Ŝ^total = 𝔉·Ŝ^core is returned beside
            the core indices — the core CI scaled by 𝔉, treating Var(ĝ) as fixed
            (conditional on the residual variance, the same convention the core CI
            already carries). ``None`` (default) ⇒ no total block, byte-identical to
            the pre-DEC-032 output. The first/second/third_order indices are always
            the CORE shares S^core = V_u/V_core (residual excluded from the denom).
        profile_intercept: fit the Stage-D profiled joint-GLS intercept on the
            augmented design ``[1, Phi]`` and take the CI covariance from the
            augmented sandwich's slope-slope block (Remark rem:intercept). Pass the
            *uncentered* response. Used for the efficient (weighted) Stage-D index
            set; ``False`` (default) is the legacy feature-only fit.

    Returns:
        Dict with:
          first_order: {i: (S_i, S_lo, S_hi)}          — CORE shares
          second_order: {(i,j): (S_ij, S_lo, S_hi)}
          third_order: {(i,j,k): (S_ijk, S_lo, S_hi)}
          component_status: per-order regular / nonregular_null /
              nonregular_boundary labels from the full delta gradient
          total: {'first_order': {...}, ...}           — 𝔉·core, only if fidelity given
          sigma_hat: noise estimate
          df: effective degrees of freedom
    """
    from scipy.stats import t as sp_t

    Phi = np.asarray(Phi, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    reg_diag = np.asarray(reg_diag, dtype=np.float64)
    if G1 is not None:  # None only on the block-driven (mixed-basis) path below
        G1 = np.asarray(G1, dtype=np.float64)

    # Get analytics. ``weights`` (Stage-D GLS precision W=diag(1/σ²)) selects the
    # weighted fit — used only for the *efficient* index set (M2); the default
    # interpretable/attribution call passes weights=None and is byte-identical.
    # ``profile_intercept`` fits the Stage-D profiled joint-GLS intercept on the
    # augmented design (Remark rem:intercept); ``w`` is then the nonconstant slopes
    # (intercept excluded from the Sobol slices) and the covariance is the
    # augmented sandwich's slope-slope block.
    analytics = ridge_analytics(Phi, y, reg_diag, weights=weights,
                                profile_intercept=profile_intercept)
    w = analytics['w']
    A_inv = analytics['A_inv']
    residuals = analytics['residuals']
    N, F = Phi.shape

    # HC3 (leverage-corrected) sandwich covariance — reuse the hat-matrix
    # diagonal already formed in ridge_analytics. Under weights this is the
    # weighted sandwich A_w^{-1}[Φ^T diag(w_n²r_n²/(1-a_n)²)Φ]A_w^{-1}.
    if profile_intercept:
        # Coherent augmented sandwich: build the full (F+1)×(F+1) covariance on
        # Z=[1,Φ] with the augmented bread (A_aug^{-1}) and augmented HC3 meat
        # (augmented leverages), then extract the slope-slope block. The
        # intercept/feature cross terms matter, so slicing A_aug^{-1} into the
        # legacy feature-only meat would NOT be equivalent (plan §9.3).
        Cov_theta = sandwich_covariance(
            analytics['Z'], A_inv, residuals, hc='HC3',
            leverages=analytics['leverages'], sample_weights=weights)
        Cov_w = Cov_theta[1:, 1:]
    else:
        Cov_w = sandwich_covariance(Phi, A_inv, residuals, hc='HC3',
                                    leverages=analytics['leverages'],
                                    sample_weights=weights)

    # Mixed per-variable basis: groups within an order have different column
    # sizes/Grams, so the uniform reconstruction below does not apply. Use the
    # explicit per-group layout instead (same delta-method machinery).
    if groups is not None:
        return _sobol_ci_block_driven(w, Cov_w, analytics, groups, F, alpha)

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
            i, j, k = (int(triple_indices[t, pos]) for pos in range(3))
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
            'component_status': {
                'first_order': {i: 'nonregular_null' for i in range(D)},
                'second_order': {key: 'nonregular_null'
                                 for key in second_order_vars},
                'third_order': {key: 'nonregular_null'
                                for key in third_order_vars},
            },
            'sigma_hat': analytics['sigma_hat'],
            'df': analytics['df'],
            'total_model_variance': 0.0,
            'interval_method': 'HC3_delta_t',
        }

    # Small-sample critical value: t at the residual effective df
    # (N - 2 tr(H) + tr(H^2)), not the normal z. The two agree as df -> inf but
    # t widens the interval in the finite-N / high-complexity regime where HC0
    # under-covered. df_residual is floored at 1 in ridge_analytics.
    t_df = max(analytics['df_residual'], 1.0)
    z_crit = float(sp_t.ppf(1.0 - alpha / 2, df=t_df))

    # ---- Full delta-method gradient over ALL component blocks ----
    # S_i = V_i / V_tot with V_tot = sum_j V_j, so S_i depends on every
    # component's coefficients through the denominator:
    #   dS_i/dw_i = 2 G w_i (1 - S_i) / V_tot          (own block)
    #   dS_i/dw_j = -S_i * 2 G_j w_j / V_tot   (j != i) (denominator coupling)
    # Dropping the j != i terms (own-block-only delta method) underestimates
    # SE(S) by a factor that does NOT vanish with N and grows with S_i —
    # measured ~13-15% SE deficit => ~90% actual coverage at 95% nominal.
    # We therefore use the full gradient with the full covariance matrix.
    #
    # U holds 2 G_j w_j stacked per component block (zero on any extra
    # columns such as residual features, which do not enter S).
    U = np.zeros(F, dtype=np.float64)
    for i in range(D):
        sl = slice(i * block1, (i + 1) * block1)
        U[sl] = 2.0 * (G1 @ w[sl])
    if K2 > 0 and P > 0 and G2 is not None:
        G2_np_u = np.asarray(G2, dtype=np.float64)
        b2 = G2_np_u.shape[0]
        for p in range(P):
            sl = slice(F1 + p * b2, F1 + (p + 1) * b2)
            U[sl] = 2.0 * (G2_np_u @ w[sl])
    if K3 > 0 and T > 0 and G3 is not None:
        G3_np_u = np.asarray(G3, dtype=np.float64)
        b3 = G3_np_u.shape[0]
        for t in range(T):
            sl = slice(F1 + F2 + t * b3, F1 + F2 + (t + 1) * b3)
            U[sl] = 2.0 * (G3_np_u @ w[sl])
    Cov_U = Cov_w @ U          # (F,)
    UCU = float(U @ Cov_U)     # scalar: U^T Cov U
    U_norm2 = float(U @ U)     # shared full-gradient norm term

    def _sobol_ci(var_component, var_total, w_slice, G_block):
        """Compute S, SE(S), and CI for one component via the full delta method.

        With g_i = (e_i ∘ U - S_i U) / V_tot (e_i = indicator of the own
        block), Var(S_i) = g_i^T Cov g_i expands to
          [U_i^T Cov_ii U_i - 2 S_i U_i^T (Cov U)_i + S_i^2 U^T Cov U] / V_tot^2
        which needs only the own-block slice of Cov and the precomputed
        Cov_U / UCU — no per-component F x F work.
        """
        S = var_component / var_total if var_total > 0 else 0.0

        own = U[w_slice]  # = 2 G_block w[w_slice]
        var_num = (float(own @ Cov_w[w_slice, w_slice] @ own)
                   - 2.0 * S * float(own @ Cov_U[w_slice])
                   + S * S * UCU)
        var_S = var_num / (var_total * var_total)
        se_S = np.sqrt(max(0.0, var_S))

        lo = max(0.0, S - z_crit * se_S)
        hi = min(1.0, S + z_crit * se_S)
        return (S, lo, hi, se_S)

    component_status = {
        'first_order': {}, 'second_order': {}, 'third_order': {}}
    # First-order CIs
    first_order_ci = {}
    for i in range(D):
        sl = slice(i * block1, (i + 1) * block1)
        S, lo, hi, se = _sobol_ci(first_order_vars[i], total_var, sl, G1)
        first_order_ci[i] = (S, lo, hi)
        component_status['first_order'][i] = _sobol_component_status(
            S, U[sl], U_norm2)

    # Second-order CIs
    second_order_ci = {}
    if K2 > 0 and G2 is not None:
        G2_np = np.asarray(G2, dtype=np.float64)
        block2 = G2_np.shape[0]
        for p, ((i, j), var_p) in enumerate(second_order_vars.items()):
            sl = slice(F1 + p * block2, F1 + (p + 1) * block2)
            S, lo, hi, se = _sobol_ci(var_p, total_var, sl, G2_np)
            second_order_ci[(i, j)] = (S, lo, hi)
            component_status['second_order'][(i, j)] = (
                _sobol_component_status(S, U[sl], U_norm2))

    # Third-order CIs
    third_order_ci = {}
    if K3 > 0 and G3 is not None:
        G3_np = np.asarray(G3, dtype=np.float64)
        block3 = G3_np.shape[0]
        for t, ((i, j, k), var_t) in enumerate(third_order_vars.items()):
            sl = slice(F1 + F2 + t * block3, F1 + F2 + (t + 1) * block3)
            S, lo, hi, se = _sobol_ci(var_t, total_var, sl, G3_np)
            third_order_ci[(i, j, k)] = (S, lo, hi)
            component_status['third_order'][(i, j, k)] = (
                _sobol_component_status(S, U[sl], U_norm2))

    # Total-variance shares Ŝ^total = 𝔉·Ŝ^core (M3/DEC-032). The core CIs above use
    # V_core (structured orders only) as the denominator; scaling by the fixed
    # fidelity 𝔉 gives the total-variance shares, with the interval conditional on the
    # (QMC-estimated) residual variance — consistent with the core CI's own
    # conditional_on_residual_variance convention. Omitted when fidelity is None.
    total_block = None
    if fidelity is not None:
        _F = float(fidelity)

        def _scale(d):
            return {k: (_F * S, _F * lo, _F * hi) for k, (S, lo, hi) in d.items()}

        total_block = {
            'first_order': _scale(first_order_ci),
            'second_order': _scale(second_order_ci),
            'third_order': _scale(third_order_ci),
            'fidelity': _F,
            'conditional_on_residual_variance': True,
        }

    return {
        'first_order': first_order_ci,
        'second_order': second_order_ci,
        'third_order': third_order_ci,
        'component_status': component_status,
        **({'total': total_block} if total_block is not None else {}),
        'sigma_hat': analytics['sigma_hat'],
        'sigma2_hat': analytics['sigma2_hat'],
        'df': analytics['df'],
        'df_residual': analytics['df_residual'],
        'loo_cv': analytics['loo_cv'],
        'gcv': analytics['gcv'],
        'aic': analytics['aic'],
        'bic': analytics['bic'],
        'total_model_variance': total_var,
        'alpha': alpha,
        # CI construction provenance (advisor items #4/#6, DEC-021):
        'sandwich': 'HC3',        # leverage-corrected robust meat
        'crit_dist': 't',         # small-sample critical value ...
        'interval_method': 'HC3_delta_t',
        't_df': t_df,             # ... at the residual effective df
        # The delta-method gradient U is zero on any residual/NN columns, so a CI
        # on S_i when a residual is present treats the residual denominator term
        # as fixed: the interval is CONDITIONAL on the (QMC-estimated) residual
        # variance. With the QMC residual its MC error is negligible, so this is
        # approximately exact rather than silently false.
        'conditional_on_residual_variance': True,
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

    w_{-k} = w + A^{-1} Phi_k^T [I - Phi_k A^{-1} Phi_k^T]^{-1} (Phi_k w - y_k)

    Removing fold k is a rank-(N/k) *downdate* of A = Phi^T Phi + R, so the inner
    matrix is [I - H_k] with H_k = Phi_k A^{-1} Phi_k^T (the fold's leverage
    block); at n_k=1 this reduces to the standard LOO residual (y-yhat)/(1-h_i).

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
    A_inv = spd_inverse(A)
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

        # Woodbury downdate: w_{-k} = w + A^{-1} Phi_k^T [I - H_k]^{-1} (Phi_k w - y_k)
        w_minus_k, _, _ = _woodbury_downdate(A_inv, Phi_k, w_full, y_k)

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
    A_inv = spd_inverse(A)
    w_full = A_inv @ (Phi.T @ y)
    r_full = y - Phi @ w_full
    M_full = A_inv @ (Phi.T @ Phi)
    df_full = float(np.trace(M_full))

    # Full-data analytics for comparison. σ̂ uses the residual effective df
    # (N - 2 tr(H) + tr(H^2)), the single convention shared with ridge_analytics
    # / the Sobol CIs (DEC-021); N - df is the shorthand it refines.
    df_resid_full = max(N - 2.0 * df_full + float(np.sum(M_full * M_full.T)), 1.0)
    sigma2_full = float(np.sum(r_full**2) / df_resid_full)
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

        # Woodbury downdate for w_{-k} (I - H_k; see kfold_cv_analytic). Reuse the
        # A_inv_PhikT / inner factors below for the exact leave-fold-out df.
        w_k, A_inv_PhikT, inner = _woodbury_downdate(A_inv, Phi_k, w_full, y_k)

        # Validation RMSE
        pred_k = Phi_k @ w_k
        rmse_k = float(np.sqrt(np.mean((y_k - pred_k) ** 2)))

        # Training residuals for sigma estimate
        Phi_train = Phi[train_mask]
        y_train = y[train_mask]
        r_train = y_train - Phi_train @ w_k
        # Exact effective df of the leave-fold-out ridge fit:
        #   df_k = tr(H_k) = F - tr(A_k^{-1} R),  R = diag(reg_diag),
        # where A_k^{-1} is the Woodbury downdate of A_full (fold k removed),
        # reusing the A_inv / A_inv_PhikT / inner factors already formed above.
        # (Replaces the earlier df_full * n_train/N linear approximation, which
        # overstated df_k because ridge df is concave in the sample size.)
        Sk = np.linalg.solve(inner, A_inv_PhikT.T)          # (n_k, F)
        Ak_inv_diag = np.diag(A_inv) + np.einsum('fi,if->f', A_inv_PhikT, Sk)
        df_k = float(F - np.sum(reg_diag * Ak_inv_diag))
        # Per-fold σ̂ keeps the (n_train - df_k) shorthand: this is a per-fold
        # stability *diagnostic* (item #5 fixed df_k itself to the exact tr(H_k));
        # the full residual-df form would need tr(H_k H_k^T) per fold and does not
        # move the diagnostic materially. The reported full-data σ̂ above uses the
        # exact residual df.
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

    # Per-variable precision from CIs.
    # NOTE: the two quantities below are first-order RULES OF THUMB, not precise
    # outputs. `se_rule_of_thumb` inverts the CI width through a *normal* 1.96
    # factor (the CIs themselves now use a t critical value), and `N_for_half_ci`
    # is the textbook SE ~ 1/sqrt(N) asymptotic (4x the data halves the SE) that
    # ignores the ridge penalty and finite-sample df. Use for rough sizing only.
    var_precision = {}
    for i, (S, lo, hi) in sobol_ci['first_order'].items():
        width = hi - lo
        var_precision[i] = {
            'sobol': S,
            'ci_width': width,
            'se_rule_of_thumb': width / (2 * 1.96),  # ~SE via normal approx (rough)
        }
        # Rule of thumb: SE ~ 1/sqrt(N) => 4x the data halves the CI.
        if width > 0:
            var_precision[i]['N_for_half_ci_rule_of_thumb'] = int(4 * N)

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
