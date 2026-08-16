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
from typing import Dict, Tuple
from scipy.optimize import minimize_scalar, minimize

# Proper-prior floor for penalty entries in the evidence log-determinants.
# Exact zeros (smoothness/curvature/spectral leave k=0 unpenalized) make R
# singular, so the Sylvester identity log|K| = log|A| - log|R| is invalid for
# the raw penalty. Flooring reg at _REG_FLOOR inside log|R| evaluates the
# evidence of the proper model with prior precision max(reg, _REG_FLOOR) —
# the same convention as the dual form's R^{-1} cap at 1/_REG_FLOOR — up to
# an O(_REG_FLOOR) perturbation of log|A|. The floored entries are constant
# in lambda, so lambda selection is unaffected; the evidence VALUE becomes
# consistent across the primal, dual, multi-lambda, and JAX paths.
_REG_FLOOR = 1e-12

# Degrees of freedom consumed by the profiled (centered-out) intercept. Every
# criterion path in this module receives y CENTERED (plain or weighted mean
# subtracted before the solve), which spends ~1 df that tr(H) of the centered
# design does not count — omitting it under-penalizes GCV/AIC/BIC by one
# parameter. The criteria therefore use df + INTERCEPT_DF; the reported 'df'
# stays tr(H) (complexity of the penalized block only). The +1 is constant in
# lambda, so criterion GRADIENTS are unchanged in form; for AIC/BIC it shifts
# the value by a lambda-independent constant (selection unchanged), for GCV it
# slightly shifts the selected lambda (the corrected criterion).
INTERCEPT_DF = 1.0


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
    # Zero penalty entries are floored (see _REG_FLOOR): dropping them from
    # log|R| while log|A| keeps their Phi^T Phi contribution would make the
    # identity, and hence the evidence, invalid.
    A = Phi.T @ Phi + np.diag(reg_diag)
    sign, logdet_A = np.linalg.slogdet(A)
    logdet_R = float(np.sum(np.log(np.maximum(reg_diag, _REG_FLOOR))))
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

    y is expected CENTERED (package convention): the GCV/AIC/BIC criteria add
    INTERCEPT_DF for the profiled mean; the returned 'df' stays tr(H).
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

    # GCV / AIC / BIC — model df includes the profiled intercept (INTERCEPT_DF)
    df_sel = df + INTERCEPT_DF
    gcv_denom = max(1e-10, 1.0 - df_sel / N)
    gcv_val = (rss_val / N) / gcv_denom ** 2
    aic_val = N * np.log(max(mse_val, 1e-15)) + 2.0 * df_sel
    bic_val = N * np.log(max(mse_val, 1e-15)) + np.log(N) * df_sel

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


class RidgePathEigSolver:
    """Evaluate ridge diagnostics for ``reg = lambda * reg_shape`` at many
    ``lambda`` from a *single* eigendecomposition.

    A regularization path (and single-scalar GCV search) solves
    ``(Phi^T Phi + lambda * diag(reg_shape)) w = Phi^T y`` for a grid of
    ``lambda`` with ``reg_shape`` **fixed**. Re-solving per point costs
    ``O(F^3)`` each; this class pays one ``O(F^3)`` eigendecomposition up front
    and then answers every ``lambda`` in ``O(F^2)`` (``O(F)`` for the scalar
    diagnostics), reproducing :func:`ridge_solve_with_diagnostics` to
    floating-point round-off.

    Math. With ``S = diag(sqrt(reg_shape))`` and ``B = Phi S^{-1}``, let
    ``M = B^T B = Q diag(mu) Q^T`` (symmetric eigendecomposition). Because
    ``A(lambda) = Phi^T Phi + lambda*diag(reg_shape) = S (M + lambda I) S``,
    every diagnostic reduces to a function of the eigenvalues ``mu`` and the
    projected target ``c = Q^T S^{-1} Phi^T y``:

    - ``w        = S^{-1} Q (c / (mu + lambda))``   (coefficients, O(F^2))
    - ``df       = sum(mu / (mu + lambda))``        (effective dof, O(F))
    - ``log|K|   = sum(log((mu+lambda)/lambda))``   (matches primal & dual)

    ``RSS`` and the profile ``sigma^2`` are formed from the fitted values
    ``Phi @ w`` **directly** (O(N*F)), not from an eigen-sum: near the
    interpolation limit (``F >= N``, small ``lambda``) the eigen-sum form
    ``||y||^2 - 2*sum(c^2/(mu+lambda)) + ...`` loses all precision to
    cancellation, whereas ``||y - Phi w||^2`` stays accurate — and O(N*F) is
    still far below the ``O(F^3)`` of a per-point solve, so the speedup holds.

    The ``log|K|`` identity makes the profile evidence agree with the code's
    dual-form evidence for ``F > N`` too (the ``F - N`` near-zero eigenvalues
    contribute ``log(lambda/lambda) = 0``).

    Requires ``reg_shape > 0`` strictly (needed to whiten by ``S^{-1}``). Penalty
    shapes with exact zeros — e.g. the ``smoothness``/``curvature``/``spectral``
    strategies leave the k=0 term unpenalized — are not supported; callers should
    fall back to per-point solves for those.
    """

    def __init__(self, Phi: np.ndarray, y: np.ndarray, reg_shape: np.ndarray):
        Phi = np.asarray(Phi, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        reg_shape = np.asarray(reg_shape, dtype=np.float64)
        if not np.all(reg_shape > 0.0):
            raise ValueError(
                "RidgePathEigSolver requires a strictly positive reg_shape "
                "(min={:.3e}); use per-point solves for penalty shapes with "
                "zeros.".format(float(reg_shape.min()))
            )
        self.N, self.F = Phi.shape
        self._Phi = Phi
        self._y = y
        self._sinv = 1.0 / np.sqrt(reg_shape)
        B = Phi * self._sinv[None, :]              # Phi S^{-1}   (N, F)
        M = B.T @ B                                # F x F, symmetric PSD
        mu, Q = np.linalg.eigh(M)
        self._mu = np.clip(mu, 0.0, None)          # kill tiny negative round-off
        self._Q = Q
        self._c = Q.T @ (B.T @ y)                  # Q^T S^{-1} Phi^T y   (F,)
        self._log2pi = float(np.log(2.0 * np.pi))

    def diagnostics(self, lam: float) -> Dict:
        """Return the same dict as ``ridge_solve_with_diagnostics(Phi, y,
        lam*reg_shape)`` for this solver's ``(Phi, y, reg_shape)``."""
        lam = float(lam)
        N, _F = self.N, self.F
        mu, c = self._mu, self._c
        denom = mu + lam
        a = c / denom
        w = self._sinv * (self._Q @ a)             # coefficients

        # RSS and profile sigma^2 from the fitted values directly (stable near
        # the interpolation limit; matches ridge_solve_with_diagnostics).
        resid = self._y - self._Phi @ w
        rss = float(resid @ resid)
        mse = rss / N

        df = float(np.sum(mu / denom))
        df_sel = df + INTERCEPT_DF          # profiled-intercept df in criteria
        gcv_denom = max(1e-10, 1.0 - df_sel / N)
        gcv = (rss / N) / gcv_denom ** 2
        aic = N * np.log(max(mse, 1e-15)) + 2.0 * df_sel
        bic = N * np.log(max(mse, 1e-15)) + np.log(N) * df_sel

        sigma2_profile = max(1e-15, float(self._y @ resid) / N)  # y^T r / N
        logdet_K = float(np.sum(np.log(denom) - np.log(lam)))
        log_evidence = (-N / 2.0 * np.log(sigma2_profile)
                        - 0.5 * logdet_K
                        - N / 2.0 * (1.0 + self._log2pi))

        return {
            'w': w,
            'rss': rss,
            'mse': mse,
            'df': df,
            'gcv': gcv,
            'aic': aic,
            'bic': bic,
            'log_evidence': float(log_evidence),
            'sigma2_ml': max(mse, 1e-15),
        }

    def criterion_and_grad(self, lam: float, method: str = 'gcv'):
        """Closed-form model-selection criterion and its exact derivative in
        ``lambda``, both from the eigendecomposition (no finite differences).

        Returns ``(value, d value / d lambda)`` where ``value`` is a *minimization*
        objective: the criterion itself for ``'gcv'``/``'aic'``/``'bic'`` and
        ``-log_evidence`` for ``'evidence'``. The derivatives follow from
        ``df(lam) = sum(mu/(mu+lam))``, ``RSS'(lam) = 2*lam*sum(c^2/(mu+lam)^3)``,
        and ``log|K|(lam) = sum(log((mu+lam)/lam))``.
        """
        lam = float(lam)
        N, F = self.N, self.F
        mu, c = self._mu, self._c
        denom = mu + lam
        c2 = c * c

        # RSS from the fitted values (stable); RSS' in closed form.
        a = c / denom
        w = self._sinv * (self._Q @ a)
        resid = self._y - self._Phi @ w
        rss = max(float(resid @ resid), 0.0)
        rss_p = 2.0 * lam * float(np.sum(c2 / denom ** 3))

        df = float(np.sum(mu / denom))
        df_p = -float(np.sum(mu / denom ** 2))

        m = method.lower()
        if m == 'gcv':
            u = rss / N
            v = 1.0 - (df + INTERCEPT_DF) / N   # gcv denominator base
            v = v if v > 1e-10 else 1e-10   # match ridge_solve_with_diagnostics guard
            up, vp = rss_p / N, -df_p / N   # INTERCEPT_DF is constant in lambda
            value = u / v ** 2
            grad = up / v ** 2 - 2.0 * u * vp / v ** 3
            return value, grad
        if m in ('aic', 'bic'):
            mse = max(rss / N, 1e-15)
            pen = 2.0 if m == 'aic' else float(np.log(N))
            value = N * np.log(mse) + pen * (df + INTERCEPT_DF)
            grad = N * rss_p / max(rss, 1e-15) + pen * df_p
            return value, grad
        if m == 'evidence':
            P = float(self._y @ resid)                 # y^T r  (= N * profile sigma^2)
            sigma2 = max(P / N, 1e-15)
            logdet_K = float(np.sum(np.log(denom) - np.log(lam)))
            log_ev = (-N / 2.0 * np.log(sigma2) - 0.5 * logdet_K
                      - N / 2.0 * (1.0 + self._log2pi))
            P_p = float(np.sum(c2 / denom ** 2))        # d(y^T r)/dlam
            dlogdet_K = float(np.sum(1.0 / denom)) - F / lam
            dlog_ev = -N / 2.0 * (P_p / max(P, 1e-15)) - 0.5 * dlogdet_K
            return -log_ev, -dlog_ev                    # minimize -log_evidence
        raise ValueError(
            f"method must be 'gcv', 'aic', 'bic', or 'evidence'; got {method!r}")


def _optimize_single_lambda_analytic(Phi, y, reg_structure, method, bounds, n_grid):
    """Analytic-gradient scalar-lambda optimization via one eigendecomposition.

    Brackets the optimum on a coarse log-lambda grid (criterion value only), then
    refines with L-BFGS-B using the closed-form criterion gradient (chain-ruled to
    log10 lambda). Returns the same diagnostics dict as the numeric path.
    """
    solver = RidgePathEigSolver(Phi, y, reg_structure)
    log_lo, log_hi = np.log10(bounds[0]), np.log10(bounds[1])
    ts = np.linspace(log_lo, log_hi, max(8, n_grid // 4))
    vals = [solver.criterion_and_grad(10.0 ** t, method)[0] for t in ts]
    t0 = float(ts[int(np.argmin(vals))])
    ln10 = np.log(10.0)

    def f_and_grad(t):
        lam = 10.0 ** float(t[0])
        v, dvdlam = solver.criterion_and_grad(lam, method)
        return v, np.array([dvdlam * lam * ln10])   # chain rule d/d(log10 lam)

    res = minimize(f_and_grad, x0=np.array([t0]), jac=True,
                   method='L-BFGS-B', bounds=[(log_lo, log_hi)])
    lam_opt = float(10.0 ** res.x[0])
    diag = solver.diagnostics(lam_opt)
    diag['lambda_opt'] = lam_opt
    diag['converged'] = bool(res.success)
    return diag


def _optimize_single_lambda_jax(Phi, y, reg_structure, method, bounds, n_grid):
    """JAX/AD-gradient scalar-lambda optimization.

    Same structure as :func:`_optimize_single_lambda_analytic` (coarse log-lambda
    bracket on criterion values, then one L-BFGS-B refine with an exact jacobian),
    but the gradient is supplied by :func:`jax.grad` via
    :mod:`hifi_anova.training.hyperopt_jax` rather than the closed-form eigen
    formula. Unlike the analytic path this does not require ``reg_structure > 0``
    (the evidence log-det masks the unpenalized support). Returns the same
    diagnostics dict as the numeric/analytic paths.
    """
    from .hyperopt_jax import criterion_valgrad_jax, optimize_lambdas_jax

    log_lo, log_hi = np.log10(bounds[0]), np.log10(bounds[1])
    ts = np.linspace(log_lo, log_hi, max(8, n_grid // 4))
    vals = [criterion_valgrad_jax([t], Phi, y, [reg_structure], method)[0]
            for t in ts]
    t0 = float(ts[int(np.argmin(vals))])
    out = optimize_lambdas_jax(Phi, y, [reg_structure], method, bounds,
                               x0_log=[t0], names=['_lam'])
    out['lambda_opt'] = out.pop('_lam')
    return out


def optimize_single_lambda(Phi: np.ndarray, y: np.ndarray,
                           reg_structure: np.ndarray,
                           method: str = 'gcv',
                           bounds: Tuple[float, float] = (1e-6, 1e2),
                           n_grid: int = 50,
                           grad: str = 'numeric') -> Dict:
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
        grad: gradient mode for the refinement step.
            'numeric' (default): derivative-free `minimize_scalar` (unchanged).
            'analytic': use closed-form criterion gradients (:class:`RidgePathEigSolver`)
                via a single eigendecomposition — one L-BFGS-B refine with exact
                jacobian instead of many finite-difference criterion evaluations.
            'jax': same objective, gradient by autodiff (:func:`jax.grad`,
                :mod:`hifi_anova.training.hyperopt_jax`). Matches 'analytic' to
                ~1e-10; does not require a strictly positive shape.
            'auto': 'analytic' when the penalty shape is strictly positive and
                well conditioned, else 'numeric'.

    Returns:
        Dict with lambda_opt and diagnostics at the optimum.
    """
    Phi = np.asarray(Phi, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    reg_structure = np.asarray(reg_structure, dtype=np.float64)
    N, F = Phi.shape

    if grad not in ('numeric', 'analytic', 'auto', 'jax'):
        raise ValueError(
            f"grad must be 'numeric'/'analytic'/'auto'/'jax'; got {grad!r}")
    if grad == 'jax':
        return _optimize_single_lambda_jax(
            Phi, y, reg_structure, method, bounds, n_grid)
    _rs_min = float(reg_structure.min()) if reg_structure.size else 0.0
    _well_cond = _rs_min > 0.0 and float(reg_structure.max() / _rs_min) < 1e8
    _use_analytic = (grad == 'analytic' or (grad == 'auto' and _well_cond))
    if grad == 'analytic' and _rs_min <= 0.0:
        raise ValueError(
            "grad='analytic' needs a strictly positive reg_structure "
            f"(min={_rs_min:.3e}); use grad='numeric' or 'auto'.")
    if _use_analytic:
        return _optimize_single_lambda_analytic(
            Phi, y, reg_structure, method, bounds, n_grid)

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


def _criterion_valgrad_multi(Phi, C, b, y, shapes, sizes, loglams, method):
    """Model-selection criterion value + gradient w.r.t. log10(lambda) for
    independent per-block lambdas, from ONE factorization of A.

    reg = sum_k (10**loglams[k]) * shapes[k]; A = C + diag(reg). Using
    U = A^{-1}: df = tr(UC); d(df)/dlam_k = -diag(UCU) . shape_k;
    RSS'(lam_k) = 2 * shape_k . (w * (U (reg*w))); d(y^T r)/dlam_k = shape_k . w^2;
    d(log|A|)/dlam_k = diag(U) . shape_k; d(log|R|)/dlam_k = sizes[k]/lam_k.
    Returns (value, grad) with `value` a minimization objective (-log_evidence for
    'evidence') and `grad` the derivative w.r.t. each log10(lambda_k).
    """
    N, F = Phi.shape
    K = len(shapes)
    lambdas = np.array([10.0 ** float(t) for t in loglams])
    reg = np.zeros(F)
    for k in range(K):
        reg = reg + lambdas[k] * shapes[k]
    A = C + np.diag(reg)
    U = np.linalg.inv(A)
    w = U @ b
    resid = y - Phi @ w
    rss = max(float(resid @ resid), 0.0)
    df = float(np.sum(U * C))                       # tr(U C)
    UC = U @ C
    diagB = np.einsum('jq,jq->j', UC, U)            # diag(U C U)
    t_vec = U @ (reg * w)
    wt = w * t_vec
    w2 = w * w
    diagU = np.diag(U)

    rss_p = np.array([2.0 * float(shapes[k] @ wt) for k in range(K)])
    df_p = np.array([-float(diagB @ shapes[k]) for k in range(K)])
    ln10 = np.log(10.0)

    m = method.lower()
    if m == 'gcv':
        u = rss / N
        v = max(1e-10, 1.0 - (df + INTERCEPT_DF) / N)
        val = u / v ** 2
        up, vp = rss_p / N, -df_p / N       # INTERCEPT_DF is constant in lambda
        glin = up / v ** 2 - 2.0 * u * vp / v ** 3
    elif m in ('aic', 'bic'):
        mse = max(rss / N, 1e-15)
        pen = 2.0 if m == 'aic' else float(np.log(N))
        val = N * np.log(mse) + pen * (df + INTERCEPT_DF)
        glin = N * rss_p / max(rss, 1e-15) + pen * df_p
    elif m == 'evidence':
        P = float(y @ resid)
        sigma2 = max(P / N, 1e-15)
        sign, logdetA = np.linalg.slogdet(A)
        # Floored log|R| (see _REG_FLOOR); floored entries are constant in
        # lambda, so only entries above the floor contribute to the gradient.
        logdetR = float(np.sum(np.log(np.maximum(reg, _REG_FLOOR))))
        logdetK = logdetA - logdetR
        val = -(-N / 2.0 * np.log(sigma2) - 0.5 * logdetK
                - N / 2.0 * (1.0 + float(np.log(2 * np.pi))))
        P_p = np.array([float(shapes[k] @ w2) for k in range(K)])
        logA_p = np.array([float(diagU @ shapes[k]) for k in range(K)])
        logR_p = np.array([
            float(np.sum((shapes[k] > 0) & (reg > _REG_FLOOR))) / lambdas[k]
            for k in range(K)])
        dlog_ev = -N / 2.0 * (P_p / max(P, 1e-15)) - 0.5 * (logA_p - logR_p)
        glin = -dlog_ev
    else:
        raise ValueError(
            f"method must be 'gcv', 'aic', 'bic', or 'evidence'; got {method!r}")

    grad_log = glin * lambdas * ln10                # chain rule d/d(log10 lam)
    return float(val), grad_log


def _optimize_lambdas_analytic(Phi, y, shapes, method, bounds, x0_log, names):
    """Analytic-gradient optimization of independent per-block lambdas.

    `shapes[k]` are disjoint-support penalty shapes (reg = sum_k lam_k*shapes[k]).
    Refines from `x0_log` (log10) with L-BFGS-B using the exact jacobian, then
    returns ridge_solve_with_diagnostics at the optimum plus `names[k]` lambdas.
    """
    Phi = np.asarray(Phi, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    shapes = [np.asarray(s, dtype=np.float64) for s in shapes]
    sizes = [int(np.sum(s > 0)) for s in shapes]
    C = Phi.T @ Phi
    b = Phi.T @ y
    log_bounds = [(np.log10(bounds[0]), np.log10(bounds[1]))] * len(shapes)

    res = minimize(
        lambda t: _criterion_valgrad_multi(Phi, C, b, y, shapes, sizes, t, method),
        x0=np.asarray(x0_log, dtype=np.float64), jac=True,
        method='L-BFGS-B', bounds=log_bounds)

    lam_opt = [float(10.0 ** t) for t in res.x]
    reg = np.zeros(Phi.shape[1])
    for k in range(len(shapes)):
        reg = reg + lam_opt[k] * shapes[k]
    final = ridge_solve_with_diagnostics(Phi, y, reg)
    for k, nm in enumerate(names):
        final[nm] = lam_opt[k]
    final['converged'] = bool(res.success)
    return final


def _resolve_lambda_optimizer(grad):
    """Return the per-block-lambda optimizer for a ``grad`` mode.

    ``'analytic'``/``'auto'`` → the closed-form-gradient optimizer;
    ``'jax'`` → the autodiff-gradient optimizer (lazily imported so the default
    numpy-only path never loads JAX). Both share the signature
    ``(Phi, y, shapes, method, bounds, x0_log, names)``.
    """
    if grad == 'jax':
        from .hyperopt_jax import optimize_lambdas_jax
        return optimize_lambdas_jax
    return _optimize_lambdas_analytic


def optimize_multi_lambda(Phi: np.ndarray, y: np.ndarray,
                          D: int, K1: int, K2: int = 0, P: int = 0,
                          strategy: str = 'variance',
                          method: str = 'gcv',
                          bounds: Tuple[float, float] = (1e-6, 1e2),
                          grad: str = 'numeric') -> Dict:
    """Optimize (lambda_1, lambda_2) jointly via a model-selection criterion.

    Args:
        Phi: (N, F) full feature matrix
        y: (N,) centered targets
        D, K1, K2, P: model structure parameters
        strategy: regularization strategy
        method: 'gcv', 'aic', 'bic', or 'evidence'
        bounds: search range for each lambda
        grad: 'numeric' (default; finite-difference L-BFGS-B, unchanged),
            'analytic'/'auto' (closed-form joint gradient from one A-factorization
            per L-BFGS-B evaluation — exact jacobian, fewer solves), or 'jax'
            (same objective, gradient by autodiff; matches 'analytic' to ~1e-10).

    Returns:
        Dict with optimal lambdas and diagnostics.
    """
    from .regularization import build_regularization_vector

    if grad not in ('numeric', 'analytic', 'auto', 'jax'):
        raise ValueError(
            f"grad must be 'numeric'/'analytic'/'auto'/'jax'; got {grad!r}")
    Phi_np = np.asarray(Phi, dtype=np.float64)
    y_np = np.asarray(y, dtype=np.float64)
    N, F = Phi_np.shape

    if K2 == 0 or P == 0:
        reg_struct = np.asarray(
            build_regularization_vector(D, K1, 0, 0, strategy, 1.0, 1.0),
            dtype=np.float64
        )
        return optimize_single_lambda(Phi_np, y_np, reg_struct, method, bounds,
                                      grad=grad)

    if grad in ('analytic', 'auto', 'jax'):
        shape1 = np.asarray(build_regularization_vector(
            D, K1, K2, P, strategy, 1.0, 0.0), dtype=np.float64)
        shape2 = np.asarray(build_regularization_vector(
            D, K1, K2, P, strategy, 0.0, 1.0), dtype=np.float64)
        _opt = _resolve_lambda_optimizer(grad)
        return _opt(
            Phi_np, y_np, [shape1, shape2], method, bounds,
            x0_log=[np.log10(0.001), np.log10(0.01)],
            names=['lambda_order1', 'lambda_order2'])

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
        elif method == 'aic':
            return r['aic']
        elif method == 'bic':
            return r['bic']
        elif method == 'evidence':
            return -r['log_evidence']
        raise ValueError(
            f"Unknown method '{method}'. Choose from: 'gcv', 'aic', 'bic', 'evidence'."
        )

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
    grad: str = 'numeric',
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
        grad: 'numeric' (default), 'analytic'/'auto', or 'jax' — gradient mode for
            the active-lambda optimization (see :func:`optimize_multi_lambda`).

    Returns:
        Dict with optimal lambdas and diagnostics at the optimum.
    """
    from .regularization import build_regularization_vector

    if grad not in ('numeric', 'analytic', 'auto', 'jax'):
        raise ValueError(
            f"grad must be 'numeric'/'analytic'/'auto'/'jax'; got {grad!r}")
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
        result = optimize_single_lambda(Phi_np, y_np, reg_struct, method, bounds,
                                        grad=grad)
        result['lambda_order1'] = result['lambda_opt']
        return result

    if n_active == 2 and 'lambda_residual' not in active_names:
        return optimize_multi_lambda(Phi_np, y_np, D, K1, K2, P,
                                     strategy, method, bounds, grad=grad)

    # --- General n-lambda optimization ---
    # Default initial values (log10 scale)
    defaults = {
        'lambda_order1': -3.0,   # 0.001
        'lambda_order2': -2.0,   # 0.01
        'lambda_order3': -1.0,   # 0.1
        'lambda_residual': 0.0,  # 1.0
    }

    if grad in ('analytic', 'auto', 'jax'):
        # One disjoint-support penalty shape per active lambda (unit lambda on
        # that order, zero elsewhere), then reg = sum_k lam_k * shape_k.
        def _unit_shape(name):
            u = {n: (1.0 if n == name else 0.0) for n in
                 ('lambda_order1', 'lambda_order2', 'lambda_order3', 'lambda_residual')}
            return np.asarray(build_regularization_vector(
                D, K1, K2, P, strategy,
                u['lambda_order1'], u['lambda_order2'],
                K3=K3, T=T, lambda_order3=u['lambda_order3'],
                M_residual=M_residual, lambda_residual=u['lambda_residual']),
                dtype=np.float64)
        shapes = [_unit_shape(nm) for nm in active_names]
        _opt = _resolve_lambda_optimizer(grad)
        return _opt(
            Phi_np, y_np, shapes, method, bounds,
            x0_log=[defaults[nm] for nm in active_names], names=active_names)

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
