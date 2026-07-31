"""Joint mean + variance regularization selection (opt-in).

Selects the mean-model regularization strength together with the variance-model
penalty ``lambda_h`` against a single criterion — the "mean-fit vs variance-fit"
tradeoff. This is the location-scale (heteroscedastic-Gaussian) smoothing-parameter
selection problem; the design here follows the standard treatment (Wood, Pya &
Säfken 2016, JASA — the ``gaulss`` family in mgcv; Rigby & Stasinopoulos GAMLSS).

The default HiFi-ANOVA path keeps ``lambda_h`` fixed; this module is a separate,
explicitly-invoked selector and does not change the trainer. It operates on the
**core linear mean + log-variance ridge/Newton fit** (first-order, optionally with
a caller-supplied higher-order mean design), deliberately keeping the variance
model **coarser than the mean model** — an identifiability convention, not a
limitation (see the module notes).

Two criteria (DEC-012):

* ``'kfold_nll'`` (default) — k-fold Gaussian negative log-likelihood. A single
  squared residual carries only ``sigma^2 * chi^2_1`` of information about
  ``sigma^2(x)`` (CV of sqrt(2) per point), so variance selection is intrinsically
  noisy; k-fold (default k=5) averages that down. A robust/capped companion NLL is
  reported alongside — a large gap between the capped and uncapped argmin flags
  ``sigma^2``-collapse rather than a real optimum.
* ``'laml'`` — the joint Laplace-approximate marginal likelihood over
  ``(lambda_mean, lambda_h)``. Split-free (best where NLL selection noise is worst)
  and smooth in lambda. Formulated on ``y`` (**not** on the squared residuals — the
  residuals already consumed ``y``; conditioning the "evidence" on them would
  double-count the data): with the penalized joint NLL ``L_pen`` at the fitted mode,

      LAML(lambda) ≈ -L_pen + 1/2 log|R_pen| - 1/2 log|H_joint|   (+ const)

  ``R_pen`` is the penalty over penalized coordinates only (the ``f0``/``h0``
  intercepts are unpenalized fixed effects — REML-style — and are excluded from
  ``log|R|`` but present in ``H``). ``H_joint`` blocks are already formed by the
  fit: mean ``Phi^T W Phi + R_mean``, variance the Newton Hessian, and the cross
  block ``sum_n (r_n/sigma_n^2) phi_n psi_n^T`` (expectation zero under the model;
  block-diagonal by default, exact cross available and used as a
  mean/variance-confounding diagnostic).

Guards (all on by default): leverage-corrected residuals ``r^2/(1-lev)`` feeding
the variance step (removes the systematic component of mean-fit shrinkage leaking
into the variance model — the practical REML correction); a ``sigma^2`` floor when
forming the mean weights; a data-scaled lower bound on ``lambda_h`` with loud
boundary warnings; mean-first initialization; an optional weak MAP-II hyperprior on
``log10 lambda_h``; and a ``df_h <= N/10`` tripwire.

All linear algebra is float64.
"""

from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import jax.numpy as jnp

from .ridge import weighted_ridge_solve
from .newton import newton_solve_log_variance
from ..model.variance_model import LOG_VAR_CLIP

_LOG2PI = float(np.log(2.0 * np.pi))


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------

def _augment(Phi: np.ndarray) -> np.ndarray:
    """Prepend an intercept column of ones."""
    N = Phi.shape[0]
    return np.concatenate([np.ones((N, 1)), Phi], axis=1)


def _nll_per_point(y, mu, h):
    """Full Gaussian negative log-likelihood per point (includes 1/2 log 2pi)."""
    hc = np.clip(h, -LOG_VAR_CLIP, LOG_VAR_CLIP)
    return 0.5 * _LOG2PI + 0.5 * hc + 0.5 * (y - mu) ** 2 * np.exp(-hc)


def _leverage_diag(Phi_aug: np.ndarray, reg_aug: np.ndarray,
                   weights: np.ndarray) -> np.ndarray:
    """Diagonal of the weighted ridge hat matrix.

    ``S = Phi_aug (Phi_aug^T W Phi_aug + diag(reg_aug))^{-1} Phi_aug^T W``;
    ``lev_n = w_n * phi_n^T A^{-1} phi_n``. ``sum(lev) = tr(S) = df_mean``.
    """
    A = Phi_aug.T @ (weights[:, None] * Phi_aug) + np.diag(reg_aug)
    X = np.linalg.solve(A, Phi_aug.T)                     # A^{-1} Phi^T  (F, N)
    quad = np.einsum('nf,fn->n', Phi_aug, X)              # phi_n^T A^{-1} phi_n
    return weights * quad


# ---------------------------------------------------------------------------
# Joint fit at fixed (reg_mean, reg_var)
# ---------------------------------------------------------------------------

class _JointFit:
    """Result of an IRLS mean+variance fit at fixed penalties."""

    __slots__ = ('w_aug', 'f0', 'w_h', 'h0', 'mu', 'h', 'sigma2', 'weights',
                 'r2', 'lev', 'df_mean', 'reg_mean_aug', 'reg_var',
                 'Phi_aug', 'Psi', 'y', 'sigma2_floor')

    def nll(self) -> float:
        return float(np.mean(_nll_per_point(self.y, self.mu, self.h)))


def _joint_fit(Phi_aug: np.ndarray, Psi: np.ndarray, y: np.ndarray,
               reg_mean_aug: np.ndarray, reg_var: np.ndarray,
               *, leverage_correct: bool = True,
               sigma2_floor: float = 0.0,
               max_outer: int = 8, newton_max: int = 15,
               tol: float = 1e-5) -> _JointFit:
    """Alternating (IRLS) heteroscedastic fit at fixed penalties.

    Mean-first initialization (unit weights), then alternate: leverage-corrected
    Newton log-variance update, then weighted-ridge mean update. ``Phi_aug`` must
    already include the intercept column; ``reg_mean_aug[0]`` should be 0.
    """
    N = Phi_aug.shape[0]
    Fh = Psi.shape[1]

    # Homoscedastic init.
    w_aug = np.asarray(weighted_ridge_solve(
        jnp.asarray(Phi_aug), jnp.asarray(y), jnp.asarray(reg_mean_aug)),
        dtype=np.float64)
    mu = Phi_aug @ w_aug
    r2 = (y - mu) ** 2
    weights = np.ones(N)
    lev = _leverage_diag(Phi_aug, reg_mean_aug, weights)
    w_h = np.zeros(Fh)
    h0 = float(np.log(max(np.mean(r2), 1e-12)))
    h = h0 + Psi @ w_h
    prev = np.inf

    for _ in range(max_outer):
        # Variance step on leverage-corrected squared residuals.
        r2_eff = r2 / np.clip(1.0 - lev, 1e-3, 1.0) if leverage_correct else r2
        w_h, h0 = newton_solve_log_variance(
            jnp.asarray(Psi), jnp.asarray(r2_eff), jnp.asarray(w_h),
            float(h0), jnp.asarray(reg_var), max_iter=newton_max)
        w_h = np.asarray(w_h, dtype=np.float64)
        h = float(h0) + Psi @ w_h
        sigma2 = np.exp(np.clip(h, -LOG_VAR_CLIP, LOG_VAR_CLIP))
        sigma2_w = np.maximum(sigma2, sigma2_floor) if sigma2_floor > 0 else sigma2
        weights = 1.0 / sigma2_w

        # Mean step: weighted ridge.
        w_aug = np.asarray(weighted_ridge_solve(
            jnp.asarray(Phi_aug), jnp.asarray(y), jnp.asarray(reg_mean_aug),
            jnp.asarray(weights)), dtype=np.float64)
        mu = Phi_aug @ w_aug
        r2 = (y - mu) ** 2
        lev = _leverage_diag(Phi_aug, reg_mean_aug, weights)

        loss = float(np.mean(_nll_per_point(y, mu, h)))
        if abs(prev - loss) / (abs(prev) + 1e-12) < tol:
            break
        prev = loss

    fit = _JointFit()
    fit.w_aug, fit.f0, fit.w_h, fit.h0 = w_aug, float(w_aug[0]), w_h, float(h0)
    fit.mu, fit.h, fit.sigma2 = mu, h, sigma2
    fit.weights, fit.r2, fit.lev = weights, r2, lev
    fit.df_mean = float(np.sum(lev))
    fit.reg_mean_aug, fit.reg_var = reg_mean_aug, reg_var
    fit.Phi_aug, fit.Psi, fit.y = Phi_aug, Psi, y
    fit.sigma2_floor = sigma2_floor
    return fit


# ---------------------------------------------------------------------------
# Effective df and the joint LAML criterion
# ---------------------------------------------------------------------------

def _variance_hessians(fit: _JointFit):
    """Return (H_data_aug, H_var_aug) for the augmented [1, Psi] variance design.

    ``H_data`` is the unpenalized Fisher block ``sum_n 1/2 (r^2/sigma^2) psi psi^T``;
    ``H_var = H_data + diag([0, reg_var])``.
    """
    Psi_aug = _augment(fit.Psi)
    ratio = fit.r2 / np.maximum(fit.sigma2, 1e-300)
    hw = 0.5 * ratio
    H_data = Psi_aug.T @ (hw[:, None] * Psi_aug)
    reg_var_aug = np.concatenate([[0.0], fit.reg_var])
    return H_data, H_data + np.diag(reg_var_aug)


def effective_df_h(fit: _JointFit) -> float:
    """Effective df of the penalized log-variance fit: ``tr((H_data+R)^{-1} H_data)``
    (the standard mgcv EDF for a non-Gaussian component; includes the ``h0``
    intercept)."""
    H_data, H_var = _variance_hessians(fit)
    return float(np.trace(np.linalg.solve(H_var, H_data)))


def joint_laml(fit: _JointFit, cross: bool = False) -> Dict:
    """Joint Laplace-approximate marginal log-likelihood at the fitted mode.

    ``LAML = -L_pen + 1/2 log|R_pen| - 1/2 log|H_joint|`` (up to a lambda-independent
    additive constant). ``cross=False`` uses a block-diagonal ``H_joint`` (the exact
    cross block has expectation zero); ``cross=True`` uses the full joint Hessian.

    Returns a dict with ``laml`` (the log-evidence, higher = better) and the pieces,
    including ``cross_ratio`` = ||H_wh|| / sqrt(||H_ww|| ||H_hh||) as a
    mean/variance-confounding diagnostic.
    """
    Phi_aug, Psi, y = fit.Phi_aug, fit.Psi, fit.y
    W = fit.weights
    r = y - fit.mu
    sigma2 = np.maximum(fit.sigma2, 1e-300)

    # Penalized joint NLL at the mode (drop the constant 1/2 N log 2pi).
    data_nll = float(np.sum(0.5 * fit.h + 0.5 * r ** 2 / sigma2))
    w_pen = fit.w_aug[1:]
    pen_mean = 0.5 * float(w_pen @ (fit.reg_mean_aug[1:] * w_pen))
    pen_var = 0.5 * float(fit.w_h @ (fit.reg_var * fit.w_h))
    L_pen = data_nll + pen_mean + pen_var

    # log|R_pen| over strictly-penalized coordinates only.
    rm = fit.reg_mean_aug[1:]
    rv = fit.reg_var
    logdetR = (float(np.sum(np.log(rm[rm > 0]))) + float(np.sum(np.log(rv[rv > 0]))))

    # Hessian blocks.
    H_ww = Phi_aug.T @ (W[:, None] * Phi_aug) + np.diag(fit.reg_mean_aug)
    _, H_hh = _variance_hessians(fit)
    Psi_aug = _augment(Psi)
    H_wh = Phi_aug.T @ ((r / sigma2)[:, None] * Psi_aug)     # cross block

    s_ww = np.linalg.slogdet(H_ww)[1]
    s_hh = np.linalg.slogdet(H_hh)[1]
    if cross:
        top = np.concatenate([H_ww, H_wh], axis=1)
        bot = np.concatenate([H_wh.T, H_hh], axis=1)
        logdetH = np.linalg.slogdet(np.concatenate([top, bot], axis=0))[1]
    else:
        logdetH = s_ww + s_hh

    laml = -L_pen + 0.5 * logdetR - 0.5 * logdetH
    cross_ratio = float(np.linalg.norm(H_wh) /
                        np.sqrt(np.linalg.norm(H_ww) * np.linalg.norm(H_hh) + 1e-300))
    return {'laml': float(laml), 'L_pen': L_pen, 'logdetR': logdetR,
            'logdetH': float(logdetH), 'cross_ratio': cross_ratio}


# ---------------------------------------------------------------------------
# k-fold NLL criterion
# ---------------------------------------------------------------------------

def _kfold_indices(N: int, k: int, seed: int) -> List[np.ndarray]:
    rng = np.random.RandomState(seed)
    perm = rng.permutation(N)
    return [perm[i::k] for i in range(k)]


def _kfold_nll(Phi_aug, Psi, y, reg_mean_aug, reg_var, *, k, seed,
               leverage_correct, sigma2_floor, single_split=False,
               cap_ratio=50.0) -> Dict:
    """Fold-averaged held-out Gaussian NLL at fixed penalties, plus a robust
    (residual-ratio-capped) companion. Each fold refits ``(w, w_h)`` on the
    fold-train and scores the fold-test. ``single_split=True`` evaluates just the
    first fold as a single (1/k) hold-out — the cheap mode."""
    N = Phi_aug.shape[0]
    folds = _kfold_indices(N, k, seed)
    if single_split:
        folds = folds[:1]
    fold_nll, fold_nll_rob = [], []
    for te in folds:
        tr = np.setdiff1d(np.arange(N), te, assume_unique=False)
        fit = _joint_fit(Phi_aug[tr], Psi[tr], y[tr], reg_mean_aug, reg_var,
                         leverage_correct=leverage_correct,
                         sigma2_floor=sigma2_floor, max_outer=8)
        mu_te = Phi_aug[te] @ fit.w_aug
        h_te = fit.h0 + Psi[te] @ fit.w_h
        pp = _nll_per_point(y[te], mu_te, h_te)
        fold_nll.append(float(np.mean(pp)))
        # Robust companion: cap the standardized residual r^2/sigma^2.
        hc = np.clip(h_te, -LOG_VAR_CLIP, LOG_VAR_CLIP)
        ratio = np.minimum((y[te] - mu_te) ** 2 * np.exp(-hc), cap_ratio)
        pp_rob = 0.5 * _LOG2PI + 0.5 * hc + 0.5 * ratio
        fold_nll_rob.append(float(np.mean(pp_rob)))
    return {'nll': float(np.mean(fold_nll)),
            'nll_robust': float(np.mean(fold_nll_rob)),
            'fold_nll': fold_nll, 'fold_nll_robust': fold_nll_rob}


# ---------------------------------------------------------------------------
# Inner mean-lambda selection (whitened closed-form) given current weights
# ---------------------------------------------------------------------------

def _select_mean_lambda(Phi: np.ndarray, y: np.ndarray, weights: np.ndarray,
                        mean_reg_shape: np.ndarray, method: str,
                        bounds: Tuple[float, float]) -> float:
    """Select the scalar mean lambda by whitening with the current precision
    weights and reusing the closed-form ridge criterion (GCV/evidence/AIC/BIC)."""
    from .hyperopt import optimize_single_lambda
    sw = np.sqrt(weights)
    wsum = np.sum(weights)
    ybar = float(np.sum(weights * y) / wsum)
    pbar = (weights[:, None] * Phi).sum(0) / wsum
    Phi_c = (Phi - pbar[None, :]) * sw[:, None]
    y_c = (y - ybar) * sw
    res = optimize_single_lambda(Phi_c, y_c, mean_reg_shape, method=method,
                                 bounds=bounds, grad='numeric')
    return float(res['lambda_opt'])


# ---------------------------------------------------------------------------
# Coordinate fit: select mean lambda (given lambda_h) then return the joint fit
# ---------------------------------------------------------------------------

def _coordinate_fit(Phi, Phi_aug, Psi, y, lambda_h, var_reg_shape,
                    mean_reg_shape, mean_method, mean_bounds,
                    fixed_lambda_mean, *, rounds, leverage_correct,
                    sigma2_floor) -> Tuple[_JointFit, float]:
    """At a fixed ``lambda_h``, coordinate-descend the scalar mean lambda against
    its closed-form criterion (given the variance-induced weights) and the joint
    IRLS fit. Returns ``(fit, lambda_mean)``. If ``fixed_lambda_mean`` is given,
    the mean lambda is held there (no inner selection)."""
    reg_var = lambda_h * var_reg_shape
    lam_mean = fixed_lambda_mean if fixed_lambda_mean is not None else 1e-3
    fit = None
    for _ in range(rounds if fixed_lambda_mean is None else 1):
        reg_mean_aug = np.concatenate([[0.0], lam_mean * mean_reg_shape])
        fit = _joint_fit(Phi_aug, Psi, y, reg_mean_aug, reg_var,
                         leverage_correct=leverage_correct,
                         sigma2_floor=sigma2_floor)
        if fixed_lambda_mean is not None:
            break
        new_lam = _select_mean_lambda(Phi, y, fit.weights, mean_reg_shape,
                                      mean_method, mean_bounds)
        if abs(np.log10(new_lam) - np.log10(lam_mean)) < 1e-3:
            lam_mean = new_lam
            break
        lam_mean = new_lam
    # Final fit at the settled mean lambda.
    reg_mean_aug = np.concatenate([[0.0], lam_mean * mean_reg_shape])
    fit = _joint_fit(Phi_aug, Psi, y, reg_mean_aug, reg_var,
                     leverage_correct=leverage_correct, sigma2_floor=sigma2_floor)
    return fit, lam_mean


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def optimize_joint_lambda(
    Phi: np.ndarray,
    Psi: np.ndarray,
    y: np.ndarray,
    mean_reg_shape: np.ndarray,
    var_reg_shape: np.ndarray,
    *,
    criterion: str = 'kfold_nll',
    mean_method: str = 'gcv',
    fixed_lambda_mean: Optional[float] = None,
    lambda_h_bounds: Tuple[float, float] = (1e-2, 1e5),
    n_grid: int = 11,
    refine: bool = True,
    n_folds: int = 5,
    coord_rounds: int = 2,
    leverage_correct: bool = True,
    sigma2_floor_frac: float = 1e-6,
    hyperprior: Optional[Tuple[float, float]] = None,
    laml_cross: bool = False,
    seed: int = 0,
) -> Dict:
    """Jointly select ``(lambda_mean, lambda_h)`` for a heteroscedastic fit.

    Args:
        Phi: (N, F) mean design (no intercept column; added internally).
        Psi: (N, F_h) variance (log-variance) design.
        y: (N,) targets (original scale).
        mean_reg_shape: (F,) strictly-positive mean penalty shape (scaled by
            ``lambda_mean``).
        var_reg_shape: (F_h,) variance penalty shape (scaled by ``lambda_h``).
        criterion: ``'kfold_nll'`` (default), ``'split_nll'`` (single 1/n_folds
            hold-out — cheap mode), or ``'laml'`` (joint Laplace evidence).
        mean_method: closed-form criterion for the inner mean-lambda step
            (``'gcv'``/``'evidence'``/``'aic'``/``'bic'``). Ignored if
            ``fixed_lambda_mean`` is set.
        fixed_lambda_mean: hold the mean lambda fixed (skip inner selection);
            useful to isolate variance selection.
        lambda_h_bounds: (lo, hi) search range for ``lambda_h``.
        n_grid: number of log-spaced ``lambda_h`` grid points.
        refine: parabolic refinement in log10 around the grid minimum.
        n_folds: folds for the k-fold / split NLL criterion.
        coord_rounds: inner mean<->variance coordinate rounds per ``lambda_h``.
        leverage_correct: feed ``r^2/(1-lev)`` into the variance step (recommended).
        sigma2_floor_frac: floor ``sigma^2`` at this fraction of ``Var(y)`` when
            forming mean weights.
        hyperprior: optional ``(mu_log10, sigma_log10)`` weak Gaussian on
            ``log10 lambda_h`` (MAP-II); ``None`` disables it.
        laml_cross: use the exact cross-block in the LAML joint Hessian.
        seed: fold-assignment seed.

    Returns:
        dict with ``lambda_h``, ``lambda_mean``, the fitted-optimum diagnostics
        (``df_mean``, ``df_h``, ``nll``/``laml``), the full ``path`` over the
        ``lambda_h`` grid, and a ``warnings`` list (boundary / degeneracy / df_h
        tripwire / robust-argmin disagreement).
    """
    Phi = np.asarray(Phi, dtype=np.float64)
    Psi = np.asarray(Psi, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mean_reg_shape = np.asarray(mean_reg_shape, dtype=np.float64)
    var_reg_shape = np.asarray(var_reg_shape, dtype=np.float64)
    if criterion not in ('kfold_nll', 'split_nll', 'laml'):
        raise ValueError("criterion must be 'kfold_nll'/'split_nll'/'laml'; "
                         f"got {criterion!r}")
    N = Phi.shape[0]
    Phi_aug = _augment(Phi)
    vary = float(np.var(y))
    sigma2_floor = sigma2_floor_frac * vary
    single_split = (criterion == 'split_nll')
    n_splits = max(n_folds, 5) if single_split else n_folds

    warns: List[str] = []
    lo, hi = np.log10(lambda_h_bounds[0]), np.log10(lambda_h_bounds[1])

    def score(lam_h: float) -> Dict:
        """Full evaluation at one lambda_h: coordinate-fit + criterion + df's."""
        fit, lam_mean = _coordinate_fit(
            Phi, Phi_aug, Psi, y, lam_h, var_reg_shape, mean_reg_shape,
            mean_method, lambda_h_bounds, fixed_lambda_mean,
            rounds=coord_rounds, leverage_correct=leverage_correct,
            sigma2_floor=sigma2_floor)
        reg_mean_aug = np.concatenate([[0.0], lam_mean * mean_reg_shape])
        rec = {'lambda_h': float(lam_h), 'lambda_mean': float(lam_mean),
               'df_mean': fit.df_mean, 'df_h': effective_df_h(fit),
               'min_sigma2': float(np.min(fit.sigma2)),
               'full_nll': fit.nll()}
        if criterion == 'laml':
            L = joint_laml(fit, cross=laml_cross)
            rec['laml'] = L['laml']
            rec['cross_ratio'] = L['cross_ratio']
            obj = -L['laml']                      # minimize -LAML
        else:
            cv = _kfold_nll(Phi_aug, Psi, y, reg_mean_aug, lam_h * var_reg_shape,
                            k=n_splits, seed=seed, single_split=single_split,
                            leverage_correct=leverage_correct,
                            sigma2_floor=sigma2_floor)
            rec.update(cv)
            obj = cv['nll']
        # MAP-II: add -log hyperprior on log10 lambda_h.
        if hyperprior is not None:
            mu0, tau = hyperprior
            obj = obj + 0.5 * ((np.log10(lam_h) - mu0) / tau) ** 2
        rec['objective'] = float(obj)
        return rec

    # --- grid sweep ---
    grid = np.logspace(lo, hi, n_grid)
    path = [score(float(l)) for l in grid]
    objs = np.array([p['objective'] for p in path])
    best_i = int(np.argmin(objs))
    best = path[best_i]

    # --- parabolic refinement in log10 around the grid minimum ---
    if refine and 0 < best_i < len(grid) - 1:
        x = np.log10(grid[best_i - 1:best_i + 2])
        f = objs[best_i - 1:best_i + 2]
        denom = (f[0] - 2 * f[1] + f[2])
        if abs(denom) > 1e-12:
            x_star = x[1] - 0.5 * (x[2] - x[0]) * (f[2] - f[0]) / (2 * denom)
            x_star = float(np.clip(x_star, lo, hi))
            cand = score(10.0 ** x_star)
            path.append(cand)
            if cand['objective'] < best['objective']:
                best = cand

    # --- warnings / tripwires ---
    log_best = np.log10(best['lambda_h'])
    edge = (hi - lo) / (n_grid - 1)
    if log_best <= lo + edge:
        warns.append(
            f"lambda_h={best['lambda_h']:.3g} at/near the LOWER bound "
            f"{lambda_h_bounds[0]:.3g}: the criterion wants weaker variance "
            f"regularization (possible sigma^2-collapse). Widen bounds or inspect.")
    if log_best >= hi - edge:
        warns.append(
            f"lambda_h={best['lambda_h']:.3g} at/near the UPPER bound "
            f"{lambda_h_bounds[1]:.3g}: the criterion prefers (near-)homoscedastic.")
    if best['df_h'] > N / 10.0:
        warns.append(
            f"df_h={best['df_h']:.1f} exceeds N/10={N/10:.1f}: the variance model "
            f"may be fitting residual noise.")
    if best['min_sigma2'] <= 1.05 * sigma2_floor and sigma2_floor > 0:
        warns.append(
            "fitted sigma^2 hit the floor: variance-collapse guard active "
            "(selection may be untrustworthy at this lambda_h).")
    if criterion != 'laml':
        rob_i = int(np.argmin([p['objective'] if 'nll_robust' not in p
                               else p['nll_robust'] for p in path]))
        if abs(np.log10(path[rob_i]['lambda_h']) - log_best) > 2 * edge:
            warns.append(
                "capped (robust) NLL and raw NLL disagree on the optimum by "
                ">2 grid steps: a red flag for sigma^2-collapse, not a true "
                "optimum. Prefer 'laml' or widen k-fold.")

    for w in warns:
        warnings.warn(w, stacklevel=2)

    # Refit at the selected optimum and attach coefficients (for variance-Sobol
    # and downstream use).
    final_fit, final_lam_mean = _coordinate_fit(
        Phi, Phi_aug, Psi, y, best['lambda_h'], var_reg_shape, mean_reg_shape,
        mean_method, lambda_h_bounds, fixed_lambda_mean,
        rounds=coord_rounds, leverage_correct=leverage_correct,
        sigma2_floor=sigma2_floor)

    out = dict(best)
    out['lambda_mean'] = float(final_lam_mean)
    out['w_mean'] = final_fit.w_aug          # [f0 | mean coefficients]
    out['w_h'] = final_fit.w_h               # log-variance coefficients (no h0)
    out['h0'] = final_fit.h0
    out['sigma2'] = final_fit.sigma2
    out['warnings'] = warns
    out['path'] = path
    out['criterion'] = criterion
    out['n_folds'] = 1 if single_split else n_folds
    out['leverage_correct'] = leverage_correct
    return out
