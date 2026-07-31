"""JAX/autodiff variant of the model-selection criterion gradients.

This is the autodiff counterpart to the closed-form lambda-gradients in
:mod:`hifi_anova.training.hyperopt` (DEC-010). The model-selection criterion
(GCV / AIC / BIC / -log_evidence) is a *pure function of lambda*, so its gradient
can be obtained by :func:`jax.grad` instead of the hand-derived analytic formula.
The two agree to floating-point round-off (~1e-10) by construction; this module
exists as a modular, opt-in ``grad='jax'`` alternative and as an independent check
on the analytic gradients.

The objective mirrors :func:`hyperopt._criterion_valgrad_multi` exactly (primal
form, ``reg = sum_k lam_k * shape_k``, ``A = Phi^T Phi + diag(reg)``):

- ``w      = A^{-1} Phi^T y``
- ``rss    = ||y - Phi w||^2``            (from the residual directly, stable)
- ``df     = tr(A^{-1} Phi^T Phi)``
- ``log|K| = log|A| - log|R|``            (Sylvester; evidence only)

The free variable is ``loglams = log10(lambda)``, so :func:`jax.grad` returns the
gradient w.r.t. ``log10(lambda)`` directly — the same quantity the L-BFGS-B
wrappers consume, and the same as ``_criterion_valgrad_multi``'s ``grad_log``.

All linear algebra is float64: callers must have enabled x64
(``jax.config.update("jax_enable_x64", True)``); the public optimizers do.
"""

from functools import partial
from typing import Dict, List, Tuple

import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize


@partial(jax.jit, static_argnums=(6, 7))
def _criterion_core(loglams, Phi, y, C, b, shapes, N, method):
    """Minimization objective as a differentiable function of ``loglams``.

    Args (all float64 jnp arrays unless noted):
        loglams: (K,) log10 of each block lambda (the differentiation variable).
        Phi: (N, F) design. C = Phi^T Phi (F, F). b = Phi^T y (F,).
        shapes: (K, F) disjoint-support penalty shapes; reg = sum_k lam_k*shapes[k].
        N: int (static). method: 'gcv'/'aic'/'bic'/'evidence' (static).

    Returns the scalar minimization objective (the criterion itself, or
    ``-log_evidence`` for ``'evidence'``).
    """
    lambdas = jnp.power(10.0, loglams)                  # (K,)
    reg = jnp.sum(lambdas[:, None] * shapes, axis=0)    # (F,)
    A = C + jnp.diag(reg)

    w = jnp.linalg.solve(A, b)
    resid = y - Phi @ w
    rss = resid @ resid
    df = jnp.trace(jnp.linalg.solve(A, C))              # tr(A^{-1} C)

    if method == 'gcv':
        u = rss / N
        v = jnp.maximum(1e-10, 1.0 - df / N)
        return u / v ** 2
    if method in ('aic', 'bic'):
        mse = jnp.maximum(rss / N, 1e-15)
        pen = 2.0 if method == 'aic' else jnp.log(float(N))
        return N * jnp.log(mse) + pen * df
    if method == 'evidence':
        P = y @ resid
        sigma2 = jnp.maximum(P / N, 1e-15)
        _, logdetA = jnp.linalg.slogdet(A)
        # log|R| over the structurally-supported entries. The support is fixed
        # (union of the shapes' supports), so the mask is static in lambda; the
        # double-`where` keeps log's gradient finite on the masked-out entries.
        pos = jnp.sum(shapes, axis=0) > 0.0
        safe_reg = jnp.where(pos, reg, 1.0)
        logdetR = jnp.sum(jnp.where(pos, jnp.log(safe_reg), 0.0))
        logdetK = logdetA - logdetR
        log_ev = (-N / 2.0 * jnp.log(sigma2) - 0.5 * logdetK
                  - N / 2.0 * (1.0 + jnp.log(2.0 * jnp.pi)))
        return -log_ev
    raise ValueError(
        f"method must be 'gcv', 'aic', 'bic', or 'evidence'; got {method!r}")


_value_and_grad = jax.jit(
    jax.value_and_grad(_criterion_core, argnums=0), static_argnums=(6, 7))


def criterion_valgrad_jax(loglams, Phi, y, shapes, method: str
                          ) -> Tuple[float, np.ndarray]:
    """Autodiff value + gradient w.r.t. ``log10(lambda)`` for per-block lambdas.

    JAX counterpart of :func:`hyperopt._criterion_valgrad_multi`: the gradient
    comes from :func:`jax.grad`, not the closed-form formula, and the two agree to
    ~1e-10. Returns ``(value, grad)`` as Python float / numpy array (log10 space).

    ``Phi``/``y``/``shapes`` are upcast to float64; x64 must be enabled.
    """
    Phi = jnp.asarray(Phi, dtype=jnp.float64)
    y = jnp.asarray(y, dtype=jnp.float64)
    shapes = jnp.asarray(np.asarray(shapes, dtype=np.float64))
    loglams = jnp.asarray(np.asarray(loglams, dtype=np.float64))
    N = int(Phi.shape[0])
    C = Phi.T @ Phi
    b = Phi.T @ y
    val, grad = _value_and_grad(loglams, Phi, y, C, b, shapes, N, method)
    return float(val), np.asarray(grad, dtype=np.float64)


def optimize_lambdas_jax(Phi, y, shapes: List[np.ndarray], method: str,
                         bounds: Tuple[float, float], x0_log, names) -> Dict:
    """JAX/AD-gradient optimization of independent per-block lambdas.

    Mirror of :func:`hyperopt._optimize_lambdas_analytic` but with the L-BFGS-B
    jacobian supplied by :func:`jax.grad` (via :func:`criterion_valgrad_jax`).
    Refines from ``x0_log`` (log10) and returns ``ridge_solve_with_diagnostics``
    at the optimum plus ``names[k]`` lambdas.
    """
    # Deferred import: hyperopt imports this module lazily, so importing it back
    # here at call time avoids a circular import at module load.
    from .hyperopt import ridge_solve_with_diagnostics

    Phi = np.asarray(Phi, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    shapes = [np.asarray(s, dtype=np.float64) for s in shapes]
    log_bounds = [(np.log10(bounds[0]), np.log10(bounds[1]))] * len(shapes)

    res = minimize(
        lambda t: criterion_valgrad_jax(t, Phi, y, shapes, method),
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
