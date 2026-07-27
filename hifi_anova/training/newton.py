"""Newton solver for the log-variance sub-problem.

The problem is CONVEX in w_h. Newton converges quadratically.
Armijo backtracking line search ensures monotone decrease and
prevents divergence when h values are extreme (exp(h) overflow).
Typically converges in 5-10 steps.
"""

import jax
import jax.numpy as jnp
from typing import Tuple

from ..model.variance_model import LOG_VAR_CLIP


def _objective(Psi_aug, theta, r2, reg_aug):
    """Evaluate the log-variance objective.

    L = sum_n [1/2 h(x_n) + r_n^2 / (2 exp(h(x_n)))] + 1/2 theta^T diag(reg) theta
    """
    h = Psi_aug @ theta
    # Clamp h to prevent exp overflow/underflow (same bound as prediction)
    h_clamped = jnp.clip(h, -LOG_VAR_CLIP, LOG_VAR_CLIP)
    sigma2 = jnp.exp(h_clamped)
    data_term = jnp.sum(0.5 * h_clamped + 0.5 * r2 / sigma2)
    reg_term = 0.5 * jnp.sum(reg_aug * theta ** 2)
    return data_term + reg_term


def newton_solve_log_variance(
    Psi: jnp.ndarray,
    squared_residuals: jnp.ndarray,
    w_h_init: jnp.ndarray,
    h0_init: float,
    reg_diag: jnp.ndarray,
    max_iter: int = 10,
    tol: float = 1e-6,
    damping: float = 1e-4,
    max_backtrack: int = 10,
    armijo_c: float = 1e-4,
) -> Tuple[jnp.ndarray, float]:
    """Newton's method for the log-variance sub-problem.

    Minimizes: sum_n [1/2 h(x_n) + r_n^2 / (2 exp(h(x_n)))] + 1/2 w_h^T diag(reg) w_h

    where h(x_n) = h0 + Psi_n @ w_h

    Uses Armijo backtracking line search for robustness on problems
    with extreme heteroscedasticity (large dynamic range in sigma^2).

    Args:
        Psi: (N, F_h) variance features
        squared_residuals: (N,) r_n^2
        w_h_init: (F_h,) initial log-variance coefficients
        h0_init: initial h0
        reg_diag: (F_h,) regularization for variance coefficients
        max_iter: maximum Newton iterations
        tol: convergence tolerance
        damping: Hessian damping for stability
        max_backtrack: maximum backtracking steps per Newton iteration
        armijo_c: Armijo sufficient decrease parameter (typically 1e-4)

    Returns:
        (w_h_optimal, h0_optimal)
    """
    Psi = jnp.asarray(Psi, dtype=jnp.float64)
    r2 = jnp.asarray(squared_residuals, dtype=jnp.float64)
    w_h = jnp.asarray(w_h_init, dtype=jnp.float64)
    h0 = jnp.float64(h0_init)
    reg_diag = jnp.asarray(reg_diag, dtype=jnp.float64)

    N, F = Psi.shape

    # Augmented features: [1, Psi] for joint (h0, w_h) optimization
    Psi_aug = jnp.concatenate([jnp.ones((N, 1), dtype=jnp.float64), Psi], axis=1)
    reg_aug = jnp.concatenate([jnp.zeros(1, dtype=jnp.float64), reg_diag])
    theta = jnp.concatenate([jnp.array([h0]), w_h])

    for iteration in range(max_iter):
        # h(x_n) = Psi_aug @ theta
        h = Psi_aug @ theta
        # Clamp to prevent exp overflow (exp(30) ~ 1e13, safe for float64);
        # LOG_VAR_CLIP is shared with predict_variance so fit == prediction.
        h_clamped = jnp.clip(h, -LOG_VAR_CLIP, LOG_VAR_CLIP)
        sigma2 = jnp.exp(h_clamped)

        # Gradient: g_n = 1/2 * (1 - r_n^2 / sigma_n^2) * psi_n
        # Plus regularization gradient: reg * theta
        ratio = r2 / sigma2  # r_n^2 / sigma_n^2
        g_per_sample = 0.5 * (1.0 - ratio)  # (N,)
        grad = Psi_aug.T @ g_per_sample + reg_aug * theta

        # Hessian: H = sum_n 1/2 * (r_n^2/sigma_n^2) * psi_n psi_n^T + diag(reg)
        h_weights = 0.5 * ratio  # (N,)
        H = (Psi_aug.T * h_weights[None, :]) @ Psi_aug + jnp.diag(reg_aug)

        # Add damping for numerical stability
        H = H + damping * jnp.eye(F + 1, dtype=jnp.float64)

        # Newton direction
        delta = jax.scipy.linalg.solve(H, grad, assume_a='pos')

        # Armijo backtracking line search
        obj_current = _objective(Psi_aug, theta, r2, reg_aug)
        directional_deriv = jnp.dot(grad, delta)  # grad^T delta (should be > 0)
        step_size = 1.0

        for _bt in range(max_backtrack):
            theta_trial = theta - step_size * delta
            obj_trial = _objective(Psi_aug, theta_trial, r2, reg_aug)
            # Armijo condition: f(x - t*d) <= f(x) - c*t*(grad^T d)
            if obj_trial <= obj_current - armijo_c * step_size * directional_deriv:
                break
            step_size *= 0.5

        theta = theta - step_size * delta

        # Check convergence
        if jnp.max(jnp.abs(step_size * delta)) < tol:
            break

    h0_opt = float(theta[0])
    w_h_opt = theta[1:]

    return w_h_opt, h0_opt
