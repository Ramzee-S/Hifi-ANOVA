"""P1-1: Newton log-variance solver boundary correctness.

The fitted log-variance ``h(x_n)`` is confined to the box
``[-LOG_VAR_CLIP, LOG_VAR_CLIP]`` (shared with ``predict_variance``). Before the
fix the objective clamped the full data term while the gradient/Hessian were
those of the *unclamped* objective, so once a sample saturated the box the Newton
direction was no longer a descent direction of the line-searched objective and
the manuscript's "exact gradient and Hessian" claim did not hold there.

The fix makes the gradient and Hessian box-consistent: a clipped sample
contributes nothing (its clamped per-sample term is flat in theta). These tests
pin:

* analytic gradient/Hessian equal finite differences of the exact clamped
  objective in the INTERIOR (exact-derivative claim), and
* they also equal the finite differences when the clip is ACTIVE — near and
  beyond both bounds (the P1-1 acceptance criterion the old code failed);
* clipped samples contribute exactly zero to the derivatives;
* the Newton direction is a genuine descent direction when clipping is active;
* the solver converges to a stationary point on an interior problem and makes
  monotone progress (no divergence) on a clipping problem.
"""
import numpy as np

from hifi_anova.model.variance_model import LOG_VAR_CLIP
from hifi_anova.training.newton import (
    _grad_hess, _objective, newton_solve_log_variance,
)


def _fd_grad(f, x, eps=1e-6):
    """Central finite-difference gradient of scalar f at x."""
    x = np.asarray(x, dtype=np.float64)
    g = np.zeros_like(x)
    for i in range(x.size):
        xp = x.copy()
        xm = x.copy()
        xp[i] += eps
        xm[i] -= eps
        g[i] = (f(xp) - f(xm)) / (2.0 * eps)
    return g


def _fd_jac(vecf, x, eps=1e-6):
    """Central finite-difference Jacobian of vector-valued vecf at x."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    J = np.zeros((n, n))
    for i in range(n):
        xp = x.copy()
        xm = x.copy()
        xp[i] += eps
        xm[i] -= eps
        J[:, i] = (vecf(xp) - vecf(xm)) / (2.0 * eps)
    return J


def _obj_of_theta(Psi_aug, r2, reg_aug):
    return lambda t: float(_objective(Psi_aug, np.asarray(t, dtype=np.float64),
                                      r2, reg_aug))


def _grad_of_theta(Psi_aug, r2, reg_aug):
    return lambda t: np.asarray(
        _grad_hess(Psi_aug, np.asarray(t, dtype=np.float64), r2, reg_aug)[0],
        dtype=np.float64)


# ---------------------------------------------------------------------------
# Fixtures (deterministic, no RNG): a 3-parameter problem [h0, w_a, w_b].
# ---------------------------------------------------------------------------
def _problem():
    # h(x_n) = h0 + w_a * a_n + w_b * b_n ; Psi = [a, b], Psi_aug = [1, a, b]
    a = np.array([1.0, -1.0, 0.1, -0.1, 0.9, -0.9])
    b = np.array([0.3, -0.2, 0.5, -0.4, 0.1, -0.6])
    N = a.size
    Psi_aug = np.column_stack([np.ones(N), a, b])
    r2 = np.array([0.5, 2.0, 1.0, 3.0, 0.2, 1.5])
    reg_aug = np.array([0.0, 0.1, 0.1])  # no reg on the intercept
    return Psi_aug, r2, reg_aug


def _theta_interior():
    # h = [.., ..] all |h| well below LOG_VAR_CLIP=30
    return np.array([0.0, 2.0, 1.0])


def _problem_clip():
    """A configuration that saturates BOTH bounds while keeping the objective at
    a moderate magnitude (so finite differences stay well-conditioned).

    With ``w_a = 40``: h = 40*a. Samples with a >= 0.75 clip HIGH (h > 30) and
    a <= -0.75 clip LOW (h < -30); the rest are interior. The LOW-clip samples
    are given near-zero residuals, which is the only realistic reason h hits the
    floor: ``0.5*r^2*exp(30)`` would otherwise dominate the objective (and is a
    genuine property of the clamped objective, not a solver artifact).
    """
    a = np.array([0.9, 0.8, 0.1, -0.1, -0.8, -0.9])   # h = [36, 32, 4, -4, -32, -36]
    b = np.array([0.3, -0.2, 0.5, -0.4, 0.1, -0.6])
    N = a.size
    Psi_aug = np.column_stack([np.ones(N), a, b])
    r2 = np.array([0.5, 2.0, 1.0, 3.0, 1e-13, 1e-13])  # low-clip samples ~ zero
    reg_aug = np.array([0.0, 0.1, 0.1])
    theta = np.array([0.0, 40.0, 0.0])
    return Psi_aug, r2, reg_aug, theta


# ---------------------------------------------------------------------------
# Interior: exact gradient and Hessian.
# ---------------------------------------------------------------------------
def test_interior_gradient_matches_fd():
    Psi_aug, r2, reg_aug = _problem()
    theta = _theta_interior()
    h = Psi_aug @ theta
    assert np.all(np.abs(h) < LOG_VAR_CLIP)  # genuinely interior

    grad, _ = _grad_hess(Psi_aug, theta, r2, reg_aug)
    fd = _fd_grad(_obj_of_theta(Psi_aug, r2, reg_aug), theta)
    np.testing.assert_allclose(np.asarray(grad), fd, atol=1e-7, rtol=1e-6)


def test_interior_hessian_matches_fd():
    Psi_aug, r2, reg_aug = _problem()
    theta = _theta_interior()
    _, H = _grad_hess(Psi_aug, theta, r2, reg_aug)
    H_fd = _fd_jac(_grad_of_theta(Psi_aug, r2, reg_aug), theta)
    np.testing.assert_allclose(np.asarray(H), H_fd, atol=1e-6, rtol=1e-5)


# ---------------------------------------------------------------------------
# Clip active: the P1-1 regression. FD of the CLAMPED objective must match the
# analytic (box-consistent) gradient/Hessian near and beyond both bounds.
# ---------------------------------------------------------------------------
def test_clipped_gradient_matches_fd():
    Psi_aug, r2, reg_aug, theta = _problem_clip()
    h = Psi_aug @ theta
    # both bounds are exercised, and every sample is far (> eps) from the kink
    assert np.any(h > LOG_VAR_CLIP) and np.any(h < -LOG_VAR_CLIP)
    assert np.min(np.abs(np.abs(h) - LOG_VAR_CLIP)) > 1e-3

    grad, _ = _grad_hess(Psi_aug, theta, r2, reg_aug)
    fd = _fd_grad(_obj_of_theta(Psi_aug, r2, reg_aug), theta)
    np.testing.assert_allclose(np.asarray(grad), fd, atol=1e-7, rtol=1e-6)


def test_clipped_hessian_matches_fd():
    Psi_aug, r2, reg_aug, theta = _problem_clip()
    _, H = _grad_hess(Psi_aug, theta, r2, reg_aug)
    H_fd = _fd_jac(_grad_of_theta(Psi_aug, r2, reg_aug), theta)
    np.testing.assert_allclose(np.asarray(H), H_fd, atol=1e-6, rtol=1e-5)


def test_clipped_samples_contribute_zero():
    """A clipped sample must not enter the gradient or the Hessian curvature."""
    Psi_aug, r2, reg_aug, theta = _problem_clip()
    h = Psi_aug @ theta
    interior_idx = np.where(np.abs(h) < LOG_VAR_CLIP)[0]

    grad, H = _grad_hess(Psi_aug, theta, r2, reg_aug)

    # Recompute using ONLY the interior rows: must reproduce grad and H exactly.
    Pi = Psi_aug[interior_idx]
    sig2 = np.exp(np.clip(Pi @ theta, -LOG_VAR_CLIP, LOG_VAR_CLIP))
    ratio = r2[interior_idx] / sig2
    grad_ref = Pi.T @ (0.5 * (1.0 - ratio)) + reg_aug * theta
    H_ref = (Pi.T * (0.5 * ratio)[None, :]) @ Pi + np.diag(reg_aug)

    np.testing.assert_allclose(np.asarray(grad), grad_ref, atol=1e-12)
    np.testing.assert_allclose(np.asarray(H), H_ref, atol=1e-12)


def test_newton_direction_is_descent_when_clipped():
    """delta = H^{-1} grad gives grad^T delta > 0, and -delta decreases the
    (clamped) objective — i.e. the direction is a genuine descent direction."""
    Psi_aug, r2, reg_aug, theta = _problem_clip()
    grad, H = _grad_hess(Psi_aug, theta, r2, reg_aug)
    H = np.asarray(H) + 1e-4 * np.eye(theta.size)  # solver's damping
    delta = np.linalg.solve(H, np.asarray(grad))

    assert float(np.dot(np.asarray(grad), delta)) > 0.0

    obj = _obj_of_theta(Psi_aug, r2, reg_aug)
    f_at = obj(theta)
    # a small step opposite the Newton direction must reduce the objective
    assert obj(theta - 1e-3 * delta) < f_at


# ---------------------------------------------------------------------------
# Full solver: interior convergence and clip-active monotone progress.
# ---------------------------------------------------------------------------
def test_solver_converges_interior():
    Psi_aug, r2, reg_aug = _problem()
    Psi = Psi_aug[:, 1:]
    reg_diag = reg_aug[1:]
    w_h, h0 = newton_solve_log_variance(
        Psi, r2, w_h_init=np.zeros(2), h0_init=0.0, reg_diag=reg_diag,
        max_iter=50, tol=1e-10)
    theta = np.concatenate([[h0], np.asarray(w_h)])
    h = Psi_aug @ theta
    assert np.all(np.abs(h) < LOG_VAR_CLIP)  # solution stays interior
    grad, _ = _grad_hess(Psi_aug, theta, r2, reg_aug)
    assert float(np.max(np.abs(np.asarray(grad)))) < 1e-6  # stationary


def test_solver_monotone_progress_with_clipping():
    """A problem whose optimum drives some h beyond the box must not diverge:
    the final objective is finite and no larger than the initial objective."""
    # Extreme dynamic range in the residuals forces saturation.
    a = np.linspace(-1.0, 1.0, 12)
    Psi = a[:, None]
    r2 = np.exp(60.0 * a) + 1e-12   # spans exp(-60)..exp(60): far beyond the box
    Psi_aug = np.column_stack([np.ones_like(a), a])
    reg_aug = np.array([0.0, 1e-3])

    theta0 = np.array([0.0, 0.0])
    obj0 = _obj_of_theta(Psi_aug, r2, reg_aug)(theta0)

    w_h, h0 = newton_solve_log_variance(
        Psi, r2, w_h_init=np.zeros(1), h0_init=0.0, reg_diag=reg_aug[1:],
        max_iter=50, tol=1e-10)
    theta = np.concatenate([[h0], np.asarray(w_h)])
    objf = _obj_of_theta(Psi_aug, r2, reg_aug)(theta)

    assert np.isfinite(objf)
    assert objf <= obj0 + 1e-8
