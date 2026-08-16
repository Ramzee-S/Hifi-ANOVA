"""Joint mean+variance regularization selection (DEC-012).

Exercises the location-scale smoothing-parameter selector: heteroscedastic
recovery, the false-positive (homoscedastic) case, the leverage-correction guard,
the two criteria (k-fold NLL and joint LAML), the LAML cross-block diagnostic, and
the boundary / df_h tripwire warnings. Test scenarios are chosen so the claim under
test is actually exercised (e.g. an *additive* mean for the homoscedastic case,
where a first-order fit is well-specified — Ishigami is never truly homoscedastic
under a first-order mean because of its x1·x3^4 interaction).
"""
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import pytest

from hifi_anova.data.synthetic import generate_ishigami
from hifi_anova.data.preprocessing import preprocess_data
from hifi_anova.core.features import build_first_order_features, basis_size
from hifi_anova.core.gram import build_gram_matrix
from hifi_anova.training.regularization import (
    build_regularization_vector, build_variance_regularization_vector)
from hifi_anova.training.joint_lambda import (
    optimize_joint_lambda, joint_laml, _joint_fit, _augment)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _feats(X, K):
    return np.asarray(build_first_order_features(
        jnp.asarray(X), K, include_linear=True, basis_name='fourier'), np.float64)


def _mshape(K1):
    return np.asarray(build_regularization_vector(3, K1, 0, 0, 'variance', 1.0, 1.0),
                      np.float64)


def _vshape(Kh):
    return np.asarray(build_variance_regularization_vector(
        3, Kh, 'variance', 1.0, include_linear_h1=True, basis_name='fourier'),
        np.float64)


def _var_sobol(w_h, Kh):
    bs = basis_size(Kh, True, 'fourier')
    G = np.asarray(build_gram_matrix(Kh, True, 'fourier'), np.float64)
    vi = np.array([max(0.0, w_h[i*bs:(i+1)*bs] @ G @ w_h[i*bs:(i+1)*bs])
                   for i in range(3)])
    t = vi.sum()
    return vi / t if t > 0 else vi


def _ishigami_hetero(n=1200, K1=8, Kh=3, seed=0):
    X, y, _ = generate_ishigami(n_samples=n, noise_std=0.0, heteroscedastic=True,
                                variance_variable=2, sigma_min=0.3, sigma_max=3.0,
                                seed=seed)
    d = preprocess_data(X, y, seed=seed)
    xtr = d['x_train']
    ytr = np.asarray(d['y_train'], np.float64)
    return _feats(xtr, K1), _feats(xtr, Kh), ytr, Kh


def _additive(n, seed, sig=1.0):
    """Purely additive mean (first-order-representable) + constant noise."""
    rng = np.random.RandomState(seed)
    X = rng.uniform(0, 1, size=(n, 3))
    f = (np.sin(2*np.pi*X[:, 0]) + 0.7*np.cos(2*np.pi*X[:, 1])
         + 0.5*np.sin(4*np.pi*X[:, 2]))
    y = f + sig * rng.normal(size=n)
    return X, y


# --------------------------------------------------------------------------- #
# 1. Heteroscedastic recovery (the headline benchmark)
# --------------------------------------------------------------------------- #

@pytest.mark.slow
@pytest.mark.parametrize("criterion", ["kfold_nll", "laml"])
def test_ishigami_heteroscedastic_recovery(criterion):
    """On the heteroscedastic Ishigami (x3 drives the variance), the selector puts
    x3 at the top of the variance-Sobol spectrum for both criteria."""
    Phi, Psi, y, Kh = _ishigami_hetero()
    r = optimize_joint_lambda(Phi, Psi, y, _mshape(8), _vshape(Kh),
                              criterion=criterion, n_grid=9, n_folds=4, seed=1)
    s = _var_sobol(r['w_h'], Kh)
    assert int(np.argmax(s)) == 2, f"x3 not dominant: {s}"
    assert s[2] > 0.6, f"x3 variance-Sobol share too low: {s}"
    assert r['lambda_h'] > 0 and np.isfinite(r['df_h'])


@pytest.mark.slow
def test_kfold_and_laml_select_compatible_lambda_h():
    """The two criteria pick lambda_h within ~1.5 decades of each other."""
    Phi, Psi, y, Kh = _ishigami_hetero()
    rk = optimize_joint_lambda(Phi, Psi, y, _mshape(8), _vshape(Kh),
                               criterion='kfold_nll', n_grid=9, n_folds=4, seed=1)
    rl = optimize_joint_lambda(Phi, Psi, y, _mshape(8), _vshape(Kh),
                               criterion='laml', n_grid=9, seed=1)
    assert abs(np.log10(rk['lambda_h']) - np.log10(rl['lambda_h'])) < 1.5


@pytest.mark.slow
def test_selection_beats_fixed_extremes_on_cv_nll():
    """The selected lambda_h achieves a cross-validated NLL no worse than fixed
    lambda_h at either end of the search range (the whole point of selecting it).
    Compared on the same k-fold estimator the selection used, so it is robust to
    the flatness of the NLL valley."""
    Phi, Psi, y, Kh = _ishigami_hetero()
    r = optimize_joint_lambda(Phi, Psi, y, _mshape(8), _vshape(Kh),
                              criterion='kfold_nll', n_grid=9, n_folds=4, seed=1)
    grid = [p for p in r['path']][:9]          # the grid sweep (refine point appended)
    lo_end, hi_end = grid[0]['nll'], grid[-1]['nll']
    assert r['nll'] <= lo_end + 1e-9
    assert r['nll'] <= hi_end + 1e-9
    # and it is a genuine improvement over at least one extreme (interior optimum).
    assert r['nll'] < max(lo_end, hi_end) - 1e-6


# --------------------------------------------------------------------------- #
# 2. False-positive: homoscedastic truth with a well-specified mean
# --------------------------------------------------------------------------- #

@pytest.mark.slow
def test_homoscedastic_false_positive():
    """With constant-variance data and an adequate additive mean, the criterion
    prefers the (near-)homoscedastic limit: lambda_h at the upper bound, df_h -> ~1,
    and the boundary warning fires."""
    X, y = _additive(1600, seed=0, sig=1.0)
    Phi, Psi = _feats(X, 6), _feats(X, 3)
    r = optimize_joint_lambda(Phi, Psi, y, _mshape(6), _vshape(3),
                              criterion='kfold_nll', n_grid=11, n_folds=4, seed=1,
                              lambda_h_bounds=(1e-2, 1e5))
    assert np.log10(r['lambda_h']) > 3.5, "did not push toward homoscedastic"
    assert r['df_h'] < 3.0, f"variance model not collapsed: df_h={r['df_h']}"
    assert any('UPPER bound' in w for w in r['warnings'])
    # Fitted variance recovers the true noise level.
    assert abs(np.mean(r['sigma2']) - 1.0) < 0.25


# --------------------------------------------------------------------------- #
# 3. Leverage-correction guard
# --------------------------------------------------------------------------- #

@pytest.mark.slow
def test_leverage_correction_recovers_sigma2():
    """At high leverage (rich mean, small N), the leverage correction recovers the
    noise variance; without it the variance model underestimates sigma^2 (it fits
    the shrunk residuals E[r^2] ~ sigma^2 (1-lev))."""
    X, y = _additive(260, seed=0, sig=1.0)
    Phi, Psi = _feats(X, 8), _feats(X, 2)      # 51-col mean on ~180 train rows
    m, v = _mshape(8), _vshape(2)
    r_on = optimize_joint_lambda(Phi, Psi, y, m, v, criterion='kfold_nll',
                                 n_grid=9, n_folds=4, seed=1, leverage_correct=True)
    r_off = optimize_joint_lambda(Phi, Psi, y, m, v, criterion='kfold_nll',
                                  n_grid=9, n_folds=4, seed=1, leverage_correct=False)
    s_on, s_off = np.mean(r_on['sigma2']), np.mean(r_off['sigma2'])
    assert s_off < s_on, "leverage-off did not underestimate relative to on"
    assert abs(s_on - 1.0) < abs(s_off - 1.0), "correction did not move toward truth"


# --------------------------------------------------------------------------- #
# 4. LAML cross-block diagnostic
# --------------------------------------------------------------------------- #

def test_laml_cross_block_diagnostic():
    """joint_laml returns a finite log-evidence and a cross-block ratio; the exact
    cross-block changes the value (it is nonzero in finite samples) but does not
    blow it up."""
    Phi, Psi, y, Kh = _ishigami_hetero(n=800)
    m, v = _mshape(8), _vshape(Kh)
    Pa = _augment(Phi)
    reg_m = np.concatenate([[0.0], 1.0 * m])
    fit = _joint_fit(Pa, Psi, y, reg_m, 10.0 * v)
    L_block = joint_laml(fit, cross=False)
    L_full = joint_laml(fit, cross=True)
    assert np.isfinite(L_block['laml']) and np.isfinite(L_full['laml'])
    assert L_block['laml'] != L_full['laml']
    assert 0.0 <= L_block['cross_ratio'] < 10.0
    assert abs(L_full['laml'] - L_block['laml']) < abs(L_block['laml'])


# --------------------------------------------------------------------------- #
# 5. Tripwires, guards, and API
# --------------------------------------------------------------------------- #

@pytest.mark.slow
def test_df_h_tripwire_warns_on_rich_variance_model():
    """A rich variance model (Kh=8, 51 cols) forced into the low-lambda_h regime on
    modest N trips the df_h > N/10 diagnostic."""
    Phi, Psi, y, Kh = _ishigami_hetero(n=500, Kh=8)   # 51-col variance model
    r = optimize_joint_lambda(Phi, Psi, y, _mshape(8), _vshape(8),
                              criterion='kfold_nll', n_grid=7, n_folds=3, seed=1,
                              lambda_h_bounds=(1e-3, 1e0))
    assert any('df_h' in w for w in r['warnings'])


def test_criterion_validation():
    Phi, Psi, y, Kh = _ishigami_hetero(n=400)
    with pytest.raises(ValueError):
        optimize_joint_lambda(Phi, Psi, y, _mshape(8), _vshape(Kh),
                              criterion='bogus')


def test_fixed_mean_lambda_and_split_mode():
    """fixed_lambda_mean is honored; split_nll runs as the cheap single-holdout."""
    Phi, Psi, y, Kh = _ishigami_hetero(n=600)
    r = optimize_joint_lambda(Phi, Psi, y, _mshape(8), _vshape(Kh),
                              criterion='split_nll', fixed_lambda_mean=0.05,
                              n_grid=7, seed=1)
    assert r['lambda_mean'] == pytest.approx(0.05)
    assert r['n_folds'] == 1


@pytest.mark.slow
def test_hyperprior_pulls_lambda_h():
    """A weak MAP-II hyperprior centered below the data optimum shifts lambda_h
    downward relative to no prior."""
    Phi, Psi, y, Kh = _ishigami_hetero(n=800)
    m, v = _mshape(8), _vshape(Kh)
    r0 = optimize_joint_lambda(Phi, Psi, y, m, v, criterion='kfold_nll',
                               n_grid=9, n_folds=3, seed=1)
    r1 = optimize_joint_lambda(Phi, Psi, y, m, v, criterion='kfold_nll',
                               n_grid=9, n_folds=3, seed=1,
                               hyperprior=(np.log10(r0['lambda_h']) - 2.0, 0.5))
    assert r1['lambda_h'] < r0['lambda_h']


# --------------------------------------------------------------------------- #
# App. D determinant-derivative ("second") term — existence tripwire (DEC-033 / M4)
#
# LAML evaluates log|H_joint| at the fitted mode theta_hat(lambda); its
# lambda-derivative carries a mode-motion term that vanishes for fixed-variance
# ridge but NOT for the joint heteroscedastic likelihood (Manuscript_Theoryv06.tex
# App. D). Current selection is derivative-free, so nothing is wrong today; the fence
# (DEC-033) is that any FUTURE joint-LAML gradient must differentiate through the
# (leverage-corrected) mode solve. An AD gradient built over a NON-traced (frozen)
# mode silently drops the term. This tripwire is EXISTENCE-based: it verifies the
# determinant-piece frozen-vs-full gap is large, robust and heteroscedastic-only, so
# that a frozen-mode gradient would be caught. Evidence: dev/tests/
# m4_laml_determinant_probe.py; the leverage subtlety: M4b_leverage_nonstationarity_note.md.
# --------------------------------------------------------------------------- #
import copy as _copy
from hifi_anova.training.joint_lambda import _variance_hessians as _m4_vhess


def _m4_problem(hetero_slope, N, seed=1):
    """Self-contained fixture with a known determinant-term magnitude (matches the
    probe). hetero_slope=0 => homoscedastic control."""
    rng = np.random.RandomState(seed)
    x = rng.uniform(-1, 1, size=(N, 1))
    Phi = np.concatenate([x, x**2, x**3], axis=1)          # mean: x, x^2, x^3
    Psi = x.copy()                                          # log-variance: linear
    y = ((1.5 * x[:, 0] - 0.8 * x[:, 0] ** 2)
         + np.exp(-1.0 + hetero_slope * x[:, 0]) * rng.randn(N))
    return Phi, Psi, y


def _m4_fit(Phi, Psi, y, lam_h):
    Phi_aug = _augment(Phi)
    reg_mean_aug = np.concatenate([[0.0], 1e-2 * np.ones(Phi.shape[1])])
    return _joint_fit(Phi_aug, Psi, y, reg_mean_aug, lam_h * np.ones(Psi.shape[1]),
                      leverage_correct=True, sigma2_floor=0.0, tol=1e-8, max_outer=40)


def _m4_det_piece(fit):
    return -0.5 * joint_laml(fit)['logdetH']


def _m4_det_gap(Phi, Psi, y, t0=1.0, h=1e-3):
    """Central-FD of d(-1/2 log|H|)/d log10(lambda_h): 'full' re-solves the mode at
    each lambda_h (correct total derivative); 'frozen' holds the mode fixed (what an
    AD gradient over a non-traced mode returns). Returns (gap, full, frozen)."""
    lamp, lamm = 10.0 ** (t0 + h), 10.0 ** (t0 - h)
    full = (_m4_det_piece(_m4_fit(Phi, Psi, y, lamp))
            - _m4_det_piece(_m4_fit(Phi, Psi, y, lamm))) / (2 * h)
    fit0 = _m4_fit(Phi, Psi, y, 10.0 ** t0)

    def froz(lam):
        f = _copy.copy(fit0)
        f.reg_var = lam * np.ones(Psi.shape[1])
        return _m4_det_piece(f)
    frozen = (froz(lamp) - froz(lamm)) / (2 * h)
    return full - frozen, full, frozen


def _m4_cond_Hjoint(fit):
    """Condition number of the (block-diagonal) joint Hessian whose log-determinant is
    differentiated — asserted modest so the FD reference stays trustworthy."""
    Phi_aug = fit.Phi_aug
    H_ww = Phi_aug.T @ (fit.weights[:, None] * Phi_aug) + np.diag(fit.reg_mean_aug)
    _, H_hh = _m4_vhess(fit)
    a = H_ww.shape[0]
    B = np.zeros((a + H_hh.shape[0], a + H_hh.shape[0]))
    B[:a, :a] = H_ww
    B[a:, a:] = H_hh
    return float(np.linalg.cond(B))


@pytest.mark.slow
def test_appD_determinant_term_detectable_and_fenced():
    """Tripwire for App. D's joint-LAML determinant-derivative term (DEC-033 / M4).

    Existence-based: assert the determinant-piece gap is large, well conditioned, and
    heteroscedastic-only, so any future frozen-mode gradient that drops the term is
    caught. Asserts on the determinant piece, NOT total LAML (which confounds the
    DEC-028 leverage non-stationarity, ~0.05).
    """
    # Well-conditioned heteroscedastic fixture, away from any variance floor.
    Phi, Psi, y = _m4_problem(hetero_slope=1.3, N=200)
    fit0 = _m4_fit(Phi, Psi, y, 10.0)
    cond = _m4_cond_Hjoint(fit0)
    assert fit0.sigma2.min() > 1e-3, "fixture must sit away from any variance floor"
    assert cond < 1e6, f"fixture must be well conditioned for the FD reference (cond={cond:.2e})"

    gap_h, full_h, frozen_h = _m4_det_gap(Phi, Psi, y)

    # Homoscedastic control: the residual is a finite-sample floor, not a structural
    # term, so it is small and decays with N. It is a random quantity per realization,
    # so assert the trend SEED-AVERAGED over a wide N gap (a single (N,2N) pair is noisy).
    def _mean_floor(N, seeds=(1, 2, 3, 4)):
        return float(np.mean([abs(_m4_det_gap(*_m4_problem(0.0, N, seed=s))[0])
                              for s in seeds]))
    floor_smallN = _mean_floor(150)
    floor_bigN = _mean_floor(600)

    # (1) EXISTENCE: the term is materially nonzero on the heteroscedastic fixture.
    assert abs(gap_h) > 0.05, f"App. D determinant term should be detectable; got {gap_h:.4f}"
    # (2) GAP-TO-FLOOR >> 1: the heteroscedastic signal dominates the finite-sample floor.
    assert abs(gap_h) > 20.0 * floor_smallN, \
        f"gap-to-floor too small: |{gap_h:.4f}| vs floor |{floor_smallN:.4f}|"
    # (3) HETEROSCEDASTIC-ONLY: seed-averaged floor is small and shrinks with N.
    assert floor_bigN < floor_smallN, \
        f"homoscedastic floor must decay with N ({floor_bigN:.4f} !< {floor_smallN:.4f})"
    assert floor_bigN < 0.02, f"homoscedastic control not near zero: {floor_bigN:.4f}"

    # (4) MUST-PASS / MUST-FAIL-IF-FROZEN, gated until a production gradient exists.
    # A future analytic/AD joint-LAML gradient MUST match the FD-of-full reference and
    # MUST NOT equal the frozen-mode value (which drops App. D's term).
    from hifi_anova.training import joint_lambda as _jl
    if hasattr(_jl, 'joint_laml_grad'):  # pragma: no cover - lands with a future gradient
        g = float(_jl.joint_laml_grad(fit0)['d_neg_half_logdetH_dloglam'])
        assert g == pytest.approx(full_h, rel=0.05), "gradient must match FD-of-full-LAML"
        assert abs(g - frozen_h) > 0.5 * abs(gap_h), \
            "gradient must NOT equal the frozen-mode value (App. D term silently dropped)"
    else:
        # No production gradient yet: confirm the two candidate answers genuinely differ,
        # so the must-fail assertion retains power against a near-homoscedastic fixture edit.
        assert abs(full_h - frozen_h) > 0.05
