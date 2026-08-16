"""Augmented profiled-intercept ridge analytics (Session-01 / Remark rem:intercept).

The Stage-D joint-GLS mean profiles an *unpenalized* intercept. The correct
diagnostic instrument is therefore the ridge on the augmented design
``Z = [1, Phi]`` with penalty ``diag(0, R)`` (Manuscript_Theoryv07
``rem:intercept``), so that df, leverage, residual df, LOO, and the sandwich
covariance all RE-PROFILE the intercept under response perturbation / row
deletion rather than holding the fitted ``f0`` fixed.

These are direct linear-algebra reconstructions on small synthetic weighted
designs with **nonzero weighted feature means** (the failure class of the old
feature-only path — a population-zero-mean basis need not have a zero *weighted*
column mean at finite N). They prove the primitive independently of the trainer.
"""
import numpy as np
import pytest

from hifi_anova.analysis.automl import (
    ridge_analytics, sandwich_covariance, sobol_confidence_intervals)
from hifi_anova.linalg import spd_inverse


def _fixture(N=200, F=5, seed=0, weighted=True):
    """A weighted ridge design whose weighted feature means are NONZERO."""
    rng = np.random.default_rng(seed)
    # Shift each column so the (unweighted and weighted) column means are far from
    # zero — this is exactly where the old feature-only intercept convention was
    # wrong for the profiled-GLS fit.
    Phi = rng.standard_normal((N, F)) + rng.uniform(1.0, 3.0, size=F)
    beta = rng.standard_normal(F)
    f0_true = 2.5
    y = f0_true + Phi @ beta + 0.3 * rng.standard_normal(N)
    reg = rng.uniform(0.01, 0.2, size=F)
    W = (0.2 + rng.uniform(0, 4, size=N)) if weighted else None
    return Phi, y, reg, W


def _direct_augmented(Phi, y, reg, W):
    """Reference profiled-GLS solve on Z=[1,Phi], penalty diag(0,reg)."""
    N = Phi.shape[0]
    Z = np.concatenate([np.ones((N, 1)), Phi], axis=1)
    reg_aug = np.concatenate([[0.0], reg])
    Wv = np.ones(N) if W is None else np.asarray(W, float)
    A = Z.T @ (Wv[:, None] * Z) + np.diag(reg_aug)
    A_inv = spd_inverse(A)
    theta = A_inv @ (Z.T @ (Wv * y))
    resid = y - Z @ theta
    lev = Wv * np.sum((Z @ A_inv) * Z, axis=1)          # a_n = W_n z^T A^-1 z
    M = A_inv @ (Z.T @ (Wv[:, None] * Z))
    df = float(np.trace(M))
    return dict(theta=theta, f0=theta[0], w=theta[1:], residuals=resid,
                leverages=lev, df=df, A_inv=A_inv, Z=Z, W=Wv)


@pytest.mark.parametrize('weighted', [True, False])
def test_augmented_matches_direct_reconstruction(weighted):
    """Intercept, slopes, fitted values, residuals == a direct NumPy augmented fit."""
    Phi, y, reg, W = _fixture(weighted=weighted)
    ana = ridge_analytics(Phi, y, reg, weights=W, profile_intercept=True)
    ref = _direct_augmented(Phi, y, reg, W)

    assert ana['f0'] == pytest.approx(ref['f0'], abs=1e-10)
    np.testing.assert_allclose(ana['w'], ref['w'], atol=1e-10)
    np.testing.assert_allclose(ana['theta'], ref['theta'], atol=1e-10)
    np.testing.assert_allclose(ana['residuals'], ref['residuals'], atol=1e-10)
    # Fitted values reconstruct from the split (f0, w).
    fitted = ana['f0'] + Phi @ ana['w']
    np.testing.assert_allclose(fitted, y - ana['residuals'], atol=1e-10)


@pytest.mark.parametrize('weighted', [True, False])
def test_leverage_sums_to_df_including_intercept(weighted):
    """sum(leverage) == df, and the intercept coordinate is counted in both."""
    Phi, y, reg, W = _fixture(weighted=weighted)
    ana = ridge_analytics(Phi, y, reg, weights=W, profile_intercept=True)
    assert ana['leverages'].sum() == pytest.approx(ana['df'], abs=1e-8)

    # The augmented df adds the (unpenalized) intercept coordinate: it is strictly
    # larger than the feature-only df, by ~1 (the free intercept parameter).
    feat = ridge_analytics(Phi, y - ana['f0'], reg, weights=W)
    assert ana['df'] > feat['df']
    assert ana['df'] - feat['df'] == pytest.approx(1.0, abs=0.05)


@pytest.mark.parametrize('weighted', [True, False])
def test_loo_residuals_match_bruteforce_intercept_reprofile(weighted):
    """Analytic PRESS residuals == brute-force deletion that RE-PROFILES f0."""
    Phi, y, reg, W = _fixture(N=120, F=4, weighted=weighted)
    ana = ridge_analytics(Phi, y, reg, weights=W, profile_intercept=True)
    loo_analytic = ana['loo_residuals']

    N = Phi.shape[0]
    Z = np.concatenate([np.ones((N, 1)), Phi], axis=1)
    reg_aug = np.concatenate([[0.0], reg])
    Wv = np.ones(N) if W is None else np.asarray(W, float)
    loo_brute = np.empty(N)
    for n in range(N):
        keep = np.ones(N, bool)
        keep[n] = False
        Zk, yk, Wk = Z[keep], y[keep], Wv[keep]
        A = Zk.T @ (Wk[:, None] * Zk) + np.diag(reg_aug)
        theta_mk = spd_inverse(A) @ (Zk.T @ (Wk * yk))   # re-profiles f0
        loo_brute[n] = y[n] - Z[n] @ theta_mk            # deleted residual

    np.testing.assert_allclose(loo_analytic, loo_brute, atol=1e-8)


def test_feature_shift_prediction_invariance():
    """Adding a constant to a feature column leaves fitted predictions invariant.

    With a profiled intercept, shifting column j by ``c`` is absorbed by ``f0``
    (slopes and fitted values are unchanged); the feature-only fit could not do
    this because it had no free intercept to absorb the shift.
    """
    Phi, y, reg, W = _fixture(weighted=True)
    ana = ridge_analytics(Phi, y, reg, weights=W, profile_intercept=True)
    fitted = ana['f0'] + Phi @ ana['w']

    Phi2 = Phi.copy()
    Phi2[:, 1] += 7.0
    ana2 = ridge_analytics(Phi2, y, reg, weights=W, profile_intercept=True)
    fitted2 = ana2['f0'] + Phi2 @ ana2['w']

    np.testing.assert_allclose(fitted2, fitted, atol=1e-8)
    np.testing.assert_allclose(ana2['residuals'], ana['residuals'], atol=1e-8)
    # The shift is absorbed by the intercept: f0 drops by c * w_j, slopes hold.
    np.testing.assert_allclose(ana2['w'], ana['w'], atol=1e-8)
    assert ana2['f0'] == pytest.approx(ana['f0'] - 7.0 * ana['w'][1], abs=1e-6)


def test_augmented_hc3_slope_covariance_matches_direct_sandwich():
    """Weighted HC3 slope covariance == the augmented sandwich's slope block.

    The slope-slope block of the FULL augmented sandwich is NOT the same as
    feeding the slope block of A_aug^{-1} to the feature-only meat — the
    intercept/feature cross terms matter — so this pins the coherent path.
    """
    Phi, y, reg, W = _fixture(weighted=True)
    ana = ridge_analytics(Phi, y, reg, weights=W, profile_intercept=True)

    # Direct augmented HC3 weighted sandwich, then slice the slope block.
    ref = _direct_augmented(Phi, y, reg, W)
    Z, A_inv, r, lev, Wv = ref['Z'], ref['A_inv'], ref['residuals'], ref['leverages'], ref['W']
    hc3 = np.abs(r) / (1.0 - np.clip(lev, 0.0, 1 - 1e-10)) * Wv
    meat = (Z * hc3[:, None]).T @ (Z * hc3[:, None])
    Cov_theta = A_inv @ meat @ A_inv
    Cov_slopes_ref = Cov_theta[1:, 1:]

    Cov_full = sandwich_covariance(ana['Z'], ana['A_inv'], ana['residuals'],
                                   hc='HC3', leverages=ana['leverages'],
                                   sample_weights=Wv)
    np.testing.assert_allclose(Cov_full[1:, 1:], Cov_slopes_ref, atol=1e-10)


def test_sobol_ci_excludes_intercept():
    """Sobol component slices exclude the intercept (D=1 share is identically 1).

    A single first-order block (D=1): the share is 1.0 computed from the slopes
    only, and the intercept column must not appear in the component energy. NOTE:
    with one component the delta-method gradient is zero, so this test does NOT
    probe the covariance wiring — that is
    ``test_sobol_ci_multiblock_reconstructs_augmented_covariance`` below.
    """
    rng = np.random.default_rng(3)
    N, B = 300, 4
    Phi = rng.standard_normal((N, B)) + 2.0        # nonzero weighted mean
    w_true = rng.standard_normal(B)
    y = 1.5 + Phi @ w_true + 0.2 * rng.standard_normal(N)
    reg = np.full(B, 0.05)
    W = 0.5 + rng.uniform(0, 2, N)
    G1 = np.eye(B)

    # Explicit per-group layout (one first-order block over all B columns): the
    # slice indexes the NONCONSTANT design Phi (intercept is never a column here),
    # so a correct implementation cannot let the profiled intercept leak in.
    ci = sobol_confidence_intervals(
        Phi, y, reg, D=1, groups=[(1, 0, slice(0, B), G1)],
        weights=W, profile_intercept=True)

    # One variable → its share is 1.0 (only block), computed from the slopes only.
    S, lo, hi = ci['first_order'][0]
    assert S == pytest.approx(1.0, abs=1e-9)
    assert 0.0 <= lo <= hi <= 1.0
    # df counts the profiled intercept (B slopes + 1 intercept, minus shrinkage).
    assert ci['df'] > 0.0


def test_sobol_ci_multiblock_reconstructs_augmented_covariance():
    """Multi-block Sobol CI is co-sourced from the AUGMENTED slope covariance.

    Two first-order groups with nonzero weighted feature means give non-trivial
    shares (S != 1), so the delta-method denominator-coupling gradient is nonzero
    and the interval genuinely depends on the covariance. Independently rebuild
    the full augmented HC3 slope covariance, the multi-block gradient, the SE, the
    Student-t critical value and the interval, and match the public CI. A negative
    control shows the *feature-only* covariance yields a measurably different
    interval — so the test fails if the profiled path silently used the wrong one.
    """
    from scipy.stats import t as sp_t

    rng = np.random.default_rng(11)
    N, B = 400, 3
    # Two blocks (variables) of B columns each, both with nonzero weighted means.
    Phi = np.concatenate([rng.standard_normal((N, B)) + 1.5,
                          rng.standard_normal((N, B)) + 2.5], axis=1)
    w_true = rng.standard_normal(2 * B)
    y = 0.7 + Phi @ w_true + 0.3 * rng.standard_normal(N)
    reg = np.full(2 * B, 0.05)
    W = 0.5 + rng.uniform(0, 2, N)
    G = np.eye(B)
    groups = [(1, 0, slice(0, B), G), (1, 1, slice(B, 2 * B), G)]
    alpha = 0.05

    ci = sobol_confidence_intervals(Phi, y, reg, D=2, groups=groups,
                                    weights=W, profile_intercept=True)

    # ---- independent reconstruction from the augmented analytics ----
    ana = ridge_analytics(Phi, y, reg, weights=W, profile_intercept=True)
    w = ana['w']
    Cov_theta = sandwich_covariance(ana['Z'], ana['A_inv'], ana['residuals'],
                                    hc='HC3', leverages=ana['leverages'],
                                    sample_weights=W)
    Cov_w = Cov_theta[1:, 1:]                      # augmented slope block
    F = 2 * B

    def _ci_from_cov(Cov):
        vars_ = [float(w[sl] @ G @ w[sl]) for (_, _, sl, G) in groups]
        tot = sum(vars_)
        U = np.zeros(F)
        for (_, _, sl, Gk) in groups:
            U[sl] = 2.0 * (Gk @ w[sl])
        Cov_U = Cov @ U
        UCU = float(U @ Cov_U)
        z = float(sp_t.ppf(1 - alpha / 2, df=max(ana['df_residual'], 1.0)))
        out = {}
        for k, (_, key, sl, Gk) in enumerate(groups):
            S = vars_[k] / tot
            own = U[sl]
            vnum = (float(own @ Cov[sl, sl] @ own)
                    - 2.0 * S * float(own @ Cov_U[sl]) + S * S * UCU)
            se = np.sqrt(max(0.0, vnum / (tot * tot)))
            out[key] = (S, max(0.0, S - z * se), min(1.0, S + z * se))
        return out

    ref = _ci_from_cov(Cov_w)
    for key in (0, 1):
        S_pub, lo_pub, hi_pub = ci['first_order'][key]
        S_ref, lo_ref, hi_ref = ref[key]
        assert S_pub == pytest.approx(S_ref, abs=1e-10)
        assert lo_pub == pytest.approx(lo_ref, abs=1e-10)
        assert hi_pub == pytest.approx(hi_ref, abs=1e-10)
        assert 0.0 < S_pub < 1.0            # non-trivial share ⇒ nonzero gradient

    # ---- negative control: the feature-only covariance is a DIFFERENT interval ----
    feat = ridge_analytics(Phi, y - ana['f0'], reg, weights=W)   # fixed-intercept
    Cov_feat = sandwich_covariance(Phi, feat['A_inv'], feat['residuals'],
                                   hc='HC3', leverages=feat['leverages'],
                                   sample_weights=W)
    wrong = _ci_from_cov(Cov_feat)
    # At least one bound must differ appreciably — otherwise the CI would be blind
    # to which covariance was used (the F4 defect).
    assert any(abs(ci['first_order'][k][j] - wrong[k][j]) > 1e-6
               for k in (0, 1) for j in (1, 2))


def test_profile_false_is_unchanged_feature_only_path():
    """profile_intercept=False is byte-identical to the legacy feature-only call."""
    Phi, y, reg, W = _fixture(weighted=True)
    a_default = ridge_analytics(Phi, y, reg, weights=W)
    a_explicit = ridge_analytics(Phi, y, reg, weights=W, profile_intercept=False)
    for k in ('df', 'sigma_hat', 'rss', 'loo_cv'):
        assert a_default[k] == a_explicit[k]
    np.testing.assert_array_equal(a_default['w'], a_explicit['w'])
    # And the feature-only path carries no intercept metadata.
    assert 'f0' not in a_default
    assert not a_default.get('intercept_profiled', False)
