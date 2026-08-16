"""M1 (DEC-031): the three-tier LOO hierarchy for the joint heteroscedastic model.

Covers Manuscript_Theoryv06 App. C:
  * the homoscedastic path stays byte-identical (loo_cv) and gains a cheap-form
    predictive loo_nll (tiers I/II/III coincide);
  * the KKT variance-floor detector and H_h conditioning check (the one genuine
    modelling judgment in M1) — validated on constructed floor-bound states, incl.
    the "at the clip but constraint inactive" case a bare value check would miss;
  * Tier II (one-step jackknife) agrees with the Tier-III exact nested refit at
    the claimed rate (per-obs O(N^-2), criterion O(N^-1)), tested across a ladder
    of N to confirm the *rate*, not just closeness;
  * result.loo(tier=1|2|3) wiring.
"""

import numpy as np
import pytest

from hifi_anova.analysis.automl import (
    ridge_analytics, joint_loo, exact_loo_nll,
)
from hifi_anova.training.fitted_design import VarianceDesign
from hifi_anova.model.variance_model import LOG_VAR_CLIP

_LOG2PI = float(np.log(2.0 * np.pi))


def _analytics(W, r, a=None):
    """A minimal weighted-analytics dict for joint_loo (only these keys used)."""
    r = np.asarray(r, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    a = np.zeros_like(r) if a is None else np.asarray(a, dtype=np.float64)
    return {'residuals': r, 'loo_residuals': r / (1.0 - a), 'weights': W}


# ---------------------------------------------------------------------------
# Homoscedastic path: byte-identical loo_cv + cheap-form predictive loo_nll
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_homoscedastic_loo_cv_byte_identical_and_nll_cheap_form():
    rng = np.random.RandomState(0)
    N, F = 200, 8
    Phi = rng.randn(N, F)
    y = Phi @ rng.randn(F) * 0.5 + rng.randn(N)
    reg = np.full(F, 1e-2)
    a = ridge_analytics(Phi, y, reg)

    # loo_cv is exactly the plug-in PRESS mean (unchanged by M1).
    r, lev = a['residuals'], a['leverages']
    assert a['loo_cv'] == float(np.mean((r / (1.0 - lev)) ** 2))
    assert a['loo_tier'] == 1

    # loo_nll is the free homoscedastic form 1/2 log s2 + loo_cv/(2 s2) + 1/2 log2pi.
    s2 = a['sigma2_hat']
    expect = 0.5 * np.log(s2) + a['loo_cv'] / (2.0 * s2) + 0.5 * _LOG2PI
    assert a['loo_nll'] == pytest.approx(expect, rel=1e-12)


@pytest.mark.smoke
def test_weighted_tier1_nll_identity():
    """The weighted Tier-I loo_nll equals -1/2 mean(log W) + 1/2 loo_cv + 1/2 log2pi."""
    rng = np.random.RandomState(3)
    N, F = 150, 6
    Phi = rng.randn(N, F)
    y = Phi @ rng.randn(F) * 0.3 + rng.randn(N)
    reg = np.full(F, 1e-2)
    W = np.exp(0.5 * rng.randn(N))            # positive precision weights
    a = ridge_analytics(Phi, y, reg, weights=W)
    expect = (-0.5 * np.mean(np.log(W)) + 0.5 * a['loo_cv'] + 0.5 * _LOG2PI)
    assert a['loo_nll'] == pytest.approx(expect, rel=1e-12)
    assert a['loo_tier'] == 1


# ---------------------------------------------------------------------------
# KKT variance-floor detector + H_h conditioning (check 4 — the detector itself)
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_floor_detection_active_vs_inactive():
    rng = np.random.RandomState(1)
    N, Fh = 40, 3
    Psi = rng.randn(N, Fh)
    var = VarianceDesign(Psi=Psi, reg_var=np.full(Fh, 1e-2),
                         w_h=np.zeros(Fh), h0=0.0)

    # Baseline: h=0 (W=1), moderate residuals everywhere -> nothing binds.
    j0 = joint_loo(_analytics(np.ones(N), 0.5 * np.ones(N)), var)
    assert j0['loo_variance_floor_active'] is False

    # ACTIVE floor: one point pinned at the lower clip with a ~zero residual, so
    # the unconstrained loss still wants h EVEN LOWER (rho < 1) — genuinely binding.
    Wa, ra = np.ones(N), 0.5 * np.ones(N)
    Wa[0] = np.exp(LOG_VAR_CLIP)     # h_raw = -LOG_VAR_CLIP (the floor)
    ra[0] = 0.0
    ja = joint_loo(_analytics(Wa, ra), var)
    assert ja['loo_variance_floor_active'] is True
    assert ja['loo_tier2_guarantee_holds'] is False

    # INACTIVE despite sitting AT the clip: same point at the floor but with a
    # normal residual — rho >> 1, the loss wants h HIGHER, so the lower clip is
    # NOT its active constraint (multiplier ~ 0). A bare value-at-clip check would
    # wrongly flag this; the KKT gradient test must not.
    Wb, rb = np.ones(N), 0.5 * np.ones(N)
    Wb[0] = np.exp(LOG_VAR_CLIP)
    rb[0] = 1.0
    jb = joint_loo(_analytics(Wb, rb), var)
    assert jb['loo_variance_floor_active'] is False


@pytest.mark.smoke
def test_h_conditioning_flag():
    rng = np.random.RandomState(2)
    N = 80
    r = 0.7 * np.ones(N)          # rho = 0.49 with W=1
    Psi = rng.randn(N, 3)

    var_ok = VarianceDesign(Psi=Psi, reg_var=np.full(3, 1e-1),
                            w_h=np.zeros(3), h0=0.0)
    j_ok = joint_loo(_analytics(np.ones(N), r), var_ok)
    assert j_ok['variance_hessian_ill_conditioned'] is False
    assert j_ok['loo_tier2_guarantee_holds'] is True

    # Duplicate a column and starve its penalty -> near-null direction in H_h.
    Psi_bad = np.column_stack([Psi[:, 0], Psi[:, 0], Psi[:, 1]])
    var_bad = VarianceDesign(Psi=Psi_bad,
                             reg_var=np.array([1e-14, 1e-14, 1e-1]),
                             w_h=np.zeros(3), h0=0.0)
    j_bad = joint_loo(_analytics(np.ones(N), r), var_bad)
    assert j_bad['variance_hessian_ill_conditioned'] is True
    assert j_bad['loo_tier2_guarantee_holds'] is False


@pytest.mark.smoke
def test_numerical_guard_caps_and_counts():
    """The correction cap clips |delta h| and flags/counts it, keeping loo_nll finite."""
    rng = np.random.RandomState(4)
    N, Fh = 60, 3
    Psi = rng.randn(N, Fh)
    var = VarianceDesign(Psi=Psi, reg_var=np.full(Fh, 1e-3),
                         w_h=np.zeros(Fh), h0=0.0)
    W = np.ones(N)
    r = np.linspace(0.1, 3.0, N)        # varied residuals -> a spread of |delta h|

    # A tiny cap forces clipping of the larger corrections; they are bounded and
    # counted, and the reported loo_nll stays finite.
    j = joint_loo(_analytics(W, r), var, delta_cap=0.01)
    assert j['loo_nll_correction_clipped'] is True
    assert j['n_correction_clipped'] >= 1
    assert np.all(np.abs(j['per_point_delta_h']) <= 0.01 + 1e-12)
    assert np.isfinite(j['loo_nll'])

    # A generous cap leaves this well-behaved state untouched (nothing clipped).
    j2 = joint_loo(_analytics(W, r), var, delta_cap=50.0)
    assert j2['loo_nll_correction_clipped'] is False
    assert j2['n_correction_clipped'] == 0


# ---------------------------------------------------------------------------
# Tier II vs Tier III: the rate (per-obs O(N^-2), criterion O(N^-1))
# ---------------------------------------------------------------------------

def _planted_joint(N, seed):
    """A correctly-specified 1-D location-scale model (interior, nonsingular)."""
    rng = np.random.RandomState(seed)
    x = rng.rand(N)
    Phi = (x - 0.5).reshape(-1, 1)          # mean: f0 + slope*(x-0.5)
    Psi = (x - 0.5).reshape(-1, 1)          # log-var: h0 + b*(x-0.5)
    h_true = -0.4 + 1.4 * (x - 0.5)
    mean_true = 1.0 * (x - 0.5)
    y = mean_true + rng.randn(N) * np.exp(0.5 * h_true)
    reg_mean = np.full(1, 1e-4)
    reg_var = np.full(1, 1e-4)
    return Phi, y, reg_mean, Psi, reg_var


def _tier2_and_tierI(Phi, y, reg_mean, Psi, reg_var):
    """Tier II (joint_loo) and Tier I, both from ONE converged full-data joint
    fit — so the only Tier-II-vs-Tier-III difference is the one-step vs exact
    jackknife (leverage correction OFF on both sides isolates the rate)."""
    from hifi_anova.training.joint_lambda import _joint_fit, _augment
    Phi_aug = _augment(Phi)
    reg_mean_aug = np.concatenate([[0.0], reg_mean])
    fit = _joint_fit(Phi_aug, Psi, y, reg_mean_aug, reg_var,
                     leverage_correct=False, sigma2_floor=0.0, max_outer=25)
    r = fit.y - fit.mu
    a = fit.lev
    W = fit.weights
    ana = {'residuals': r, 'loo_residuals': r / (1.0 - a), 'weights': W}
    var = VarianceDesign(Psi=Psi, reg_var=reg_var, w_h=fit.w_h, h0=fit.h0)
    j2 = joint_loo(ana, var)
    loo_cv_I = float(np.mean(W * (r / (1.0 - a)) ** 2))
    loo_nll_I = -0.5 * float(np.mean(np.log(W))) + 0.5 * loo_cv_I + 0.5 * _LOG2PI
    return j2, loo_nll_I


def _newton_var_np(Psi, r2, reg_var, w0, h0, iters=60):
    """Exact convex Newton for the log-variance NLL on raw r^2 — the SAME
    objective (clamp, penalized Hessian) as joint_loo's one-step and
    training.newton, in plain NumPy for speed. Returns theta = [h0, w_h]."""
    Psi_aug = np.column_stack([np.ones(len(r2)), np.asarray(Psi, float)])
    reg_aug = np.concatenate([[0.0], np.asarray(reg_var, float)])
    theta = np.concatenate([[h0], np.asarray(w0, float)])
    r2 = np.asarray(r2, float)
    for _ in range(iters):
        h = np.clip(Psi_aug @ theta, -LOG_VAR_CLIP, LOG_VAR_CLIP)
        ratio = r2 / np.exp(h)
        grad = Psi_aug.T @ (0.5 * (1.0 - ratio)) + reg_aug * theta
        H = (Psi_aug.T * (0.5 * ratio)[None, :]) @ Psi_aug + np.diag(reg_aug)
        step = np.linalg.solve(H, grad)
        theta = theta - step
        if np.max(np.abs(step)) < 1e-12:
            break
    return theta


@pytest.mark.integration
def test_tier2_variance_jackknife_rate_On2():
    """Per-obs O(N^-2): the one-step variance jackknife (joint_loo) vs the EXACT
    deleted variance refit, with the mean held fixed so only the variance block's
    one-step error is measured."""
    Ns = [150, 300, 600]
    d_pp = []
    for N in Ns:
        rng = np.random.RandomState(2000 + N)
        x = rng.rand(N)
        Psi = (x - 0.5).reshape(-1, 1)
        h_true = -0.4 + 1.4 * (x - 0.5)
        r = rng.randn(N) * np.exp(0.5 * h_true)        # residuals about a fixed (zero) mean
        r2 = r ** 2
        reg_var = np.full(1, 1e-4)
        # Full-data variance fit to the exact stationary point.
        theta = _newton_var_np(Psi, r2, reg_var, np.zeros(1),
                               float(np.log(np.mean(r2))))
        h_full = np.clip(theta[0] + Psi @ theta[1:], -LOG_VAR_CLIP, LOG_VAR_CLIP)
        W = np.exp(-h_full)
        # Tier-II one-step deleted predictions (mean fixed => loo_residuals = r).
        j2 = joint_loo({'residuals': r, 'loo_residuals': r, 'weights': W},
                       VarianceDesign(Psi=Psi, reg_var=reg_var,
                                      w_h=theta[1:], h0=float(theta[0])),
                       delta_cap=50.0)
        h_del_II = j2['per_point_h_del']
        # Exact deleted-variance predictions per fold.
        h_del_exact = np.empty(N)
        for n in range(N):
            keep = np.ones(N, dtype=bool); keep[n] = False
            th = _newton_var_np(Psi[keep], r2[keep], reg_var,
                                theta[1:], float(theta[0]))
            h_del_exact[n] = np.clip(th[0] + Psi[n] @ th[1:],
                                     -LOG_VAR_CLIP, LOG_VAR_CLIP)
        d_pp.append(float(np.mean(np.abs(h_del_II - h_del_exact))))
    slope = np.log(d_pp[-1] / d_pp[0]) / np.log(Ns[-1] / Ns[0])
    print(f"\n[var-jackknife] Ns={Ns} d_pp={d_pp} slope={slope:.3f}")
    assert d_pp[-1] < d_pp[1] < d_pp[0]                 # monotone shrink
    assert slope <= -1.6, f"variance one-step slope {slope:.3f} shallower than ~N^-2"


@pytest.mark.integration
def test_tier2_vs_tier3_criterion_rate_On1():
    """Criterion-level convergence of the full Tier-II hybrid to the fully-exact
    Tier-III oracle (~O(N^-1)), and Tier II strictly beats Tier I."""
    Ns = [120, 240, 480]
    d_crit, closer = [], []
    for N in Ns:
        Phi, y, reg_mean, Psi, reg_var = _planted_joint(N, seed=1000 + N)
        j2, nll_I = _tier2_and_tierI(Phi, y, reg_mean, Psi, reg_var)
        j3 = exact_loo_nll(Phi, y, reg_mean, Psi, reg_var,
                           leverage_correct=False, max_outer=25)
        d_crit.append(abs(j2['loo_nll'] - j3['loo_nll']))
        closer.append(abs(j2['loo_nll'] - j3['loo_nll'])
                      < abs(nll_I - j3['loo_nll']))
    slope_crit = np.log(d_crit[-1] / d_crit[0]) / np.log(Ns[-1] / Ns[0])
    print(f"\n[criterion] Ns={Ns} d_crit={d_crit} slope={slope_crit:.3f}")
    assert all(closer), closer                          # II corrects I's optimism
    assert d_crit[-1] < d_crit[0]                        # converges to the oracle
    assert slope_crit <= -0.7, f"criterion slope {slope_crit:.3f} shallower than ~N^-1"


# ---------------------------------------------------------------------------
# result.loo(tier=...) wiring through the one-call API
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_result_loo_method_hetero():
    from hifi_anova.api import hifi_anova
    from hifi_anova.data.synthetic import generate_ishigami
    X, y, _ = generate_ishigami(n_samples=1200, heteroscedastic=True, seed=0)
    res = hifi_anova(X, y, heteroscedastic=True, K1=8, K2=4, Kh=3,
                     seed=42, verbose=False, max_outer_iter=6)

    with pytest.raises(ValueError):
        res.loo(tier=4)

    # Invariant: this test exists to exercise the Tier-II heteroscedastic LOO
    # wiring, so the DEC-039 guard KEEPING the genuinely heteroscedastic Ishigami
    # signal is a precondition. Assert it (fail-fast) rather than a pytest.skip
    # that would silently mask a Stage-D regression as a green skip (X5 §7.1/§9).
    assert res.loo_tier == 2, (
        f"hetero fixture must select Stage D (loo_tier={res.loo_tier}); the "
        "Tier-II LOO wiring under test requires a weighted Stage-D fit.")

    # Reported default is Tier II; the method reproduces it.
    d2 = res.loo(tier=2)
    assert d2['loo_tier'] == 2
    assert res.loo_nll == pytest.approx(d2['loo_nll'], rel=1e-9)
    assert res.loo_cv == pytest.approx(d2['loo_cv'], rel=1e-9)

    # Tier I is reachable and (weakly) more optimistic than Tier II.
    d1 = res.loo(tier=1)
    assert d1['loo_tier'] == 1
    assert d2['loo_nll'] >= d1['loo_nll'] - 1e-6

    # Tier III via the shared oracle on a small subset (bounded cost) is finite.
    rec = res._fitted_design
    y_ff = rec.y_centered + rec.f0
    sub = np.arange(0, rec.Phi.shape[0], max(1, rec.Phi.shape[0] // 15))[:15]
    t3 = exact_loo_nll(rec.Phi, y_ff, rec.reg_diag,
                       rec.variance.Psi, rec.variance.reg_var, subset=sub)
    assert t3['loo_tier'] == 3
    assert np.all(np.isfinite(t3['per_point_nll']))


@pytest.mark.integration
def test_result_loo_homoscedastic_tiers_coincide():
    from hifi_anova.api import hifi_anova
    from hifi_anova.data.synthetic import generate_ishigami
    X, y, _ = generate_ishigami(n_samples=900, noise_std=0.3, seed=0)
    res = hifi_anova(X, y, mode="second", K1=8, K2=4, seed=42, verbose=False)
    assert res.loo_tier == 1
    assert res.loo_tier2_guarantee_holds is None
    for t in (1, 2, 3):
        d = res.loo(tier=t)
        assert d['loo_nll'] == pytest.approx(res.loo_nll, rel=1e-9)


@pytest.mark.slow
def test_result_loo_tier3_full_small():
    """End-to-end result.loo(tier=3): the exact nested refit through the API."""
    from hifi_anova.api import hifi_anova
    from hifi_anova.data.synthetic import generate_ishigami
    X, y, _ = generate_ishigami(n_samples=220, heteroscedastic=True, seed=2)
    res = hifi_anova(X, y, heteroscedastic=True, K1=6, K2=3, Kh=2,
                     seed=42, verbose=False, max_outer_iter=5)
    d3 = res.loo(tier=3)
    assert np.isfinite(d3['loo_nll'])
    if res.loo_tier == 2:
        assert d3['loo_tier'] == 3
        assert d3['n_folds'] == res._fitted_design.Phi.shape[0]
