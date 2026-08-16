"""Phase-3 end-to-end: the heteroscedastic one-call uses the weighted fit.

A Stage-D fit must report *weighted* (GLS) diagnostics drawn from the design the
trainer actually solved, with sigma_hat reinterpreted as a whitened calibration
scale and sigma^2(x) available separately (the two-fit convention's predictive
side). Exercises the full alternating fit, so it is an integration test.
"""

import numpy as np
import pytest

from hifi_anova.api import hifi_anova
from hifi_anova.analysis.automl import ridge_analytics, joint_loo
from hifi_anova.data.synthetic import generate_ishigami

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def hetero_result():
    X, y, _ = generate_ishigami(n_samples=1600, heteroscedastic=True, seed=0)
    res = hifi_anova(X, y, heteroscedastic=True, K1=8, K2=4, Kh=3,
                     seed=42, verbose=False, max_outer_iter=6)
    rec = res.train_results['fitted_design']
    # This is a heteroscedastic-INVARIANT fixture: every test below characterizes
    # the *selected* weighted Stage-D fit, so selection is a precondition, not a
    # data-dependent maybe. Assert it here (fail-fast, single enforcement point)
    # instead of a per-test pytest.skip that would silently turn a real Stage-D
    # regression into a green skip (X5 analysis §7.1/§9). The Ishigami het signal
    # is strong and calibrated to keep under the DEC-039 guard on both the local
    # and ws2 environments; if this ever reverts, strengthen the fixture's
    # heteroscedastic signal — do NOT re-introduce a conditional skip.
    sd = res.train_results.get('stage_D', {})
    assert sd.get('selected') == 'heteroscedastic' and rec.is_weighted, (
        "hetero fixture must select Stage D (got "
        f"selected={sd.get('selected')!r}, is_weighted={rec.is_weighted}); "
        "the invariant tests below require a weighted Stage-D fit.")
    return res, rec


def test_fixture_selects_stage_d_with_joint_gls_mean(hetero_result):
    # The guard keeps genuine heteroscedasticity (DEC-039), and the shipped mean
    # is the profiled joint-GLS estimator — asserted explicitly, and recorded as
    # such in both the training results and the fitted-design record (DEC-039
    # estimator-convention provenance).
    res, rec = hetero_result
    from hifi_anova.training.fitted_design import (
        MEAN_INTERCEPT_PROFILED_JOINT_GLS)
    sd = res.train_results['stage_D']
    assert sd['selected'] == 'heteroscedastic'
    assert sd['mean_intercept_mode'] == MEAN_INTERCEPT_PROFILED_JOINT_GLS
    assert rec.mean_intercept_mode == MEAN_INTERCEPT_PROFILED_JOINT_GLS
    # The unit-weight attribution companion is the ordinary centered mean.
    from hifi_anova.training.fitted_design import MEAN_INTERCEPT_UNWEIGHTED
    assert rec.interpretive.mean_intercept_mode == MEAN_INTERCEPT_UNWEIGHTED


def test_weighted_record_present_and_consistent(hetero_result):
    res, rec = hetero_result

    # Precision weights are attached and physically sane.
    W = rec.sample_weights
    assert W is not None and W.shape[0] == rec.Phi.shape[0]
    assert np.all(np.isfinite(W)) and np.all(W > 0)

    # The recorded weights are consistent with the recorded coefficients: a fresh
    # profiled joint-GLS solve on (Phi, y, reg, W) reproduces the trainer's w.
    # Session 01: the shipped mean profiles an unpenalized intercept (Remark
    # rem:intercept), so the diagnostic instrument is the AUGMENTED design
    # [1, Phi] with penalty diag(0, R). Pass the uncentered response so f0 is
    # recovered; the nonconstant slopes match the trainer's w.
    from hifi_anova.training.fitted_design import MEAN_INTERCEPT_PROFILED_JOINT_GLS
    assert rec.mean_intercept_mode == MEAN_INTERCEPT_PROFILED_JOINT_GLS
    a = ridge_analytics(rec.Phi, rec.y_centered + rec.f0, rec.reg_diag,
                        weights=W, profile_intercept=True)
    np.testing.assert_allclose(a['w'], rec.w, rtol=1e-4, atol=1e-6)
    assert a['f0'] == pytest.approx(float(rec.f0), rel=1e-4, abs=1e-6)

    # (F5) Prediction-level invariant: the augmented analytics' fitted MEAN
    # (f0 + Phi w) matches the shipped model's mean core. The analytics re-solve
    # the penalized-GLS optimum for the record's weights; any residual gap is the
    # pre-existing within-iterate mean/weights lag (the mean is solved with the
    # iterate's entry weights, sample_weights come from the post-update variance),
    # tiny (<<1% of the response scale) and identical under the old feature-only
    # analytics — not introduced by the augmented-intercept correction.
    Phi_np = np.asarray(rec.Phi, np.float64)
    pred_ana = a['f0'] + Phi_np @ a['w']
    pred_shipped = float(rec.f0) + Phi_np @ np.asarray(rec.w, np.float64)
    yscale = float(np.std(np.asarray(rec.y_centered, np.float64)))
    assert np.max(np.abs(pred_ana - pred_shipped)) < 1e-3 * yscale

    # The API reports exactly these weighted profiled-intercept diagnostics.
    assert res.sigma_hat == pytest.approx(a['sigma_hat'], rel=1e-9)
    assert res.df == pytest.approx(a['df'], rel=1e-9)

    # loo_cv / loo_nll are the Tier-II one-step variance jackknife (M1/DEC-031),
    # NOT the plug-in Tier-I PRESS a['loo_cv'] — so they are checked against
    # joint_loo on the record's surfaced variance sub-problem, and Tier II should
    # differ from (and be more pessimistic than) the Tier-I value it corrects.
    assert res.loo_tier == 2
    j = joint_loo(a, rec.variance)
    assert res.loo_cv == pytest.approx(j['loo_cv'], rel=1e-9)
    assert res.loo_nll == pytest.approx(j['loo_nll'], rel=1e-9)
    assert res.loo_cv != pytest.approx(a['loo_cv'], rel=1e-9)   # genuinely Tier II


def test_weighting_actually_changes_diagnostics(hetero_result):
    res, rec = hetero_result

    # An unweighted refit of the same design gives materially different df/sigma —
    # proving the weighted branch is engaged, not a no-op.
    unw = ridge_analytics(rec.Phi, rec.y_centered, rec.reg_diag)
    assert abs(unw['df'] - res.df) > 1e-6
    assert not np.isclose(unw['sigma_hat'], res.sigma_hat, rtol=1e-3)


def test_calibration_semantics(hetero_result):
    res, rec = hetero_result

    # sigma_hat is a whitened calibration scale ~ O(1), flagged as such.
    assert res.noise_scale_is_calibration is True
    assert 0.2 < res.sigma_hat < 5.0

    # sigma^2(x) is genuinely input-dependent.
    X, _y, _ = generate_ishigami(n_samples=200, heteroscedastic=True, seed=99)
    s2 = res.sigma_x2(X)
    assert s2.shape[0] == X.shape[0]
    assert np.all(s2 > 0)
    assert s2.std() / s2.mean() > 0.05        # not a flat constant


def test_unit_weight_attribution_companion(hetero_result):
    res, rec = hetero_result

    # A unit-weight companion is attached and is the fit attribution reads.
    comp = rec.interpretive
    assert comp is not None
    assert comp.is_weighted is False
    assert rec.attribution_record() is comp
    # Same design, unit-weight centering: the companion's target is centered on
    # the plain mean, not the GLS-weighted intercept, so it differs from the
    # weighted record's centered target.
    assert np.array_equal(comp.Phi, rec.Phi)
    assert not np.allclose(comp.y_centered, rec.y_centered)

    # Reported Sobol CIs come from the companion and each bracket their own point.
    for name, (S, lo, hi) in res.sobol_ci.items():
        assert lo <= S <= hi
        assert 0.0 <= lo <= hi <= 1.0


def test_weighted_epistemic_intervals(hetero_result):
    res, rec = hetero_result

    from hifi_anova.model.predict import predict_intervals
    from hifi_anova.linalg import spd_inverse

    X, _y, _ = generate_ishigami(n_samples=120, heteroscedastic=True, seed=5)
    X_t = np.clip(res.transformer.transform(X), 0, 1)
    # DEC-056: the public predict path runs the float64 NumPy exact core by
    # default, so the manual reference below must also be float64 — a float32
    # reference desyncs from res.predict_intervals(X) at ~1e-6 round-off.
    x = np.asarray(X_t, dtype=np.float64)

    # df_residual mirrors the public path: intervals use Student-t with the
    # exact residual df (z undercovers when sigma is estimated).
    common = dict(Phi_train=res._Phi_train, reg_diag=res._reg_diag,
                  sigma2_hat=res.sigma_hat ** 2, df_residual=res.df_residual)
    wtd = predict_intervals(res.model, x, weights=res._sample_weights,
                            profile_intercept=True, **common)
    unw = predict_intervals(res.model, x, weights=None, **common)

    assert np.all(np.isfinite(wtd['var_epistemic']))
    assert np.all(wtd['var_epistemic'] >= 0)
    assert np.all(wtd['lower'] <= wtd['upper'])
    assert not np.allclose(wtd['var_epistemic'], unw['var_epistemic'])

    # (F3) Independent reconstruction of the AUGMENTED weighted epistemic term:
    # var_ep(x) = z_new^T A_aug^{-1} z_new with Z=[1,Phi], A_aug = Z^T W Z +
    # diag(0, R). Must equal predict_intervals(profile_intercept=True).
    Phi_tr = np.asarray(res._Phi_train, np.float64)
    reg = np.asarray(res._reg_diag, np.float64)
    W = np.asarray(res._sample_weights, np.float64)
    Ntr = Phi_tr.shape[0]
    Z_tr = np.concatenate([np.ones((Ntr, 1)), Phi_tr], axis=1)
    A_aug = Z_tr.T @ (W[:, None] * Z_tr) + np.diag(np.concatenate([[0.0], reg]))
    A_aug_inv = spd_inverse(A_aug)
    Phi_new = np.asarray(res.model.build_phi_all(x), np.float64)
    Z_new = np.concatenate([np.ones((Phi_new.shape[0], 1)), Phi_new], axis=1)
    var_ep_ref = np.sum((Z_new @ A_aug_inv) * Z_new, axis=1)
    np.testing.assert_allclose(wtd['var_epistemic'], var_ep_ref, rtol=1e-6,
                               atol=1e-9)

    # The profiled epistemic term differs from the fixed-intercept weighted one
    # (nonzero weighted feature means ⇒ the intercept carries real uncertainty).
    wtd_fixed = predict_intervals(res.model, x, weights=res._sample_weights,
                                  profile_intercept=False, **common)
    assert not np.allclose(wtd['var_epistemic'], wtd_fixed['var_epistemic'])

    # (F3) The PUBLIC path selects the augmented branch: its bounds equal the
    # profile_intercept=True replica and differ from the fixed-intercept one.
    lo, hi = res.predict_intervals(X)
    np.testing.assert_allclose(lo, np.asarray(wtd['lower']), rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(hi, np.asarray(wtd['upper']), rtol=1e-6, atol=1e-9)
    assert not np.allclose(hi - lo,
                           np.asarray(wtd_fixed['upper'] - wtd_fixed['lower']))


# --- M2: two-fit reporting surface + gap (Thm projection Part ii) -------------

def test_efficient_index_set_present_and_wellformed(hetero_result):
    res, rec = hetero_result

    # A heteroscedastic fit exposes the efficient (precision-weighted) index set
    # alongside the interpretable one, over the same named components.
    eff = res.sobol_ci_efficient
    assert eff is not None
    assert set(eff.keys()) == set(res.sobol_ci.keys())
    for name, (S, lo, hi) in eff.items():
        assert lo <= S <= hi
        assert 0.0 <= lo <= hi <= 1.0

    # The efficient set is genuinely a *different* estimator than the
    # interpretable one under heteroscedasticity (else there is no gap to report).
    interp_pts = {n: v[0] for n, v in res.sobol_ci.items()}
    eff_pts = {n: v[0] for n, v in eff.items()}
    assert any(abs(eff_pts[n] - interp_pts[n]) > 1e-9 for n in eff_pts)


def test_gap_equals_efficient_minus_interpretable(hetero_result):
    res, rec = hetero_result

    gap = res.sobol_gap
    assert gap is not None
    assert set(gap.keys()) == {'first_order', 'second_order'}

    # First-order gap is exactly efficient point − interpretable point.
    for name, g in gap['first_order'].items():
        s_eff = res.sobol_ci_efficient[name][0]
        s_int = res.sobol_ci[name][0]
        assert g == pytest.approx(s_eff - s_int, abs=1e-12)

    # Second-order gap keys are (i, j) index tuples.
    for key in gap['second_order']:
        assert isinstance(key, tuple) and len(key) == 2


def test_homoscedastic_collapses_no_efficient_no_gap():
    # A homoscedastic fit: the efficient and interpretable fits coincide, so the
    # two-fit surface collapses — no separate efficient set, no gap row. Keeps the
    # default reported surface byte-identical to the pre-M2 code.
    X, y, _ = generate_ishigami(n_samples=600, heteroscedastic=False, seed=1)
    res = hifi_anova(X, y, mode='second', K1=6, K2=3, seed=42, verbose=False)
    assert res.sobol_ci_efficient is None
    assert res.sobol_gap is None
