"""Stage-D Option B (joint-GLS mean) — flag-matrix contract + df/σ̂ identity.

These pin the invariants the advisor required before the default flip. They run
with the flags explicitly set, so they are valid whatever the defaults are (i.e.
they keep asserting the contract after a future default flip). Marked
``integration`` (several heteroscedastic fits).
"""
import warnings
import numpy as np
import pytest

from hifi_anova.data.test_functions import T3_2_shared_variable
from hifi_anova.data.preprocessing import preprocess_data
from hifi_anova.training.trainer import HiFiANOVATrainer
from hifi_anova.training.fitted_design import (
    MEAN_INTERCEPT_PROFILED_JOINT_GLS, MEAN_INTERCEPT_LEGACY_FIXED,
    MEAN_INTERCEPT_UNWEIGHTED)

_BASE = dict(K1=5, K2=3, Kh=3, strategy='curvature', lambda_order1=0.001,
             lambda_order2=0.005, lambda_h=0.01, stages=['A', 'B', 'D'],
             residual_nn={'enabled': False}, max_outer_iter=10,
             alternating_tol=1e-4, newton_max_iter=15)


@pytest.fixture(scope='module')
def tier3_data():
    X, y, _ = T3_2_shared_variable(n_samples=15000, seed=42)
    return preprocess_data(X, y, seed=42)


def _fit(data, **cfg):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return HiFiANOVATrainer(dict(_BASE, **cfg)).fit(
            data['x_train'], data['y_train'], data['x_val'], data['y_val'])


# A first-order heteroscedastic fixture whose GLS-weighted (joint-GLS) mean is
# measurably different from the pre-Stage-D unweighted mean: the noise variance
# grows steeply with x0, so the precision weights reshape the mean fit (and hence
# the Sobol shares) relative to the unweighted Stage-A mean. First-order only
# (K2=0) so the reconstruction denominator is a clean sum of first-order variances.
_FO_BASE = dict(K1=6, K2=0, strategy='variance', max_outer_iter=8)


@pytest.fixture(scope='module')
def hetero_first_order_data():
    rng = np.random.default_rng(7)
    N, D = 3000, 3
    X = rng.uniform(0, 1, (N, D))
    mean = (1.4 * np.sin(2 * np.pi * X[:, 0]) + 1.0 * (X[:, 1] - 0.5)
            + 0.9 * np.cos(2 * np.pi * X[:, 2]))
    sigma = 0.03 + 3.0 * X[:, 0] ** 3        # steep, correlated with x0
    y = mean + sigma * rng.standard_normal(N)
    return preprocess_data(X, y, seed=7)


def _fit_fo(data, **cfg):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return HiFiANOVATrainer(dict(_FO_BASE, **cfg)).fit(
            data['x_train'], data['y_train'], data['x_val'], data['y_val'])


def _first_order_shares(model):
    """Reconstruct first-order Sobol shares from a model's mean coefficients."""
    from hifi_anova.analysis.sobol import _mean_component_variances
    fo, so, to = _mean_component_variances(model)
    tot = sum(fo.values()) + sum(so.values()) + sum(to.values())
    return {i: fo[i] / tot for i in fo}, fo


@pytest.mark.integration
def test_two_flag_matrix(tier3_data):
    """The four {joint_gls_mean} x {mean_fallback} combos on genuine hetero data.

    Legacy mean (joint off): the guard false-reverts unless X4B's mean-fallback
    routes around it. Joint-GLS (joint on): the guard keeps the variance model
    unaided, and X4B's mean-fallback MUST NOT fire (the invariant monitor).

    The two joint-off cases pin variance_selection_mean_consistent=False (the
    pre-flip package comparison) so they reproduce the ORIGINAL false-revert
    regardless of the now-flipped default; the joint-on cases use the shipped
    defaults (mean-consistent guard on).
    """
    # joint off + legacy package comparison -> false-revert (documents the bug)
    _, r = _fit(tier3_data, stage_d_joint_gls_mean=False,
                variance_selection_mean_consistent=False,
                variance_selection_mean_fallback=False)
    assert r['stage_D']['selected'] == 'homoscedastic'

    # joint off + legacy comparison + fallback -> ships unit-weight mean + var (X4B)
    m, r = _fit(tier3_data, stage_d_joint_gls_mean=False,
                variance_selection_mean_consistent=False,
                variance_selection_mean_fallback=True)
    assert r['stage_D']['selected'] == 'mean_fallback'
    assert m.variance_model is not None

    # joint on (shipped defaults) -> guard keeps the variance model unaided
    m, r = _fit(tier3_data, stage_d_joint_gls_mean=True,
                variance_selection_mean_fallback=False)
    assert r['stage_D']['selected'] == 'heteroscedastic'
    assert m.variance_model is not None

    # joint on, fallback on -> still 'heteroscedastic'; mean-fallback must NOT
    # fire (no anomaly): a correct joint-GLS mean cannot lose to the unit mean.
    m, r = _fit(tier3_data, stage_d_joint_gls_mean=True,
                variance_selection_mean_fallback=True)
    assert r['stage_D']['selected'] == 'heteroscedastic'
    assert m.variance_model is not None
    assert not r['stage_D'].get('mean_fallback_anomaly', False)


@pytest.mark.integration
def test_joint_gls_analytics_profile_the_intercept(tier3_data):
    """The record's Stage-D analytics profile the SAME intercept as the shipped fit.

    Under joint-GLS the mean update profiles an unpenalized intercept (Remark
    rem:intercept: augmented design Z=[1,Φ], penalty diag(0,R)). Session 01
    corrected the downstream analytics to use that augmented smoother, so the
    reported df now INCLUDES the intercept coordinate and equals the independent
    augmented df (previously it omitted ~1). This replaces the old "df omits +1"
    pin, which locked the *wrong* feature-only convention.
    """
    from hifi_anova.analysis.automl import (ridge_analytics,
                                            _ridge_analytics_weighted)
    from hifi_anova.linalg import spd_inverse

    m, r = _fit(tier3_data, stage_d_joint_gls_mean=True)
    assert r['stage_D']['selected'] == 'heteroscedastic'
    rec = r['fitted_design']

    # The profiled analytics run on the UNCENTERED response so f0 is recovered.
    y_unc = np.asarray(rec.y_centered, np.float64) + float(rec.f0)
    ana = ridge_analytics(rec.Phi, y_unc, rec.reg_diag,
                          weights=rec.sample_weights, profile_intercept=True)

    # Independent augmented df (the correct instrument).
    Phi = np.asarray(rec.Phi, np.float64)
    W = np.asarray(rec.sample_weights, np.float64)
    reg = np.asarray(rec.reg_diag, np.float64)
    Phi_aug = np.concatenate([np.ones((Phi.shape[0], 1)), Phi], axis=1)
    reg_aug = np.concatenate([[0.0], reg])
    A_aug = spd_inverse(Phi_aug.T @ (W[:, None] * Phi_aug) + np.diag(reg_aug))
    df_aug = float(np.trace(A_aug @ (Phi_aug.T @ (W[:, None] * Phi_aug))))

    # (1) The reported analytics df now EQUALS the augmented df (no +1 omission).
    assert ana['df'] == pytest.approx(df_aug, abs=1e-6)
    # (2) Leverages sum to df, intercept included.
    assert float(ana['leverages'].sum()) == pytest.approx(ana['df'], abs=1e-6)
    # (3) The old feature-only convention omitted exactly the intercept df (~1),
    #     confirming the correction is the intercept coordinate and nothing else.
    feat = _ridge_analytics_weighted(rec.Phi, rec.y_centered,
                                     rec.reg_diag, rec.sample_weights)
    assert ana['df'] - feat['df'] == pytest.approx(1.0, abs=0.05)
    # (4) The augmented analytics reproduce the record's profiled-GLS optimum
    #     (its own Φ/reg/weights) at machine precision — the shipped instrument.
    theta = A_aug @ (Phi_aug.T @ (W * y_unc))
    assert ana['f0'] == pytest.approx(float(theta[0]), abs=1e-8)
    np.testing.assert_allclose(ana['w'], theta[1:], atol=1e-8)


@pytest.mark.integration
def test_estimator_convention_recorded(hetero_first_order_data):
    """The EFFECTIVE mean-estimator convention is recorded (DEC-039 provenance).

    A downstream consumer / saved artifact must be able to tell which estimator
    vintage produced the fitted mean. The tag reflects the resolved flag and the
    shipped mean, not merely the request: the joint-GLS default records
    ``profiled_joint_gls``; the (non-default, legacy) fixed-intercept flag records
    ``legacy_fixed_intercept_uncentered_features``; a homoscedastic fit records
    the ordinary unit-weight ``unweighted_centered``.
    """
    data = hetero_first_order_data

    # Default (joint-GLS): kept heteroscedastic, profiled-joint-GLS convention on
    # BOTH the training-results dict and the fitted-design record.
    m_d, r_d = _fit_fo(data, stages=['A', 'D'], Kh=3)
    assert r_d['stage_D']['selected'] == 'heteroscedastic'
    assert (r_d['stage_D']['mean_intercept_mode']
            == MEAN_INTERCEPT_PROFILED_JOINT_GLS)
    assert (r_d['fitted_design'].mean_intercept_mode
            == MEAN_INTERCEPT_PROFILED_JOINT_GLS)

    # Legacy compatibility flag (non-default): force-keep so the legacy weighted
    # mean actually ships, and pin that its provenance tag is the legacy vintage.
    # This exercises the flag's availability + metadata WITHOUT promoting it (the
    # guard-off is scoped to exposing the legacy estimator's record convention).
    m_l, r_l = _fit_fo(data, stages=['A', 'D'], Kh=3,
                       stage_d_joint_gls_mean=False, heteroscedastic_guard=False)
    assert (r_l['fitted_design'].mean_intercept_mode
            == MEAN_INTERCEPT_LEGACY_FIXED)
    assert r_l['stage_D']['mean_intercept_mode'] == MEAN_INTERCEPT_LEGACY_FIXED

    # Homoscedastic (no Stage D): ordinary unit-weight centered mean.
    m_a, r_a = _fit_fo(data, stages=['A'])
    assert 'stage_D' not in r_a
    assert r_a['fitted_design'].mean_intercept_mode == MEAN_INTERCEPT_UNWEIGHTED


@pytest.mark.integration
def test_sobol_provenance_final_joint_gls_mean(hetero_first_order_data):
    """Positive control: the reported structural Sobol spectrum is recomputed from
    the FINAL joint-GLS Stage-D mean, not a stale pre-Stage-D mean (DEC-039).

    The one-call API assembles ``result.sobol`` from ``compute_sobol_indices`` on
    the fitted (predictive) model, so ``result.sobol['mean_sobol']`` is the
    structural spectrum of the shipped joint-GLS mean — distinct from the
    interpretable unit-weight ``sobol_ci`` headline (two-fit convention, DEC-030;
    this test targets the GLS structural field, the only one that reflects the
    Stage-D mean flip). We prove:
      (1) it equals an independent reconstruction from the SHIPPED Stage-D mean
          coefficients + fitted design (agreement), and
      (2) it disagrees, by a safe margin, with the same reconstruction from a
          deliberately stale pre-Stage-D (Stage-A) mean (it is NOT the old mean),
    then (3) assert the same through the actual one-call ``hifi_anova`` API so an
    API-assembly regression can't leave this green. The fixture is chosen so the
    two means genuinely differ.
    """
    from hifi_anova.analysis.sobol import compute_sobol_indices
    data = hetero_first_order_data

    m_d, r_d = _fit_fo(data, stages=['A', 'D'], Kh=3)      # final joint-GLS mean
    # Selection is a precondition of the whole control — assert it first.
    assert r_d['stage_D']['selected'] == 'heteroscedastic'
    assert (r_d['stage_D']['mean_intercept_mode']
            == MEAN_INTERCEPT_PROFILED_JOINT_GLS)

    m_a, _ = _fit_fo(data, stages=['A'])                   # stale pre-D mean

    # Fixture premise: the final GLS-weighted mean coefficients genuinely differ
    # from the pre-D unweighted ones (else there is nothing to attribute wrongly).
    w1_final = np.asarray(m_d.mean_model.w1, dtype=np.float64)
    w1_preD = np.asarray(m_a.mean_model.w1, dtype=np.float64)
    assert np.max(np.abs(w1_final - w1_preD)) > 0.05

    shares_final, _ = _first_order_shares(m_d)
    shares_preD, _ = _first_order_shares(m_a)

    reported = compute_sobol_indices(m_d, data['x_test'])['mean_sobol']['first_order']

    # (1) Agreement: the reported spectrum is the final-mean reconstruction.
    for i in reported:
        assert reported[i] == pytest.approx(shares_final[i], abs=1e-9)

    # (2) Disagreement with the stale pre-D reconstruction, by a safe margin.
    max_stale_gap = max(abs(reported[i] - shares_preD[i]) for i in reported)
    assert max_stale_gap > 6e-3, (
        f"reported Sobol shares are indistinguishable from the pre-Stage-D mean "
        f"(max gap {max_stale_gap:.2e}); the provenance control cannot "
        f"discriminate — strengthen the fixture's heteroscedasticity.")

    # (3) Exercise the ACTUAL one-call API (steps 1–2 drive the trainer directly).
    # ``result.sobol`` must be the structural spectrum of the SHIPPED (joint-GLS)
    # model, so an api.py regression that assembled it from a different fit would
    # break this — the earlier steps would stay green on their own.
    from hifi_anova import hifi_anova
    rng = np.random.default_rng(7)
    Xr = rng.uniform(0, 1, (3000, 3))
    yr = (1.4 * np.sin(2 * np.pi * Xr[:, 0]) + 1.0 * (Xr[:, 1] - 0.5)
          + 0.9 * np.cos(2 * np.pi * Xr[:, 2])
          + (0.03 + 3.0 * Xr[:, 0] ** 3) * rng.standard_normal(3000))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = hifi_anova(Xr, yr, K1=6, K2=0, heteroscedastic=True,
                         mode='heteroscedastic', seed=42, verbose=False)
    assert res.train_results['stage_D']['selected'] == 'heteroscedastic'
    assert (res.train_results['stage_D']['mean_intercept_mode']
            == MEAN_INTERCEPT_PROFILED_JOINT_GLS)
    api_first = res.sobol['mean_sobol']['first_order']
    recon_first, _ = _first_order_shares(res.model)
    for i in api_first:
        assert api_first[i] == pytest.approx(recon_first[i], abs=1e-9)
