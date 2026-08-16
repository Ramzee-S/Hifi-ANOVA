"""Phase-1 invariant: the fitted-design record reproduces the API diagnostics.

The one-call ``hifi_anova`` still computes its ``sigma_hat`` / ``df`` / ``loo_cv``
from the legacy rebuild (Phase 2 has not rewired it). The trainer now also stashes
a :class:`FittedDesign` in ``results['fitted_design']``. For the homoscedastic,
≤2nd-order single-basis path — where the rebuild and the real fit coincide —
``ridge_analytics`` on the record must reproduce the reported diagnostics
bit-for-bit. This is the safety net that keeps the golden master frozen while
Phase 2 swaps the API over to the record.

(Third-order and heteroscedastic fits deliberately diverge: the record carries
the real per-order penalty / precision weights the legacy rebuild got wrong — the
whole point of the refactor — so they are exercised elsewhere, not here.)
"""

import numpy as np
import pytest

from hifi_anova.api import hifi_anova, _legacy_analytics_rebuild
from hifi_anova.analysis.automl import ridge_analytics
from hifi_anova.data.synthetic import generate_friedman1, generate_ishigami
from hifi_anova.training.fitted_design import FittedDesign

pytestmark = pytest.mark.integration

# Tight: the record and the rebuild are the *same* unweighted ridge on the same
# float64 design, so the only slack is float32→float64 feature ULPs and the
# jnp.mean/np.mean intercept difference — both far under 1e-7.
RTOL, ATOL = 1e-7, 1e-10


def _fit(**kw):
    X, y = generate_friedman1(n_samples=600, noise_std=1.0,
                              n_irrelevant=2, seed=1)
    res = hifi_anova(X, y, seed=42, verbose=False, **kw)
    rec = res.train_results['fitted_design']
    assert isinstance(rec, FittedDesign)
    return res, rec


@pytest.mark.parametrize(('kw', 'expect_order2'), [
    (dict(mode='first', K1=6, basis_name='fourier'), False),
    (dict(mode='second', K1=6, K2=3, basis_name='fourier'), True),
    (dict(mode='first', K1=5, basis_name='legendre'), False),
    (dict(mode='first', K1=6, basis_name='haar'), False),
    (dict(mode='second', K1=5, K2=3, basis_name='legendre'), True),
])
def test_record_reproduces_api_diagnostics(kw, expect_order2):
    res, rec = _fit(**kw)

    # Record structure sanity.
    assert rec.sample_weights is None            # homoscedastic
    assert rec.is_weighted is False
    assert rec.block(1) is not None
    assert (rec.block(2) is not None) == expect_order2
    assert rec.Phi.dtype == np.float64

    a = ridge_analytics(rec.Phi, rec.y_centered, rec.reg_diag)

    assert a['sigma_hat'] == pytest.approx(res.sigma_hat, rel=RTOL, abs=ATOL)
    assert a['df'] == pytest.approx(res.df, rel=RTOL, abs=ATOL)
    assert a['loo_cv'] == pytest.approx(res.loo_cv, rel=RTOL, abs=ATOL)


def test_record_reg_diag_matches_api_for_second_order():
    """For ≤2nd-order the record's penalty equals the API's rebuilt one."""
    res, rec = _fit(mode='second', K1=6, K2=3, basis_name='fourier')
    assert rec.reg_diag.shape[0] == rec.Phi.shape[1]
    # The API stashes the reg diagonal it used on the result.
    np.testing.assert_allclose(rec.reg_diag, res._reg_diag, rtol=1e-12, atol=0)


def test_third_order_record_fixes_legacy_rebuild():
    """K3 fit: the record diverges from the (buggy) legacy rebuild — the fix.

    The legacy rebuild padded the third-order penalty with lambda_order2 and
    dropped the third-order variance from the Sobol denominator. The record uses
    the real lambda_order3 and includes K3/T/G3, so its diagnostics differ — and
    the API reports the record's.
    """
    Xi, yi, _ = generate_ishigami(n_samples=1500, noise_std=0.1, seed=0)
    res = hifi_anova(Xi, yi, mode='second', K1=8, K2=4, K3=2,
                     lambda_order3=0.1, triple_selection='all',
                     seed=42, verbose=False)
    rec = res.train_results['fitted_design']
    b3 = rec.block(3)
    assert b3 is not None and b3.n_groups >= 1          # (0,1,2) triple retained

    a_rec = ridge_analytics(rec.Phi, rec.y_centered, rec.reg_diag)
    _phi, _reg, a_leg, _ci = _legacy_analytics_rebuild(
        res.model, res._data, res.config, res.model.D, 8, 4,
        res.config['strategy'])

    # Real lambda_order3 (0.1) vs the legacy lambda_order2 (0.01) padding on the
    # 125-column third-order block => materially different effective df.
    assert not np.isclose(a_rec['df'], a_leg['df'], rtol=1e-3)
    # The API reports the record's (fixed) diagnostics, not the legacy ones.
    assert res.sigma_hat == pytest.approx(a_rec['sigma_hat'], rel=1e-9)
    assert res.df == pytest.approx(a_rec['df'], rel=1e-9)
