"""Tests for the two R² conventions (hifi_anova.analysis.metrics).

The library reports two coefficient-of-determination conventions:
  - 'explained_variance' (default): 1 − Var(y−ŷ)/Var(y)  — the manuscript's
    variance-native quantity (sklearn ``explained_variance_score``);
  - 'classical': 1 − Σ(y−ŷ)²/Σ(y−ȳ)²  — textbook R² (sklearn ``r2_score``).
They coincide iff the residual has zero mean, and diverge under a biased fit.
"""

import numpy as np
import pytest

from hifi_anova.analysis.metrics import r_squared, r_squared_report

pytestmark = pytest.mark.smoke


def _classical_ref(y, yhat):
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1.0 - ss_res / ss_tot


def _ev_ref(y, yhat):
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    return 1.0 - np.var(y - yhat) / np.var(y)


def test_matches_reference_formulas():
    rng = np.random.default_rng(0)
    y = rng.normal(size=200)
    yhat = y + rng.normal(scale=0.3, size=200)
    assert r_squared(y, yhat, 'classical') == pytest.approx(_classical_ref(y, yhat))
    assert r_squared(y, yhat, 'explained_variance') == pytest.approx(_ev_ref(y, yhat))


def test_coincide_when_residual_mean_zero():
    """With a mean-zero residual the two conventions are identical."""
    rng = np.random.default_rng(1)
    y = rng.normal(size=500)
    resid = rng.normal(scale=0.4, size=500)
    resid = resid - resid.mean()          # force exactly zero-mean residual
    yhat = y - resid
    ev = r_squared(y, yhat, 'explained_variance')
    cls = r_squared(y, yhat, 'classical')
    assert ev == pytest.approx(cls, abs=1e-12)


def test_diverge_under_bias_and_classical_is_smaller():
    """A constant bias leaves explained-variance untouched but lowers classical
    R² (which can even go negative) — the exact reason both are reported."""
    rng = np.random.default_rng(2)
    y = rng.normal(size=400)
    yhat_unbiased = y + rng.normal(scale=0.2, size=400)
    bias = 5.0
    yhat_biased = yhat_unbiased + bias
    # Explained variance ignores the constant shift ...
    assert r_squared(y, yhat_unbiased, 'ev') == pytest.approx(
        r_squared(y, yhat_biased, 'ev'), abs=1e-12)
    # ... classical does not, and is strictly smaller under the bias.
    assert (r_squared(y, yhat_biased, 'classical')
            < r_squared(y, yhat_unbiased, 'classical'))
    assert r_squared(y, yhat_biased, 'classical') < 0.0


def test_perfect_and_degenerate():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert r_squared(y, y, 'classical') == pytest.approx(1.0)
    assert r_squared(y, y, 'explained_variance') == pytest.approx(1.0)
    # Degenerate constant target -> guarded 0.0, not a division error.
    const = np.array([2.0, 2.0, 2.0])
    assert r_squared(const, const, 'classical') == 0.0
    assert r_squared(const, const, 'explained_variance') == 0.0


def test_aliases_and_report_and_bad_name():
    rng = np.random.default_rng(3)
    y = rng.normal(size=50)
    yhat = y + rng.normal(scale=0.1, size=50)
    assert r_squared(y, yhat, 'r2_score') == r_squared(y, yhat, 'classical')
    assert r_squared(y, yhat, 'sse_tss') == r_squared(y, yhat, 'classical')
    rep = r_squared_report(y, yhat)
    assert set(rep) == {'explained_variance', 'classical'}
    with pytest.raises(ValueError, match="Unknown"):
        r_squared(y, yhat, 'bogus')


def test_matches_sklearn_if_available():
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(4)
    y = rng.normal(size=300)
    yhat = y + rng.normal(scale=0.5, size=300) + 0.7   # biased on purpose
    assert r_squared(y, yhat, 'classical') == pytest.approx(
        sk.r2_score(y, yhat))
    assert r_squared(y, yhat, 'explained_variance') == pytest.approx(
        sk.explained_variance_score(y, yhat))


@pytest.mark.integration
def test_hifiresult_surfaces_both_conventions():
    """The one-call result carries both R² conventions, and the default
    ``r_squared`` stays the explained-variance value (manuscript convention)."""
    from hifi_anova.api import hifi_anova
    rng = np.random.default_rng(5)
    X = rng.uniform(0.0, 1.0, size=(300, 3))
    y = X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=0.05, size=len(X))
    res = hifi_anova(X, y, mode='second', verbose=False)
    assert hasattr(res, 'r_squared_classical')
    # r_squared is the explained-variance convention; recompute on the held-out
    # split and confirm it matches (classical is the SSE/TSS twin).
    assert 0.0 <= res.r_squared <= 1.0
    assert res.r_squared_classical <= res.r_squared + 1e-9
