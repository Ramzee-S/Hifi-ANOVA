"""Order-selective variable membership (X11C-S03 / GUI3 request BR-06).

``variable_orders={j: [orders]}`` picks, per variable, which interaction
orders it participates in — a non-empty subset of ``{1, 2}``:

* ``[2]`` admits variable ``j`` to PAIR terms only. Its first-order block is
  EXCLUDED from every mean design (no df spent); the model keeps the uniform
  ``w1`` layout with EXACT zeros in ``j``'s block, so ``j``'s first-order
  Sobol share is identically 0 and prediction/slicing work unchanged. This is
  a NON-HIERARCHICAL model (the pair terms absorb any true marginal along
  ``j``) — recorded as an honesty note in ``results['term_structure']``.
* ``[1]`` keeps ``j``'s marginal but drops every pair touching it.
* ``[1, 2]`` is the default full membership → a no-op.

Because a variable whose first-order block is excluded is invisible to any
data-driven pair search, ``variable_orders`` first-order exclusions compose
only with a data-INDEPENDENT pair set (a per-pair ``K2`` mapping,
``pair_selection=None/'all'``, or an explicit ``pair_selection`` list). These
tests cover the two entry paths (K2 mapping and pair_selection list), the
Stage-D interaction (guard-revert + mean-fallback with the ragged first-order
design), persistence, and the guards.

The all-orders default path is untouched by construction (every branch keys on
``variable_orders is not None``); the golden characterization pins that
separately.
"""

import warnings

import numpy as np
import pytest

from hifi_anova.api import hifi_anova
from hifi_anova.core.features import basis_size

pytestmark = pytest.mark.integration


def _data(N=400, D=4, seed=0, noise=0.1):
    """x2 (index 2) enters ONLY through a pure interaction with x3 — no
    marginal — so an order-2-only membership for x2 is the correct model."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(size=(N, D))
    y = (np.sin(2 * np.pi * X[:, 0]) + X[:, 1] ** 2
         + 1.2 * np.sin(2 * np.pi * X[:, 2]) * np.sin(2 * np.pi * X[:, 3])
         + noise * rng.standard_normal(N))
    return X, y


def _fit(X, y, **kw):
    # variable_orders needs a data-independent pair set; the one-call API's
    # implicit variable_selection='bic' must be neutralized explicitly.
    kw.setdefault('variable_selection', None)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return hifi_anova(X, y, verbose=False, **kw)


# ---------------------------------------------------------- order-2-only var

def test_order2_only_via_k2_mapping():
    """``variable_orders={2: [2]}`` with a K2 mapping: x2's first-order share
    is identically 0, its first-order columns are absent from the record's
    design (no df spent), its w1 block is exact zero, and the record has no
    order-1 Sobol group for x2 — while the (2, 3) pair recovers the planted
    pure interaction."""
    X, y = _data()
    res = _fit(X, y, K1=4, K2={(2, 3): 4, (0, 1): 2}, variable_orders={2: [2]})
    m = res.model
    b1 = basis_size(4, True, 'fourier')

    # first-order share of the order-2-only variable is exactly zero
    fo = res.sobol['mean_sobol']['first_order']
    assert fo[2] == 0.0
    # ...and its uniform-layout w1 block is exact zeros
    np.testing.assert_array_equal(
        np.asarray(m.mean_model.w1)[2 * b1:3 * b1], 0.0)

    # the planted (2, 3) interaction is recovered
    so = res.sobol['mean_sobol']['second_order']
    assert set(so) == {(0, 1), (2, 3)}
    assert so[(2, 3)] > 0.1

    # the fitted-design record spent NO df on x2's marginal: the first-order
    # groups skip x2, and Phi's first-order block spans only the 3 kept vars.
    rec = res._fitted_design
    order1_groups = [g[1] for g in rec.sobol_groups if g[0] == 1]
    assert order1_groups == [0, 1, 3]
    F1 = 3 * b1
    assert rec.Phi.shape[1] >= F1
    # the order-1 group slices tile exactly [0, F1) — no column for x2
    covered = [g[2] for g in rec.sobol_groups if g[0] == 1]
    assert covered[0].start == 0 and covered[-1].stop == F1

    # provenance carries the non-hierarchical honesty note
    ts = res.train_results['term_structure']
    assert ts['first_order_excluded'] == [2]
    assert 'non-hierarchical' in ts['note']

    # diagnostics + prediction paths finite on the ragged layout
    assert np.isfinite(res.df) and np.isfinite(res.loo_cv)
    assert np.all(np.isfinite(res.predict(X[:16])))
    lo, hi = res.predict_intervals(X[:5])
    assert np.all(np.asarray(hi) >= np.asarray(lo))
    for _name, (S, clo, chi) in res.sobol_ci.items():
        assert clo <= S <= chi


def test_order2_only_via_pair_selection_list():
    """The same order-2-only exclusion through an explicit ``pair_selection``
    list (active-variable indices) rather than a K2 mapping: x2's first-order
    share is still identically 0 and it still participates in pairs."""
    X, y = _data(seed=1)
    res = _fit(X, y, K1=4, K2=4, pair_selection=[0, 1, 2, 3],
               variable_orders={2: [2]})
    fo = res.sobol['mean_sobol']['first_order']
    assert fo[2] == 0.0
    so = res.sobol['mean_sobol']['second_order']
    # x2 still appears in pair terms; the planted (2, 3) pair is strong
    assert any(2 in p for p in so)
    assert so.get((2, 3), 0.0) > 0.1
    assert np.all(np.isfinite(res.predict(X[:8])))


def test_order1_only_drops_pairs_touching_the_variable():
    """``[1]`` keeps x2's marginal but removes every pair that touches it."""
    X, y = _data(seed=2)
    res = _fit(X, y, K1=4, K2=4, pair_selection=[0, 1, 2, 3],
               variable_orders={2: [1]})
    so = set(res.sobol['mean_sobol']['second_order'])
    assert so  # some pairs survive
    assert all(2 not in p for p in so)
    # x2's marginal is retained (still a first-order key)
    assert 2 in res.sobol['mean_sobol']['first_order']
    ts = res.train_results['term_structure']
    assert ts['pair_excluded'] == [2]


def test_full_membership_is_a_noop():
    """``[1, 2]`` is the default membership: identical predictions to the same
    fit without any ``variable_orders``."""
    X, y = _data(seed=3)
    ref = _fit(X, y, K1=4, K2={(2, 3): 4, (0, 1): 2})
    same = _fit(X, y, K1=4, K2={(2, 3): 4, (0, 1): 2},
                variable_orders={2: [1, 2]})
    xg = np.asarray(ref._data['x_test'])[:32]
    np.testing.assert_allclose(
        np.asarray(ref.model.predict_mean_only(xg)),
        np.asarray(same.model.predict_mean_only(xg)), rtol=0, atol=0)
    # no term_structure provenance is recorded for a pure no-op membership...
    # (a K2 mapping still records its own pair_k2 provenance)
    assert same.train_results['term_structure']['first_order_excluded'] == []
    assert same.train_results['term_structure']['pair_excluded'] == []


# --------------------------------------------------------------- Stage D

def test_order2_only_heteroscedastic_stage_d():
    """Stage-D fit with an order-2-only variable: the ragged first-order design
    flows through the variance solve and the final mean-predict; σ²>0, x2's
    first-order share stays 0, predictions finite."""
    rng = np.random.default_rng(4)
    N, D = 500, 4
    X = rng.uniform(size=(N, D))
    sigma = 0.05 + 0.4 * X[:, 0]
    y = (np.sin(2 * np.pi * X[:, 0])
         + 1.5 * np.sin(2 * np.pi * X[:, 2]) * np.sin(2 * np.pi * X[:, 3])
         + sigma * rng.standard_normal(N))
    res = _fit(X, y, K1=4, K2={(2, 3): 4}, variable_orders={2: [2]},
               heteroscedastic=True, mode='heteroscedastic')
    assert res.sobol['mean_sobol']['first_order'][2] == 0.0
    assert (2, 3) in res.sobol['mean_sobol']['second_order']
    assert np.all(np.asarray(res.sigma_x2(X[:10])) > 0)
    assert np.all(np.isfinite(res.predict(X[:10])))
    # the weighted epistemic posterior needs Phi_new column-consistent with the
    # subset record.Phi (regression: the uniform build_phi_all mismatched it).
    lo, hi = res.predict_intervals(X[:5])
    assert np.all(np.isfinite(lo)) and np.all(np.asarray(hi) >= np.asarray(lo))


def test_stage_d_guard_revert_with_fo_included():
    """Homoscedastic data forces the Stage-D guard to REVERT to constant
    variance; the revert path re-predicts the mean via the ragged design
    (``_mean_predict_on_designs`` with ``fo_included``) and ships a homoscedastic
    fallback model that must still carry the order-selective layout so its
    epistemic intervals stay column-consistent. Exercised for both the default
    guard and the mean-fallback flag."""
    rng = np.random.default_rng(3)
    N, D = 400, 4
    X = rng.uniform(size=(N, D))
    # truly homoscedastic noise ⇒ no variance model is warranted ⇒ revert
    y = (np.sin(2 * np.pi * X[:, 0])
         + 1.5 * np.sin(2 * np.pi * X[:, 2]) * np.sin(2 * np.pi * X[:, 3])
         + 0.15 * rng.standard_normal(N))
    for mean_fallback in (False, True):
        res = _fit(X, y, K1=4, K2={(2, 3): 4}, variable_orders={2: [2]},
                   heteroscedastic=True, mode='heteroscedastic',
                   variance_selection_mean_fallback=mean_fallback)
        sd = res.train_results['stage_D']
        assert sd.get('reverted') is True
        assert sd.get('selected') == 'homoscedastic'
        # the excluded marginal survives the revert re-prediction
        assert res.sobol['mean_sobol']['first_order'][2] == 0.0
        assert np.all(np.isfinite(res.predict(X[:8])))
        lo, hi = res.predict_intervals(X[:5])
        assert np.all(np.isfinite(lo)) and np.all(np.asarray(hi) >= np.asarray(lo))


# --------------------------------------------------------------- guards

def test_order2_only_with_stage_c_residual():
    """A residual (Stage C) can sit on top of an order-selective mean: the fit
    runs, the residual attaches, and prediction + intervals stay finite (the
    residual columns are penalty-only, outside the Sobol/record blocks)."""
    X, y = _data(seed=7)
    res = _fit(X, y, K1=4, K2={(2, 3): 3}, variable_orders={2: [2]},
               stages=['A', 'B', 'C'], residual='rbf')
    assert res.model.residual_net is not None
    assert res.sobol['mean_sobol']['first_order'][2] == 0.0
    assert np.all(np.isfinite(res.predict(X[:8])))
    lo, hi = res.predict_intervals(X[:5])
    assert np.all(np.isfinite(lo)) and np.all(np.asarray(hi) >= np.asarray(lo))


def test_variable_orders_guards():
    X, y = _data(N=160)
    # a first-order exclusion needs a data-independent pair set
    with pytest.raises(ValueError, match="data-independent pair set"):
        _fit(X, y, K1=3, K2=3, variable_orders={2: [2]},
             pair_selection='both')
    with pytest.raises(ValueError, match="data-independent pair set"):
        hifi_anova(X, y, K1=3, K2=3, variable_orders={2: [2]},
                   variable_selection='bic', verbose=False)
    # first_order_pruning cannot select over a user-specified first-order set
    with pytest.raises(ValueError, match="first_order_pruning"):
        _fit(X, y, K1=3, K2={(2, 3): 3}, variable_orders={2: [2]},
             first_order_pruning='bic')
    # no third-order / mixed / auto composition
    with pytest.raises(ValueError, match="third-order"):
        _fit(X, y, K1=3, K2={(0, 1): 3}, variable_orders={2: [2]}, K3=1)
    with pytest.raises(ValueError, match="mixed"):
        _fit(X, y, K1=3, K2={(0, 1): 3}, variable_orders={0: [2]},
             basis_per_variable={0: {'K': 3}})
    with pytest.raises(ValueError, match="auto"):
        _fit(X, y, K1=3, variable_orders={2: [2]}, mode='auto')
    # a K2 mapping naming a pair whose variable is order-2-EXCLUDED conflicts
    with pytest.raises(ValueError, match="excludes from order 2"):
        _fit(X, y, K1=3, K2={(2, 3): 3}, variable_orders={2: [1]})


def test_variable_orders_validation_errors():
    X, y = _data(N=120)
    with pytest.raises(ValueError, match="out of range"):
        _fit(X, y, K1=3, K2={(0, 1): 3}, variable_orders={9: [2]})
    with pytest.raises(ValueError, match="unsupported order"):
        _fit(X, y, K1=3, K2={(0, 1): 3}, variable_orders={2: [3]})
    # BR-12: an empty order list is mean-EXCLUDED membership — accepted with a
    # UserWarning (an intercept-only / complement-only base), no longer a
    # ValueError. Call hifi_anova directly since the _fit helper suppresses
    # warnings; the fitted mean-excluded behavior is covered elsewhere.
    with pytest.warns(UserWarning, match="empty order set"):
        hifi_anova(X, y, K1=3, K2={(0, 1): 3}, variable_orders={2: []},
                   verbose=False)
    with pytest.raises(ValueError, match="mapping"):
        _fit(X, y, K1=3, K2={(0, 1): 3}, variable_orders=[2])


def test_summary_surfaces_term_structure():
    """summary() prints the equation system and the non-hierarchical caveat so
    the honesty note is user-visible, not only in results['term_structure']."""
    import contextlib
    import io
    X, y = _data(N=300)
    res = _fit(X, y, K1=4, K2={(2, 3): 3}, variable_orders={2: [2]})
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res.summary()
    out = buf.getvalue()
    assert 'per-pair K2' in out
    assert 'NON-HIERARCHICAL' in out
    assert 'excluded for variable(s) [2]' in out


def test_redecompose_rejects_term_structure():
    """The Fourier/NN re-decomposition slices the mean vector with uniform
    block widths, which would mis-align a ragged term-structure w2 — it must
    reject such a model rather than silently corrupt it."""
    from hifi_anova.training.redecompose import alternating_ridge_nn
    X, y = _data(N=200)
    res = _fit(X, y, K1=4, K2={(2, 3): 3})
    with pytest.raises(NotImplementedError, match="term structure"):
        alternating_ridge_nn(res.model, X, y, X, y, np.ones(4))
    # an order-selective (fo_included) model is rejected too
    res2 = _fit(X, y, K1=4, K2={(2, 3): 3}, variable_orders={2: [2]})
    with pytest.raises(NotImplementedError, match="term structure"):
        alternating_ridge_nn(res2.model, X, y, X, y, np.ones(4))


def test_order2_only_orphan_warns():
    """An order-2-only variable that ends up in NO retained pair is absent
    from the model entirely — the trainer warns."""
    X, y = _data(N=200)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        hifi_anova(X, y, K1=4, K2={(0, 1): 3}, variable_orders={2: [2]},
                   variable_selection=None, verbose=False)
    assert any('absent from the model entirely' in str(w.message)
               for w in caught)


# --------------------------------------------------------------- persistence

def test_order2_only_save_load_roundtrip(tmp_path):
    """The order-selective model is a uniform-layout model with zeros in the
    excluded block, so the standard persistence path round-trips it exactly."""
    from hifi_anova.model.io import save_model, load_model
    X, y = _data(N=250)
    res = _fit(X, y, K1=3, K2={(2, 3): 3}, variable_orders={2: [2]})
    path = str(tmp_path / 'm')
    save_model(res.model, path, config=res.config)
    loaded = load_model(path)['model']
    xg = np.asarray(res._data['x_test'])[:16]
    np.testing.assert_allclose(
        np.asarray(res.model.predict_mean_only(xg)),
        np.asarray(loaded.predict_mean_only(xg)))
    b1 = basis_size(3, True, 'fourier')
    np.testing.assert_array_equal(
        np.asarray(loaded.mean_model.w1)[2 * b1:3 * b1], 0.0)
