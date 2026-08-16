"""Variance-variable subset (X11C-S03 / GUI3 request BR-01 Option A).

``variance_variables=[...]`` restricts the FIRST-ORDER variance (Stage-D) model
to a user-named subset of inputs. Excluding a variable ASSERTS the noise is
homoscedastic along it — a modeling assumption, not a data-driven finding — so:

* the fitted variance design ``Psi`` spends df only on the subset
  (``len(vv) * basis_size(Kh, …)`` first-order columns);
* ``sigma_x2`` is flat along every excluded variable, and each excluded
  variable's log-variance Sobol share is identically 0 (the Sʰ keys still span
  all D so downstream consumers see the full spectrum);
* ``VarianceModel.get_coefficients_for_variable`` maps an excluded variable to
  a zero block.

The subset composes with second-order variance (``K2h``: pairs must lie inside
the subset), ``var_pair_selection='auto'``, the Stage-D guard, and the mean-side
term structure (BR-04 per-pair K2 + BR-06 variable_orders) — the full
user-defined equation system in one fit. Persistence is via the model pickle
(a variance model can't be rebuilt from the uniform template); the field
survives and is mirrored (descriptively) into meta.json.

Default (``variance_variables=None``) behavior is untouched by construction and
pinned by the golden characterization; a full-set subset normalizes back to the
uniform path.
"""

import warnings

import numpy as np
import pytest

from hifi_anova.api import hifi_anova
from hifi_anova.core.features import basis_size

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _fit(X, y, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return hifi_anova(X, y, verbose=False, **kw)


def _hetero_x0(N=600, D=4, seed=0):
    """Noise scale rises along x0 only; the mean lives elsewhere."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(size=(N, D))
    sigma = 0.05 + 0.5 * X[:, 0]
    y = (np.sin(2 * np.pi * X[:, 1]) + 0.7 * X[:, 2]
         + sigma * rng.standard_normal(N))
    return X, y


# ---------------------------------------------------------- subset semantics

def test_variance_subset_excludes_are_flat_and_zero():
    X, y = _hetero_x0()
    D = X.shape[1]
    res = _fit(X, y, K1=4, Kh=3, variance_variables=[0],
               heteroscedastic=True, mode='heteroscedastic',
               heteroscedastic_guard=False)
    vm = res.model.variance_model
    assert vm.variance_variables == (0,)

    # log-variance Sobol keys span all D; every excluded variable is exactly 0
    sh = res.sobol['log_variance_sobol']['first_order']
    assert set(sh) == set(range(D))
    assert sh[0] > 0.0
    assert all(sh[i] == 0.0 for i in (1, 2, 3))

    # sigma^2(x) is flat along an excluded variable, varies along the included one
    def _grid(vary):
        g = np.full((24, D), 0.3)
        g[:, vary] = np.linspace(0.05, 0.95, 24)
        return g
    s_excl = np.asarray(res.sigma_x2(_grid(1)))
    s_incl = np.asarray(res.sigma_x2(_grid(0)))
    np.testing.assert_allclose(s_excl, s_excl[0], rtol=1e-5, atol=1e-8)
    assert s_incl.max() / s_incl.min() > 1.5

    # coefficient accessor: excluded -> zero block, included -> nonzero
    assert np.any(np.asarray(vm.get_coefficients_for_variable(0)) != 0)
    np.testing.assert_array_equal(
        np.asarray(vm.get_coefficients_for_variable(1)), 0.0)


def test_variance_subset_spends_df_on_subset_only():
    X, y = _hetero_x0(seed=1)
    res = _fit(X, y, K1=4, Kh=3, variance_variables=[0],
               heteroscedastic=True, mode='heteroscedastic',
               heteroscedastic_guard=False)
    rec = res._fitted_design
    var_rec = rec.variance
    assert var_rec is not None
    # first-order variance design width = len(vv) * basis_size(Kh)
    b1h = basis_size(3, True, 'fourier')
    # only the first-order block is subset-controlled here (K2h=0)
    assert var_rec.Psi.shape[1] == 1 * b1h


# ------------------------------------------------------------- compositions

def test_variance_subset_with_k2h_pairs_inside_subset():
    rng = np.random.default_rng(2)
    N, D = 700, 4
    X = rng.uniform(size=(N, D))
    sigma = 0.05 + 0.5 * X[:, 0] * X[:, 1]        # hetero along the (0,1) pair
    y = np.sin(2 * np.pi * X[:, 2]) + sigma * rng.standard_normal(N)
    res = _fit(X, y, K1=4, Kh=3, K2h=2, variance_variables=[0, 1],
               var_pair_selection='all', heteroscedastic=True,
               mode='heteroscedastic', heteroscedastic_guard=False)
    so2 = res.sobol['log_variance_sobol'].get('second_order', {})
    # the only in-subset pair is (0, 1); no pair may touch an excluded var
    assert set(so2) == {(0, 1)}


def test_variance_subset_auto_pair_selection():
    rng = np.random.default_rng(2)
    N, D = 700, 4
    X = rng.uniform(size=(N, D))
    sigma = 0.05 + 0.5 * X[:, 0] * X[:, 1]
    y = np.sin(2 * np.pi * X[:, 2]) + sigma * rng.standard_normal(N)
    res = _fit(X, y, K1=4, Kh=3, K2h=2, variance_variables=[0, 1],
               var_pair_selection='auto', heteroscedastic=True,
               mode='heteroscedastic', heteroscedastic_guard=False)
    so2 = res.sobol['log_variance_sobol'].get('second_order', {})
    # auto may keep or drop the pair, but can never propose one outside subset
    assert all(i in (0, 1) and j in (0, 1) for (i, j) in so2)


def test_variance_subset_guard_revert():
    """Homoscedastic data + a variance subset ⇒ the Stage-D guard reverts to
    constant variance; the model still predicts and gives finite intervals."""
    rng = np.random.default_rng(3)
    N, D = 500, 4
    X = rng.uniform(size=(N, D))
    y = np.sin(2 * np.pi * X[:, 1]) + 0.15 * rng.standard_normal(N)
    res = _fit(X, y, K1=4, Kh=3, variance_variables=[0],
               heteroscedastic=True, mode='heteroscedastic')
    sd = res.train_results['stage_D']
    assert sd.get('reverted') is True
    assert np.all(np.isfinite(res.predict(X[:8])))
    lo, hi = res.predict_intervals(X[:5])
    assert np.all(np.isfinite(lo)) and np.all(np.asarray(hi) >= np.asarray(lo))


def test_full_user_defined_system():
    """BR-04 (per-pair K2) + BR-06 (order-2-only var) + BR-01 (variance subset)
    in one heteroscedastic fit: the mean drops x2's marginal and keeps the
    (2,3) pair, the variance lives on x0 only, and prediction + intervals are
    finite."""
    rng = np.random.default_rng(2)
    N, D = 700, 4
    X = rng.uniform(size=(N, D))
    sigma = 0.05 + 0.5 * X[:, 0]                  # variance along x0 (in subset)
    y = (np.sin(2 * np.pi * X[:, 2]) * np.sin(2 * np.pi * X[:, 3])
         + np.sin(2 * np.pi * X[:, 1]) + sigma * rng.standard_normal(N))
    res = _fit(X, y, K1=4, K2={(2, 3): 3}, variable_orders={2: [2]},
               variable_selection=None, Kh=3, variance_variables=[0],
               heteroscedastic=True, mode='heteroscedastic',
               heteroscedastic_guard=False)
    # mean side (BR-06 + BR-04)
    assert res.sobol['mean_sobol']['first_order'][2] == 0.0
    assert (2, 3) in res.sobol['mean_sobol']['second_order']
    # variance side (BR-01)
    sh = res.sobol['log_variance_sobol']['first_order']
    assert sh[0] > 0.0 and all(sh[i] == 0.0 for i in (1, 2, 3))
    # end-to-end prediction + intervals
    assert np.all(np.isfinite(res.predict(X[:8])))
    lo, hi = res.predict_intervals(X[:5])
    assert np.all(np.isfinite(lo)) and np.all(np.asarray(hi) >= np.asarray(lo))


# ------------------------------------------------------------- persistence

def test_variance_subset_save_load_roundtrip(tmp_path):
    from hifi_anova.model.io import save_model, load_model
    X, y = _hetero_x0(seed=4)
    res = _fit(X, y, K1=4, Kh=3, variance_variables=[0],
               heteroscedastic=True, mode='heteroscedastic',
               heteroscedastic_guard=False)
    path = str(tmp_path / 'm')
    save_model(res.model, path, config=res.config)
    loaded = load_model(path)['model']
    assert getattr(loaded.variance_model, 'variance_variables', None) == (0,)
    # meta.json carries the descriptive mirror
    import json
    import os
    with open(os.path.join(path, 'meta.json')) as f:
        meta = json.load(f)
    assert meta['variance_variables'] == [0]
    # sigma^2(x) round-trips exactly
    xg = np.asarray(res._data['x_test'])[:16]
    _m0, v0 = res.model.predict(xg)
    _m1, v1 = loaded.predict(xg)
    np.testing.assert_allclose(np.asarray(v0), np.asarray(v1))


# -------------------------------------------------------------- validation

def test_variance_variables_validation_errors():
    X, y = _hetero_x0(N=200, seed=5)
    with pytest.raises(ValueError, match="at least one variable"):
        _fit(X, y, variance_variables=[], heteroscedastic=True,
             mode='heteroscedastic')
    with pytest.raises(ValueError, match="repeats index"):
        _fit(X, y, variance_variables=[0, 0], heteroscedastic=True,
             mode='heteroscedastic')
    with pytest.raises(ValueError, match="out of range"):
        _fit(X, y, variance_variables=[9], heteroscedastic=True,
             mode='heteroscedastic')
    # a variance pair on a variable outside the subset is contradictory
    with pytest.raises(ValueError, match="outside variance_variables"):
        _fit(X, y, K1=4, Kh=3, K2h=2, variance_variables=[0, 1],
             var_pair_selection=[(0, 2)], heteroscedastic=True,
             mode='heteroscedastic')


def test_summary_surfaces_variance_structure():
    """summary() reports the variance subset and its homoscedasticity-asserted
    caveat (visible, not only in the model field)."""
    import contextlib
    import io
    X, y = _hetero_x0(seed=7)
    res = _fit(X, y, K1=4, Kh=3, variance_variables=[0],
               heteroscedastic=True, mode='heteroscedastic',
               heteroscedastic_guard=False)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res.summary()
    out = buf.getvalue()
    assert 'Variance structure' in out
    assert 'ASSERTED' in out
    assert '[0]' in out


def test_full_set_normalizes_to_uniform():
    """Naming every variable is the uniform variance model: the subset field is
    normalized away so the default path (and its persistence template) apply."""
    X, y = _hetero_x0(seed=6)
    D = X.shape[1]
    res = _fit(X, y, K1=4, Kh=3, variance_variables=list(range(D)),
               heteroscedastic=True, mode='heteroscedastic',
               heteroscedastic_guard=False)
    assert getattr(res.model.variance_model, 'variance_variables', None) is None
