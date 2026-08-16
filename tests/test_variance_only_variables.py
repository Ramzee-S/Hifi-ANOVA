"""Variance-only variables + independent variance pairs (X10C XSes04).

``variable_orders={j: []}`` gives variable j VARIANCE-ONLY membership: no
mean term at any order, column kept for the Stage-D variance model. Together
with the existing pieces (BR-06 order-selective mean membership, BR-01
variance_variables, BR-05 explicit var_pair_selection — which the backend
validates against variance variables, NOT against the mean pair set) this
completes fully independent first/second-order muting on the mean and
variance sides. User-asserted structure, disclosed in
``train_results['term_structure']`` — not a data-driven selection.
"""
import numpy as np
import pytest

import hifi_anova as ha


def _hetero_data(n=400, seed=5):
    rng = np.random.default_rng(seed)
    X = rng.random((n, 4))
    sigma = 0.05 + 0.4 * X[:, 1]          # noise driven by x1 only
    y = (np.sin(2 * np.pi * X[:, 0]) + 0.6 * X[:, 2] * X[:, 3]
         + sigma * rng.standard_normal(n))  # mean has NO x1
    return X, y


def test_variance_only_variable_and_independent_var_pair():
    X, y = _hetero_data()
    r = ha.hifi_anova(
        X, y, K1=3, mode="second", basis_name="legendre",
        heteroscedastic=True, K2={(2, 3): 2},
        variable_orders={1: []},           # x1: variance-only
        K2h=1, var_pair_selection=[(0, 1)],  # NOT a mean pair — independent
        verbose=False, seed=42, backend="numpy")
    ts = r.train_results.get("term_structure") or {}
    assert ts.get("mean_excluded") == [1]
    # the variance model finds the x1 noise driver the mean model never sees
    vs = r.sobol.get("log_variance_sobol") or {}
    sh = {int(k): float(v) for k, v in (vs.get("first_order") or {}).items()}
    assert sh[1] > 0.5 and sh[1] == max(sh.values())
    # the independent variance pair was fitted
    assert (0, 1) in (vs.get("second_order") or {})
    # mean side: x1 carries (essentially) no mean share
    ms = {int(k): float(v)
          for k, v in r.sobol["mean_sobol"]["first_order"].items()}
    assert ms.get(1, 0.0) < 1e-6
    # sigma^2(x) tracks x1
    sig = np.asarray(r.sigma_x2(X))
    assert np.corrcoef(X[:, 1], sig)[0, 1] > 0.5


def test_variance_only_cross_backend_parity():
    X, y = _hetero_data(300, seed=6)
    kw = dict(K1=2, mode="second", basis_name="legendre",
              heteroscedastic=True, K2={(2, 3): 1},
              variable_orders={1: []}, verbose=False, seed=42,
              precision="float64")
    rn = ha.hifi_anova(X, y, backend="numpy", **kw)
    rj = ha.hifi_anova(X, y, backend="jax", **kw)
    assert np.max(np.abs(rn.predict(X) - rj.predict(X))) < 1e-8
    assert np.max(np.abs(np.asarray(rn.sigma_x2(X))
                         - np.asarray(rj.sigma_x2(X)))) < 1e-8


def test_variance_only_refusals():
    X, y = _hetero_data(120, seed=7)
    # BR-12 (Ses06): both former refusals are now WARNINGS — the column stays
    # in X as a complement input (constant fit), and all-excluded is the
    # legitimate INTERCEPT-ONLY complement-only base.
    with pytest.warns(UserWarning, match="neither the mean nor"):
        r = ha.hifi_anova(X, y, K1=2, mode="first",
                          variable_orders={1: []}, verbose=False,
                          backend="numpy")
    assert r.model.fo_included == (0, 2, 3)
    with pytest.warns(UserWarning, match="INTERCEPT-ONLY"):
        r = ha.hifi_anova(X, y, K1=2, heteroscedastic=True,
                          variable_orders={i: [] for i in range(4)},
                          verbose=False, backend="numpy")
    assert r.model.fo_included == ()
    # no spurious 'absent from the model' warning for a variance-only var
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("error",
                        append=False)  # any UserWarning would raise
        _w.filterwarnings("ignore", message=".*prediction inputs.*")
        _w.filterwarnings("ignore", message=".*variable_selection.*")
        ha.hifi_anova(X, y, K1=2, mode="first", basis_name="legendre",
                      heteroscedastic=True, variable_orders={1: []},
                      verbose=False, seed=42, backend="numpy")
