"""Public second-order variance model (X11C-S02 / GUI3 request BR-05).

``K2h`` was always threaded internally through Stage D; these tests pin the
PUBLIC surface: a one-call heteroscedastic fit can ask for a second-order
variance model and gets a populated ``log_variance_sobol['second_order']``,
``sigma_x2`` reflecting the pair variance term, and — new in X11C-S02 — an
EXPLICIT ``var_pair_selection`` list of (i, j) pairs (previously a list
silently behaved as 'all'). Default ``K2h=0`` behavior is unchanged (pinned
by the golden characterization and the existing Stage-D suites).
"""

import warnings

import numpy as np
import pytest

from hifi_anova.api import hifi_anova

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _hetero_pair_noise(N=700, seed=3):
    rng = np.random.default_rng(seed)
    X = rng.uniform(size=(N, 3))
    h = (-2.5 + 1.2 * np.sin(2 * np.pi * X[:, 0]) * np.sin(2 * np.pi * X[:, 1])
         + 0.8 * (X[:, 2] - 0.5))
    y = (np.sin(2 * np.pi * X[:, 0]) + X[:, 1]
         + np.exp(0.5 * h) * rng.standard_normal(N))
    return X, y


def test_public_k2h_populates_second_order_variance_sobol():
    X, y = _hetero_pair_noise()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = hifi_anova(X, y, K1=3, K2=0, heteroscedastic=True,
                         mode='heteroscedastic', K2h=2, verbose=False,
                         heteroscedastic_guard=False)
    vm = res.model.variance_model
    assert vm is not None and vm.K2h == 2
    lvs = res.sobol['log_variance_sobol']
    assert lvs['second_order'], "second-order variance Sobol should be populated"
    assert all(isinstance(k, tuple) and len(k) == 2 for k in lvs['second_order'])
    assert np.all(np.asarray(res.sigma_x2(X[:16])) > 0)
    # the planted (0, 1) log-variance interaction dominates the pair shares
    top = max(lvs['second_order'], key=lvs['second_order'].get)
    assert top == (0, 1)


def test_explicit_var_pair_selection_list():
    X, y = _hetero_pair_noise()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = hifi_anova(X, y, K1=3, K2=0, heteroscedastic=True,
                         mode='heteroscedastic', K2h=2,
                         var_pair_selection=[(0, 1)], verbose=False,
                         heteroscedastic_guard=False)
    vm = res.model.variance_model
    assert np.asarray(vm.pair_indices_h).tolist() == [[0, 1]]
    assert set(res.sobol['log_variance_sobol']['second_order']) == {(0, 1)}


def test_var_pair_selection_list_validation():
    X, y = _hetero_pair_noise(N=200)
    with pytest.raises(ValueError, match="canonical"):
        hifi_anova(X, y, heteroscedastic=True, mode='heteroscedastic',
                   K2h=2, var_pair_selection=[(1, 0)], verbose=False)
    with pytest.raises(ValueError, match="out of range"):
        hifi_anova(X, y, heteroscedastic=True, mode='heteroscedastic',
                   K2h=2, var_pair_selection=[(0, 7)], verbose=False)
    with pytest.raises(ValueError, match="pair"):
        hifi_anova(X, y, heteroscedastic=True, mode='heteroscedastic',
                   K2h=2, var_pair_selection=[0, 1], verbose=False)
