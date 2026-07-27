"""Integrity checks for the committed heteroscedastic-Ishigami benchmark CSVs."""

import os
import numpy as np
import pytest

BENCH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     'benchmarks', 'ishigami_hetero')

pytestmark = pytest.mark.skipif(
    not os.path.isdir(BENCH),
    reason="benchmark dataset not present")


def _load(name):
    return np.genfromtxt(os.path.join(BENCH, name), delimiter=',', names=True)


def test_shapes_and_columns():
    tr, te, tt = _load('train.csv'), _load('test.csv'), _load('test_truth.csv')
    assert tr.shape[0] == 2000 and te.shape[0] == 5000 and tt.shape[0] == 5000
    assert set(tr.dtype.names) == {'x1', 'x2', 'x3', 'y'}
    assert set(tt.dtype.names) == {'x1', 'x2', 'x3', 'f_true', 'sigma_true'}
    # test.csv and test_truth.csv share the same inputs, row-aligned.
    for c in ('x1', 'x2', 'x3'):
        assert np.allclose(te[c], tt[c])


def test_truth_matches_ishigami_and_noise_ramp():
    tt = _load('test_truth.csv')
    x1, x2, x3 = tt['x1'], tt['x2'], tt['x3']
    f = np.sin(x1) + 7.0 * np.sin(x2) ** 2 + 0.1 * (x3 ** 4) * np.sin(x1)
    assert np.allclose(tt['f_true'], f, atol=1e-6)
    # sigma ramps linearly 0.3 -> 3.0 across x3 in [-pi, pi].
    u = (x3 + np.pi) / (2 * np.pi)
    assert np.allclose(tt['sigma_true'], 0.3 + 2.7 * u, atol=1e-6)
    assert tt['sigma_true'].min() >= 0.3 - 1e-6


def test_inputs_in_range():
    for name in ('train.csv', 'test.csv'):
        d = _load(name)
        for c in ('x1', 'x2', 'x3'):
            assert d[c].min() >= -np.pi - 1e-6 and d[c].max() <= np.pi + 1e-6
