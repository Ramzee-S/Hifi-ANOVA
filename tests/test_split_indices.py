"""Train/val/test split provenance on the fitted result (BR-07).

``preprocess_data`` shuffles with ``RandomState(seed)`` and holds out val/test;
these tests pin that the split row indices are now carried through to the
result so a caller can map a per-point diagnostic (LOO residual, leverage,
worst out-of-sample row) back to an original dataset row id WITHOUT re-deriving
the seeded permutation.

Acceptance (BR-07):
* ``X[result.split_indices['train']]`` reproduces the rows the fitted design's
  ``Phi`` was built from, in ``Phi`` order (same for y);
* present for every fit path that runs ``preprocess_data``;
* read-only — no fitting behavior changes (the golden characterization pins
  that defaults stay byte-identical).
"""

import warnings

import numpy as np
import pytest

from hifi_anova.api import hifi_anova
from hifi_anova.data.preprocessing import preprocess_data

pytestmark = pytest.mark.integration


def _data(N=300, D=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(size=(N, D))
    y = (np.sin(2 * np.pi * X[:, 0]) + X[:, 1] ** 2
         + 0.8 * X[:, 2] * X[:, 3] + 0.1 * rng.standard_normal(N))
    return X, y


def test_preprocess_returns_partition_indices():
    """The three index arrays partition 0..N-1 with the documented sizes and
    row order (train first, then val, then test)."""
    X, y = _data(N=200)
    d = preprocess_data(X, y, seed=42)
    tr, va, te = d['train_indices'], d['val_indices'], d['test_indices']
    assert len(tr) == d['n_train'] and len(va) == d['n_val'] and len(te) == d['n_test']
    allidx = np.concatenate([tr, va, te])
    # a genuine partition of every row, no overlaps
    assert np.array_equal(np.sort(allidx), np.arange(len(X)))
    assert len(np.unique(allidx)) == len(X)
    # rows are NOT re-sorted — they carry the permutation order used for Phi
    assert not np.array_equal(tr, np.sort(tr))


def test_split_indices_reproduce_phi_rows():
    """X[train] transformed reproduces x_train (Phi rows) in order; y matches
    exactly. This is the BR-07 contract the console needs."""
    X, y = _data(N=300)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = hifi_anova(X, y, K1=4, K2=3, verbose=False)
    split = res.split_indices
    assert set(split) == {'train', 'val', 'test'}

    # y is only dtype-cast, never transformed → exact reproduction
    y_train = np.asarray(res._data['y_train'])
    np.testing.assert_allclose(y[split['train']].astype(y_train.dtype), y_train)

    # X[train] through the fitted transformer (clipped) == the model's x_train
    x_train = np.asarray(res._data['x_train'])
    x_from_idx = np.clip(res.transformer.transform(X[split['train']]), 0, 1)
    np.testing.assert_allclose(
        x_from_idx.astype(x_train.dtype), x_train, rtol=0, atol=1e-6)

    # partition covers the whole dataset
    allidx = np.concatenate([split['train'], split['val'], split['test']])
    assert np.array_equal(np.sort(allidx), np.arange(len(X)))


@pytest.mark.parametrize("kwargs", [
    {'K1': 4, 'K2': 0},                                    # first-order only
    {'K1': 4, 'K2': 3},                                    # second order
    {'K1': 4, 'K2': 3, 'heteroscedastic': True,
     'mode': 'heteroscedastic'},                           # Stage D
    {'K1': 4, 'basis_per_variable': {0: {'basis': 'legendre', 'K': 4}}},  # mixed-K
])
def test_split_indices_present_across_fit_paths(kwargs):
    """Every fit path that runs preprocess_data carries the split (BR-07)."""
    X, y = _data(N=320)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = hifi_anova(X, y, verbose=False, **kwargs)
    split = res.split_indices
    assert split is not None
    # y reproduction holds regardless of the mean/variance structure
    y_train = np.asarray(res._data['y_train'])
    np.testing.assert_allclose(y[split['train']].astype(y_train.dtype), y_train)


def test_split_indices_none_without_preprocess():
    """A result without the preprocessing split returns None, not a crash."""
    from hifi_anova.api import HiFiResult
    r = HiFiResult.__new__(HiFiResult)
    object.__setattr__(r, '_data', None)
    assert r.split_indices is None
