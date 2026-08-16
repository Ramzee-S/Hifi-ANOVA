"""Selectable SPD-inverse backend (DEC-035).

Default is ``'inv'`` (``numpy.linalg.inv``) — byte-identical to the historical
behaviour, so the golden master is untouched. ``'cholesky'`` is opt-in (more
stable / cheaper, but shifts one near-noiseless overfit scenario's tiny
``sigma_hat``, which is why it is not the default).
"""

import os

import numpy as np
import pytest

from hifi_anova import linalg as L


@pytest.fixture(autouse=True)
def _clean_linalg_state():
    L.set_linalg_method(None)
    saved = os.environ.pop("HIFI_LINALG", None)
    yield
    L.set_linalg_method(None)
    if saved is not None:
        os.environ["HIFI_LINALG"] = saved
    else:
        os.environ.pop("HIFI_LINALG", None)


def _spd(n=6, seed=0):
    rng = np.random.RandomState(seed)
    B = rng.randn(n, n)
    return B @ B.T + n * np.eye(n)   # symmetric positive-definite


def test_default_is_inv():
    assert L.resolve_linalg_method() == "inv"


def test_default_inverse_is_byte_identical_to_numpy():
    A = _spd()
    np.testing.assert_array_equal(L.spd_inverse(A), np.linalg.inv(A))


def test_cholesky_matches_inv_closely():
    A = _spd()
    chol = L.spd_inverse(A, method="cholesky")
    np.testing.assert_allclose(chol, np.linalg.inv(A), rtol=1e-10, atol=1e-12)
    # ...and actually inverts A.
    np.testing.assert_allclose(A @ chol, np.eye(A.shape[0]), atol=1e-9)


def test_precedence_arg_over_env_and_override():
    L.set_linalg_method("cholesky")
    os.environ["HIFI_LINALG"] = "cholesky"
    assert L.resolve_linalg_method("inv") == "inv"


def test_env_selects_cholesky():
    os.environ["HIFI_LINALG"] = "cholesky"
    assert L.resolve_linalg_method() == "cholesky"


def test_override_selects_cholesky():
    L.set_linalg_method("cholesky")
    assert L.resolve_linalg_method() == "cholesky"


def test_invalid_method_raises():
    with pytest.raises(ValueError):
        L.resolve_linalg_method("qr")


def test_cholesky_falls_back_on_non_pd():
    # A singular (not PD) matrix: Cholesky fails internally → falls back to inv,
    # which itself raises LinAlgError — i.e. no silent wrong answer.
    A = np.zeros((3, 3))
    with pytest.raises(np.linalg.LinAlgError):
        L.spd_inverse(A, method="cholesky")
