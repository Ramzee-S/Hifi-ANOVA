"""Pin the shared DEC-028 leverage-correction primitive.

``debias_squared_residuals`` is the single source of truth for the
``r^2 / clip(1 - lev, 1e-3, 1)`` de-biasing that the trainer's Stage-D
alternating loop and ``joint_lambda._joint_fit`` both feed into the Newton
log-variance solve. The trainer comment says the two "must stay in sync"; this
test locks the exact formula and the ``1e-3`` clip floor so a future edit to one
call site cannot silently diverge from the other.
"""

import numpy as np
import pytest

from hifi_anova.training.ridge import (
    debias_squared_residuals,
    _LEV_CORRECTION_CLIP_LO,
)


def _reference(r2, lev):
    """The exact expression previously inlined at every call site."""
    return np.asarray(r2, dtype=np.float64) / np.clip(1.0 - lev, 1e-3, 1.0)


def test_matches_inline_reference():
    rng = np.random.RandomState(0)
    r2 = rng.rand(50) ** 2
    lev = rng.rand(50) * 0.95  # leverages in [0, 0.95)
    np.testing.assert_array_equal(
        debias_squared_residuals(r2, lev), _reference(r2, lev))


def test_no_correction_is_passthrough():
    r2 = np.array([0.1, 0.4, 2.0])
    out = debias_squared_residuals(r2, np.array([0.5, 0.9, 0.99]), correct=False)
    np.testing.assert_array_equal(out, r2.astype(np.float64))


def test_zero_leverage_is_identity():
    r2 = np.array([0.25, 1.0, 3.5])
    np.testing.assert_array_equal(
        debias_squared_residuals(r2, np.zeros(3)), r2.astype(np.float64))


def test_clip_floor_bounds_the_blowup():
    # lev -> 1 (interpolation) must not blow up: denominator floored at 1e-3.
    r2 = np.array([2.0])
    out = debias_squared_residuals(r2, np.array([1.0]))
    assert np.isclose(out[0], 2.0 / _LEV_CORRECTION_CLIP_LO)
    # even lev slightly above 1 (numerical) stays clamped to the same floor.
    out_over = debias_squared_residuals(r2, np.array([1.5]))
    assert np.isclose(out_over[0], 2.0 / _LEV_CORRECTION_CLIP_LO)


def test_returns_float64():
    out = debias_squared_residuals(np.array([1, 2, 3], dtype=np.float32),
                                   np.array([0.1, 0.2, 0.3], dtype=np.float32))
    assert out.dtype == np.float64


@pytest.mark.parametrize("lev_val", [0.0, 0.3, 0.6, 0.9, 0.999])
def test_monotone_in_leverage(lev_val):
    # de-biased r^2 is non-decreasing in leverage (weights inflate where the
    # mean fit is tight) — a sanity check on the correction's direction.
    r2 = np.array([1.0])
    base = debias_squared_residuals(r2, np.array([0.0]))[0]
    assert debias_squared_residuals(r2, np.array([lev_val]))[0] >= base - 1e-15
