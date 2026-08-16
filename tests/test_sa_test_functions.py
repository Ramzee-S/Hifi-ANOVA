"""Tests for the sensitivity-analysis benchmark generators added in Phase 2.4.

Sobol G-function (+ analytic first-order Sobol) and the Morris (1991) function,
in hifi_anova.data.test_functions.
"""

import numpy as np
import pytest

from hifi_anova.data import (sobol_g_function, sobol_g_sobol, morris_function,
                             SOBOL_G_DEFAULT_A, friedman1_sobol_indices)


# ── Friedman-1 exact Sobol indices (quadrature) ──────────────────────────────

@pytest.mark.smoke
def test_friedman1_sobol_exact():
    s = friedman1_sobol_indices()
    fo, s12 = s['first_order'], s['second_order'][(0, 1)]
    # First-order indices match the literature/SALib reference to 3 decimals.
    for i, ref in zip(range(5), [0.197, 0.197, 0.093, 0.350, 0.087]):
        assert abs(fo[i] - ref) < 1e-3, f"S{i+1}={fo[i]:.4f} vs ref {ref}"
    assert abs(s12 - 0.075) < 1e-3          # closed (x1,x2) pair index
    # Full first-order + the single interaction partition the variance (=1).
    assert abs(sum(fo.values()) + s12 - 1.0) < 1e-9
    # Symmetry S1 == S2, and determinism across quadrature resolutions.
    assert abs(fo[0] - fo[1]) < 1e-12
    s2 = friedman1_sobol_indices(n_quad=96)
    assert abs(s2['first_order'][3] - fo[3]) < 1e-8


# ── Sobol G-function ─────────────────────────────────────────────────────────

@pytest.mark.smoke
def test_sobol_g_shapes_and_determinism():
    rng = np.random.default_rng(0)
    X = rng.random((500, 8))
    y1 = sobol_g_function(X)
    y2 = sobol_g_function(X)
    assert y1.shape == (500,)
    assert np.array_equal(y1, y2)                      # pure function of X


@pytest.mark.smoke
def test_sobol_g_analytic_total_variance():
    """MC variance of the G-function must match the analytic total variance
    V_T = prod(1 + V_i) - 1 that the Sobol formula is built on."""
    a = np.array(SOBOL_G_DEFAULT_A, dtype=float)
    Vi = 1.0 / (3.0 * (1.0 + a) ** 2)
    VT = float(np.prod(1.0 + Vi) - 1.0)
    rng = np.random.default_rng(1)
    X = rng.random((400_000, 8))
    y = sobol_g_function(X)
    assert np.var(y) == pytest.approx(VT, rel=0.03)


@pytest.mark.smoke
def test_sobol_g_indices_normalized_and_ordered():
    s = sobol_g_sobol(8)
    assert set(s) == set(range(8))
    # interactions exist, so first-order indices sum to < 1
    assert 0.0 < sum(s.values()) < 1.0
    # importance decreases as a_i increases (default a is non-decreasing)
    vals = [s[i] for i in range(8)]
    assert all(vals[i] >= vals[i + 1] - 1e-12 for i in range(7))
    assert s[0] > s[7]


@pytest.mark.smoke
def test_sobol_g_custom_a_all_equal():
    # a_i all equal => all first-order indices equal
    s = sobol_g_sobol(8, a=np.ones(8))
    assert np.allclose(list(s.values()), s[0])


# ── Morris function ──────────────────────────────────────────────────────────

@pytest.mark.smoke
def test_morris_shapes_determinism_and_dim_guard():
    rng = np.random.default_rng(2)
    X = rng.random((300, 20))
    y1 = morris_function(X)
    y2 = morris_function(X)
    assert y1.shape == (300,) and np.array_equal(y1, y2)
    with pytest.raises(ValueError):
        morris_function(rng.random((10, 12)))          # D < 20


@pytest.mark.smoke
def test_morris_inactive_variables_have_no_effect():
    """Variables 10..19 have zero main-effect coefficient and no interaction,
    so the output must be exactly invariant to them."""
    rng = np.random.default_rng(3)
    X = rng.random((200, 20))
    y_ref = morris_function(X)
    X2 = X.copy()
    X2[:, 10:] = rng.random((200, 10))                 # scramble inactive block
    assert np.array_equal(morris_function(X2), y_ref)
