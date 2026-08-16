"""Unit tests for the FittedDesign record (Phase 0 — dataclass only).

These exercise the record's structure and accessors in isolation; no trainer or
API wiring is involved yet. See ``hifi_anova/training/fitted_design.py`` and
``FittedDesignRecord_brief.md``.
"""

import numpy as np
import pytest

from hifi_anova.training.fitted_design import DesignBlock, FittedDesign

pytestmark = pytest.mark.smoke


def _block1(D=3, B=5, cols=slice(0, 15)):
    return DesignBlock(order=1, K=2, basis_name='fourier', include_linear=True,
                       columns=cols, gram=np.eye(B), n_groups=D)


def _homoscedastic_record(D=3):
    N, F = 40, 15
    return FittedDesign(
        Phi=np.zeros((N, F)),
        w=np.zeros(F),
        reg_diag=np.ones(F),
        y_centered=np.zeros(N),
        f0=0.5,
        D=D,
        sample_weights=None,
        blocks=[_block1(D=D)],
    )


def test_construction_defaults():
    rec = _homoscedastic_record()
    assert rec.D == 3
    assert rec.sample_weights is None
    assert rec.interpretive is None
    assert len(rec.blocks) == 1


def test_is_weighted_flag():
    rec = _homoscedastic_record()
    assert rec.is_weighted is False
    rec.sample_weights = np.ones(40)
    assert rec.is_weighted is True


def test_block_lookup():
    b2 = DesignBlock(order=2, K=3, basis_name='fourier', include_linear=True,
                     columns=slice(15, 15 + 2 * 49), gram=np.eye(49), n_groups=2,
                     indices=np.array([[0, 1], [0, 2]]))
    rec = _homoscedastic_record()
    rec.blocks.append(b2)
    assert rec.block(1).order == 1
    assert rec.block(2) is b2
    assert rec.block(3) is None


def test_attribution_record_homoscedastic_is_self():
    rec = _homoscedastic_record()
    assert rec.attribution_record() is rec


def test_attribution_record_weighted_uses_companion():
    rec = _homoscedastic_record()
    companion = _homoscedastic_record()
    rec.sample_weights = np.ones(40)
    rec.interpretive = companion
    assert rec.attribution_record() is companion


def test_sobol_ci_kwargs_first_order_only():
    rec = _homoscedastic_record(D=3)
    kw = rec.sobol_ci_kwargs()
    assert kw['D'] == 3
    assert kw['K1'] == 2
    assert kw['basis_name'] == 'fourier'
    assert kw['include_linear_1'] is True
    assert kw['G1'].shape == (5, 5)
    # No second/third-order keys when only order-1 present.
    assert 'K2' not in kw
    assert 'K3' not in kw


def test_sobol_ci_kwargs_includes_higher_orders():
    rec = _homoscedastic_record(D=3)
    rec.blocks.append(DesignBlock(
        order=2, K=3, basis_name='fourier', include_linear=True,
        columns=slice(15, 113), gram=np.eye(49), n_groups=2,
        indices=np.array([[0, 1], [0, 2]])))
    rec.blocks.append(DesignBlock(
        order=3, K=2, basis_name='fourier', include_linear=True,
        columns=slice(113, 238), gram=np.eye(125), n_groups=1,
        indices=np.array([[0, 1, 2]])))
    kw = rec.sobol_ci_kwargs()
    assert kw['K2'] == 3 and kw['P'] == 2 and kw['G2'].shape == (49, 49)
    assert kw['pair_indices'].shape == (2, 2)
    assert kw['K3'] == 2 and kw['T'] == 1 and kw['G3'].shape == (125, 125)
    assert kw['triple_indices'].shape == (1, 3)


def test_sobol_ci_kwargs_skips_empty_higher_orders():
    """An order with zero retained groups (e.g. all pairs pruned) is omitted."""
    rec = _homoscedastic_record(D=3)
    rec.blocks.append(DesignBlock(
        order=2, K=3, basis_name='fourier', include_linear=True,
        columns=slice(15, 15), gram=np.eye(49), n_groups=0, indices=None))
    kw = rec.sobol_ci_kwargs()
    assert 'K2' not in kw


def test_sobol_ci_kwargs_requires_first_order():
    rec = _homoscedastic_record()
    rec.blocks = []
    with pytest.raises(ValueError):
        rec.sobol_ci_kwargs()
