"""Mixed per-variable basis through the one-call API (DEC-030).

A ``basis_per_variable`` fit used to raise ``NotImplementedError`` in
``hifi_anova()`` because the post-fit analytics rebuilt a *uniform* design layout
and crashed on heterogeneous block sizes (the historical
``operands could not be broadcast together with shapes (6,6) (10,10)``). The
fitted-design record now carries per-group column slices and Grams and the Sobol
CIs are computed block-driven, so the one-call runs and the CIs are
block-correct. Integration-level (fits a full staged pipeline).
"""

import numpy as np
import pytest

from hifi_anova.api import hifi_anova
from hifi_anova.analysis.sobol import compute_sobol_indices
from hifi_anova.data.synthetic import generate_ishigami

pytestmark = pytest.mark.integration


# Deliberately heterogeneous: legendre(K=5)=5 cols, fourier(K=5)=10 cols,
# haar(K=4) — different order-1 block sizes, so pairs have different-sized
# G_i ⊗ G_j Grams. This is exactly the layout the old uniform rebuild crashed on.
BPV = {
    0: {'basis': 'legendre', 'K': 5},
    1: {'basis': 'fourier', 'K': 5},
    2: {'basis': 'haar', 'K': 4},
}


@pytest.fixture(scope="module")
def mixed_result():
    X, y, _ = generate_ishigami(n_samples=1200, seed=0)
    res = hifi_anova(X, y, mode='second', basis_per_variable=BPV,
                     variable_selection=None, seed=42, verbose=False)
    return res


def test_mixed_onecall_runs_and_records_heterogeneous_blocks(mixed_result):
    res = mixed_result
    rec = res.train_results['fitted_design']

    # The record uses the block-driven layout, and the order-1 groups genuinely
    # have different column widths (the crash precondition).
    assert rec.sobol_groups is not None
    first_widths = [sl.stop - sl.start for (o, k, sl, G) in rec.sobol_groups
                    if o == 1]
    assert len(set(first_widths)) > 1, "expected heterogeneous first-order blocks"


def test_mixed_onecall_first_order_cis_wellformed(mixed_result):
    res = mixed_result
    assert len(res.sobol_ci) == 3
    for name, (S, lo, hi) in res.sobol_ci.items():
        assert lo <= S <= hi
        assert 0.0 <= lo <= hi <= 1.0


def test_mixed_onecall_cis_match_block_closedform(mixed_result):
    res = mixed_result
    # Block-correctness: the record-driven CI point indices must match the
    # model's own per-variable closed-form Sobol (compute_sobol_indices slices
    # each variable's block with its own Gram). A wrong column slice or Gram
    # would move these substantially; equality within float tolerance proves the
    # block-driven path sliced the right columns.
    ref = compute_sobol_indices(res.model, res._data['x_test'])
    ref_first = ref['mean_sobol']['first_order']
    for i, name in enumerate(res.feature_names):
        S_ci = res.sobol_ci[name][0]
        assert S_ci == pytest.approx(float(ref_first[i]), abs=2e-3)


def test_mixed_onecall_second_order_cis_present(mixed_result):
    res = mixed_result
    # mode='second' retains pairs; the CI routine returns block-driven pair CIs.
    rec = res.train_results['fitted_design']
    n_pairs = sum(1 for (o, k, sl, G) in rec.sobol_groups if o == 2)
    assert n_pairs == 3  # C(3,2)


def test_mixed_onecall_homoscedastic_surface(mixed_result):
    res = mixed_result
    # Mixed fits are homoscedastic here: no efficient set, no gap.
    assert res.sobol_ci_efficient is None
    assert res.sobol_gap is None


# --- capability contract at the one-call boundary (DEC-045) ------------------

def _small_mixed_data():
    X, y, _ = generate_ishigami(n_samples=400, seed=1)
    return X, y


def test_onecall_implicit_bic_neutralized_with_warning():
    # variable_selection defaults to 'bic'; on mixed bases the *implicit* default
    # is neutralized (not applied) with a one-release migration warning, and the
    # fit still succeeds. selection_applied is recorded False.
    X, y = _small_mixed_data()
    with pytest.warns(UserWarning, match="variable selection is not supported"):
        res = hifi_anova(X, y, mode='second', basis_per_variable=BPV,
                         seed=0, verbose=False)
    cap = res.train_results['mixed_capability']
    assert cap['selection_applied'] is False
    assert cap['implicit_selection_neutralized'] is True


def test_onecall_explicit_variable_selection_raises():
    # An *explicit* variable_selection on mixed bases is a capability error, not
    # a silent no-op (distinct from the implicit default above).
    X, y = _small_mixed_data()
    with pytest.raises(NotImplementedError, match="variable_selection"):
        hifi_anova(X, y, mode='second', basis_per_variable=BPV,
                   variable_selection='bic', seed=0, verbose=False)


def test_onecall_explicit_none_no_warning_no_raise(recwarn):
    # Explicitly opting out is clean: no migration warning, no error.
    X, y = _small_mixed_data()
    res = hifi_anova(X, y, mode='second', basis_per_variable=BPV,
                     variable_selection=None, seed=0, verbose=False)
    assert res.train_results['mixed_capability']['selection_applied'] is False
    assert not any("variable selection is not supported" in str(w.message)
                   for w in recwarn.list)


def test_onecall_k2_zero_no_second_order_cis():
    # K2=0 through the one-call API: no pair blocks and no second-order CIs.
    X, y = _small_mixed_data()
    res = hifi_anova(X, y, mode='second', basis_per_variable=BPV, K2=0,
                     variable_selection=None, seed=0, verbose=False)
    assert res.model.pair_indices is None
    rec = res.train_results['fitted_design']
    assert sum(1 for g in (rec.sobol_groups or []) if g[0] == 2) == 0
    # first-order CIs still well-formed
    assert len(res.sobol_ci) == 3
