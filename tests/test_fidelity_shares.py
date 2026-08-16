"""M3/DEC-032: structural fidelity 𝔉 and core vs. total Sobol shares.

Covers the core→total bridge ``𝔉 = V_core/(V_core+Var(ĝ))`` with
``Ŝ_u^total = 𝔉·Ŝ_u^core`` (Manuscript_Theoryv06 §3.2 ``eq:index-scales`` / §8
``eq:fidelity``), the orthogonality-defect diagnostic ``2·Ĉov(f̂_core,ĝ)/Var(f̂)``,
and the homoscedastic/no-residual collapse (𝔉≡1, total≡core, ``sobol_ci_total``
None). The pure-math ``compute_fidelity`` checks are smoke; the reconciliation
across the two Sobol surfaces (point + CI) after a real fit is integration.
"""
import numpy as np
import pytest

from hifi_anova.analysis.sobol import compute_fidelity


@pytest.mark.smoke
class TestComputeFidelity:
    """Pure-math checks on the single-source-of-truth fidelity helper."""

    def test_no_residual_is_unit(self):
        f = compute_fidelity(v_core=2.5, residual_var=0.0, cross_cov=0.0)
        assert f['value'] == 1.0
        assert f['var_residual'] == 0.0
        assert f['orthogonality_defect'] == 0.0
        assert f['conditional_on_residual_variance'] is True

    def test_fidelity_ratio_and_total_identity(self):
        f = compute_fidelity(v_core=3.0, residual_var=1.0, cross_cov=0.0)
        assert abs(f['value'] - 0.75) < 1e-12            # 3/(3+1)
        s_core = 0.4                                     # Ŝ^total = 𝔉·Ŝ^core
        assert abs(f['value'] * s_core - 0.30) < 1e-12

    def test_zero_total_variance_defaults_unit(self):
        f = compute_fidelity(v_core=0.0, residual_var=0.0, cross_cov=0.0)
        assert f['value'] == 1.0                         # denom 0 → 𝔉 defaults to 1

    def test_orthogonality_defect_reported_not_folded(self):
        # Var(f̂) = V_core + Var(ĝ) + 2·cov = 3 + 1 + 2·0.5 = 5; defect = 1/5.
        f = compute_fidelity(v_core=3.0, residual_var=1.0, cross_cov=0.5)
        assert abs(f['orthogonality_defect'] - 0.2) < 1e-12
        assert f['cross_covariance'] == 0.5
        # 𝔉 itself must NOT absorb the cross term (folding forbidden, §8).
        assert abs(f['value'] - 0.75) < 1e-12

    def test_negative_cross_cov(self):
        f = compute_fidelity(v_core=2.0, residual_var=0.5, cross_cov=-0.25)
        # Var(f̂) = 2 + 0.5 - 0.5 = 2.0; defect = -0.5/2.0 = -0.25.
        assert abs(f['orthogonality_defect'] + 0.25) < 1e-12


def _fit_data(seed=0):
    """Structured signal + a 3-D radial bump the ≤2nd-order basis cannot represent,
    so a Stage-C residual reliably has genuine (non-trivial) variance."""
    rng = np.random.RandomState(seed)
    N, D = 500, 3
    X = rng.uniform(0.0, 1.0, (N, D))
    struct = np.sin(2 * np.pi * X[:, 0]) + 0.6 * X[:, 1]
    bump = 2.0 * np.exp(-8.0 * ((X[:, 0] - 0.5) ** 2 + (X[:, 1] - 0.5) ** 2
                                + (X[:, 2] - 0.5) ** 2))
    y = struct + bump + 0.03 * rng.randn(N)
    return X, y


@pytest.mark.integration
class TestFidelityFromFit:
    """The two Sobol surfaces (point + CI) reconcile onto one labeled convention."""

    def test_no_residual_collapses_byte_consistent(self):
        from hifi_anova.api import hifi_anova
        X, y = _fit_data()
        r = hifi_anova(X, y, mode='second', verbose=False)
        assert r.fidelity['value'] == 1.0
        assert r.fidelity['var_residual'] == 0.0
        assert r.fidelity['orthogonality_defect'] == 0.0
        assert r.sobol_ci_total is None                  # collapse to single set
        # Legacy fractions == labeled total == labeled core (all coincide at 𝔉=1).
        ms = r.sobol['mean_sobol']['first_order']
        msc = r.sobol['mean_sobol_core']['first_order']
        mst = r.sobol['mean_sobol_total']['first_order']
        for i in ms:
            assert abs(ms[i] - msc[i]) < 1e-9
            assert abs(ms[i] - mst[i]) < 1e-9

    def test_residual_reconciliation_point_and_ci(self):
        from hifi_anova.api import hifi_anova
        X, y = _fit_data()
        r = hifi_anova(X, y, mode='second', residual='rbf', verbose=False)
        f = r.fidelity
        assert f['var_residual'] > 0.0                   # residual genuinely present
        assert 0.0 < f['value'] < 1.0                    # 𝔉 strictly interior
        F = f['value']

        # Point path: labeled total == 𝔉·core, and legacy == total (back-compat alias).
        msc = r.sobol['mean_sobol_core']['first_order']
        mst = r.sobol['mean_sobol_total']['first_order']
        ms = r.sobol['mean_sobol']['first_order']
        for i in msc:
            assert abs(mst[i] - F * msc[i]) < 1e-9
            assert abs(ms[i] - mst[i]) < 1e-6

        # CI path: total reported and == 𝔉·core, conditional flag set.
        assert r.sobol_ci_total is not None
        for name in r.sobol_ci:
            assert abs(r.sobol_ci_total[name][0] - F * r.sobol_ci[name][0]) < 1e-9

        # Cross-term diagnostic exists and 𝔉 did not absorb it.
        assert 'cross_covariance' in f and 'orthogonality_defect' in f


@pytest.mark.integration
class TestReportingSurfaceDEC034:
    """DEC-034: naming/normalization reporting surface — labels, headline option,
    the sum-to-one invariant, and the opt-in observed-variance view."""

    def _residual_fit(self):
        from hifi_anova.api import hifi_anova
        X, y = _fit_data(seed=1)
        return hifi_anova(X, y, mode='second', residual='rbf', verbose=False)

    def test_point_path_shares_sum_to_one(self):
        """Fitted-variance shares (all orders) + the residual share = 1 — the
        accounting identity the new residual row makes visible."""
        r = self._residual_fit()
        ms = r.sobol['mean_sobol']            # total (fitted-variance) normalization
        total = (sum(ms['first_order'].values())
                 + sum(ms.get('second_order', {}).values())
                 + sum(ms.get('third_order', {}).values())
                 + ms.get('residual', 0.0))
        assert abs(total - 1.0) < 1e-6

    def test_observed_is_fitted_variance_times_fixed_scale(self):
        """sobol_ci_observed = sobol_ci_total · (Var f̂/Var Y), a single FIXED scale
        across components — and that scale is NOT R² for a regularized fit (the Q4
        rationale: computing the scaling is what keeps the number correct)."""
        r = self._residual_fit()
        assert r.sobol_ci_observed is not None
        scales = [r.sobol_ci_observed[n][0] / r.sobol_ci_total[n][0]
                  for n in r.sobol_ci_total if r.sobol_ci_total[n][0] > 1e-9]
        assert len(scales) >= 2
        assert max(scales) - min(scales) < 1e-9          # one fixed scale
        # Var(f̂)/Var(Y) ≠ R² once the fit is regularized/nonlinear.
        assert abs(scales[0] - r.r_squared) > 1e-3
        # Bounds/CI carried through the same fixed scale.
        for n in r.sobol_ci_total:
            k = scales[0]
            for a, b in zip(r.sobol_ci_observed[n], r.sobol_ci_total[n]):
                assert abs(a - b * k) < 1e-9

    def test_observed_populated_on_homoscedastic_from_core(self):
        """No residual (𝔉≡1) ⇒ sobol_ci_total is None but the observed view still
        populates, scaling the CORE shares by the fixed Var(f̂)/Var(Y)."""
        from hifi_anova.api import hifi_anova
        X, y = _fit_data(seed=2)
        r = hifi_anova(X, y, mode='second', verbose=False)
        assert r.sobol_ci_total is None
        assert r.sobol_ci_observed is not None
        scales = [r.sobol_ci_observed[n][0] / r.sobol_ci[n][0]
                  for n in r.sobol_ci if r.sobol_ci[n][0] > 1e-9]
        assert max(scales) - min(scales) < 1e-9

    def test_headline_option_and_validation(self):
        r = self._residual_fit()
        r.summary(headline='core')                       # both run without error
        r.summary(headline='fitted_variance')
        with pytest.raises(ValueError):
            r.summary(headline='bogus')

    def test_no_line_calls_the_fidelity_scaled_quantity_total_sobol(self, capsys):
        """The 'total' collision guard: the printed summary never calls the 𝔉-scaled
        share a 'total Sobol' index, while the genuine total-EFFECT index S_T remains
        available under mean_sobol['total_order']."""
        r = self._residual_fit()
        r.summary(observed=True)
        out = capsys.readouterr().out
        assert 'total Sobol' not in out
        assert 'fitted-variance' in out
        assert '(residual ĝ)' in out and '1−𝔉' in out
        assert 'OBSERVED' in out
        assert 'total_order' in r.sobol['mean_sobol']    # S_T still there, unrenamed

    def test_homoscedastic_summary_is_single_column(self, capsys):
        """𝔉≡1 fits keep a single-column conditional-inference display."""
        from hifi_anova.api import hifi_anova
        X, y = _fit_data(seed=2)
        r = hifi_anova(X, y, mode='second', verbose=False)
        r.summary()
        out = capsys.readouterr().out
        assert 'Sobol Indices (conditional intervals):' in out
        assert 'fitted-variance' not in out
        assert '(residual ĝ)' not in out
