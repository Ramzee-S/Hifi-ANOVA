"""Tests for Sobol index computation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hifi_anova.core.gram import build_gram_matrix, build_gram_matrix_2d
from hifi_anova.core.pairs import PairManager
from hifi_anova.model.mean_model import MeanModel
from hifi_anova.model.hifi_anova import HiFiANOVA
from hifi_anova.analysis.sobol import compute_sobol_indices

pytestmark = pytest.mark.smoke


def make_test_model(D=5, K1=5, K2=0, w1=None):
    """Helper to create a test model with known coefficients."""
    G1 = build_gram_matrix(K1)
    block = 2 * K1 + 1

    if w1 is None:
        w1 = jnp.zeros(D * block)

    mean_model = MeanModel(
        f0=jnp.array(0.0),
        w1=jnp.array(w1, dtype=jnp.float32),
        w2=jnp.array([], dtype=jnp.float32),
        K1=K1, K2=K2, D=D,
    )

    model = HiFiANOVA(
        mean_model=mean_model,
        K1=K1, K2=K2, Kh=0, D=D,
        pair_indices=None,
    )
    return model


class TestSobolIndices:
    def test_single_variable_linear(self):
        """One variable with linear coefficient only => S_i = 1."""
        D = 3
        K1 = 5
        block = 2 * K1 + 1

        w1 = np.zeros(D * block)
        w1[0] = 1.0  # Only variable 0, linear term

        model = make_test_model(D=D, K1=K1, w1=w1)
        results = compute_sobol_indices(model)

        # Variable 0 should have Sobol = 1.0
        assert abs(results['mean_sobol']['first_order'][0] - 1.0) < 1e-6
        # Others should be 0
        assert abs(results['mean_sobol']['first_order'][1]) < 1e-6
        assert abs(results['mean_sobol']['first_order'][2]) < 1e-6

    def test_two_variables_equal_contribution(self):
        """Two variables with same Var => S_i = 0.5 each."""
        D = 3
        K1 = 5
        block = 2 * K1 + 1

        w1 = np.zeros(D * block)
        # Variable 0: cos1 coefficient = 1 => Var = 0.5
        w1[1] = 1.0
        # Variable 1: cos1 coefficient = 1 => Var = 0.5
        w1[block + 1] = 1.0

        model = make_test_model(D=D, K1=K1, w1=w1)
        results = compute_sobol_indices(model)

        assert abs(results['mean_sobol']['first_order'][0] - 0.5) < 1e-6
        assert abs(results['mean_sobol']['first_order'][1] - 0.5) < 1e-6
        assert abs(results['mean_sobol']['first_order'][2]) < 1e-6

    def test_sum_to_one(self):
        """All Sobol indices should sum to 1."""
        D = 5
        K1 = 5
        block = 2 * K1 + 1

        # Random coefficients
        rng = np.random.RandomState(42)
        w1 = rng.randn(D * block) * 0.1

        model = make_test_model(D=D, K1=K1, w1=w1)
        results = compute_sobol_indices(model)

        total = sum(results['mean_sobol']['first_order'].values())
        assert abs(total - 1.0) < 1e-6

    def test_known_cross_term_variance(self):
        """Test variance computation with cross terms (linear + sin1)."""
        D = 2
        K1 = 5
        block = 2 * K1 + 1
        G1 = build_gram_matrix(K1)

        w1 = np.zeros(D * block)
        # Variable 0: linear=1, sin1=1
        w1[0] = 1.0
        w1[2] = 1.0

        # Expected variance: 1/12 + 1/2 + 2*(-1/(2*pi)) = 1/12 + 1/2 - 1/pi
        expected_var = 1.0/12.0 + 0.5 + 2.0*(-1.0/(2.0*np.pi))

        w_vec = jnp.array(w1[:block], dtype=jnp.float64)
        actual_var = float(w_vec @ jnp.asarray(G1, dtype=jnp.float64) @ w_vec)
        assert abs(actual_var - expected_var) < 1e-8

    def test_with_second_order(self):
        """Test Sobol with both first and second order terms."""
        D = 3
        K1 = 3
        K2 = 2
        block1 = 2 * K1 + 1
        block2 = (2 * K2 + 1) ** 2

        pm = PairManager(D)
        G1 = build_gram_matrix(K1)
        G2 = build_gram_matrix_2d(build_gram_matrix(K2))

        # First-order: only variable 0 active
        w1 = np.zeros(D * block1)
        w1[1] = 1.0  # cos1 for var 0

        # Second-order: pair (0,1) active
        w2 = np.zeros(pm.P * block2)
        # Set the (cos1, cos1) entry for pair 0
        w2[block2 // 2] = 1.0  # Some coefficient

        mean_model = MeanModel(
            f0=jnp.array(0.0),
            w1=jnp.array(w1, dtype=jnp.float32),
            w2=jnp.array(w2, dtype=jnp.float32),
            K1=K1, K2=K2, D=D,
        )

        model = HiFiANOVA(
            mean_model=mean_model,
            K1=K1, K2=K2, Kh=0, D=D,
            pair_indices=pm.pair_indices,
        )

        results = compute_sobol_indices(model)

        # Should have both first and second order indices
        assert len(results['mean_sobol']['first_order']) == D
        assert len(results['mean_sobol']['second_order']) > 0

        # Total should sum to ~1
        total = (sum(results['mean_sobol']['first_order'].values())
                 + sum(results['mean_sobol']['second_order'].values()))
        assert abs(total - 1.0) < 1e-4


class TestLegendreSobol:
    """Sobol indices with Legendre basis and analytic ground truth."""

    def test_legendre_single_variable(self):
        """One variable with Legendre coefficients => S = 1."""
        D, K1 = 3, 5
        G1 = build_gram_matrix(K1, basis_name='legendre')
        block = K1  # Legendre: K features

        w1 = np.zeros(D * block)
        w1[0] = 2.0  # Variable 0, degree 1 (P_tilde_1)

        mean_model = MeanModel(
            f0=jnp.array(0.0),
            w1=jnp.array(w1, dtype=jnp.float32),
            w2=jnp.array([], dtype=jnp.float32),
            K1=K1, K2=0, D=D,
            basis_name='legendre',
        )
        model = HiFiANOVA(
            mean_model=mean_model,
            K1=K1, K2=0, Kh=0, D=D,
            pair_indices=None,
            basis_name='legendre',
        )
        results = compute_sobol_indices(model)
        assert abs(results['mean_sobol']['first_order'][0] - 1.0) < 1e-6

    def test_legendre_known_variance(self):
        """Legendre Var(f_i) = sum(w_j^2 / (2j+3)) for j=0..K-1 (degree k=j+1).

        G[j,j] = 1/(2j+3): j=0 => 1/3, j=1 => 1/5, j=2 => 1/7, ...

        Variable 0: w[0]=3 (j=0, degree 1), w[1]=2 (j=1, degree 2)
          Var = 3^2/3 + 2^2/5 = 3.0 + 0.8 = 3.8
        Variable 1: w[0]=1 (j=0, degree 1)
          Var = 1^2/3 = 0.333...
        S0 = 3.8 / 4.133 = 0.9194...
        S1 = 0.333 / 4.133 = 0.0806...
        """
        D, K1 = 3, 5
        G1 = build_gram_matrix(K1, basis_name='legendre')
        block = K1

        w1 = np.zeros(D * block)
        w1[0] = 3.0  # var 0, j=0 (degree 1)
        w1[1] = 2.0  # var 0, j=1 (degree 2)
        w1[block] = 1.0  # var 1, j=0 (degree 1)

        var0 = 9.0/3.0 + 4.0/5.0   # 3.8
        var1 = 1.0/3.0              # 0.3333...
        total = var0 + var1
        S0_expected = var0 / total
        S1_expected = var1 / total

        # Verify via Gram matrix
        w0_vec = jnp.array(w1[:block], dtype=jnp.float64)
        G1_64 = jnp.asarray(G1, dtype=jnp.float64)
        var0_gram = float(w0_vec @ G1_64 @ w0_vec)
        assert abs(var0_gram - var0) < 1e-10, \
            f"Gram variance {var0_gram} != analytic {var0}"

        mean_model = MeanModel(
            f0=jnp.array(0.0),
            w1=jnp.array(w1, dtype=jnp.float32),
            w2=jnp.array([], dtype=jnp.float32),
            K1=K1, K2=0, D=D,
            basis_name='legendre',
        )
        model = HiFiANOVA(
            mean_model=mean_model,
            K1=K1, K2=0, Kh=0, D=D,
            pair_indices=None,
            basis_name='legendre',
        )
        results = compute_sobol_indices(model)
        assert abs(results['mean_sobol']['first_order'][0] - S0_expected) < 1e-5, \
            f"S0={results['mean_sobol']['first_order'][0]}, expected {S0_expected}"
        assert abs(results['mean_sobol']['first_order'][1] - S1_expected) < 1e-5, \
            f"S1={results['mean_sobol']['first_order'][1]}, expected {S1_expected}"
        assert abs(results['mean_sobol']['first_order'][2]) < 1e-10

    def test_legendre_sobol_from_fit(self):
        """Fit a polynomial function with Legendre basis, verify Sobol recovery."""
        from hifi_anova.core.features import build_first_order_features
        from hifi_anova.training.ridge import weighted_ridge_solve
        from hifi_anova.training.regularization import build_regularization_vector

        N, D = 10000, 3
        np.random.seed(42)
        x = jnp.array(np.random.uniform(0, 1, (N, D)))

        # f = 3*(2x0-1) + 2*(2x0-1)^2 - 1*(2x1-1)  (Legendre-native)
        # Var(f0) = 9/3 + 4/5 = 3.8  (degrees 1,2)
        # Var(f1) = 1/3  (degree 1)
        t0 = 2 * x[:, 0] - 1
        t1 = 2 * x[:, 1] - 1
        y = 3.0 * t0 + 2.0 * (1.5 * t0**2 - 0.5) - 1.0 * t1
        y = y - jnp.mean(y)

        phi = build_first_order_features(x, K=5, basis_name='legendre')
        reg = build_regularization_vector(D=D, K1=5, K2=0, P=0,
                                           strategy='uniform',
                                           lambda_order1=0.0001,
                                           basis_name='legendre')
        w = weighted_ridge_solve(phi, y, reg)
        G = build_gram_matrix(5, basis_name='legendre')
        block = 5

        vars_per = []
        for i in range(D):
            wi = jnp.asarray(w[i * block:(i + 1) * block], dtype=jnp.float64)
            G64 = jnp.asarray(G, dtype=jnp.float64)
            vars_per.append(float(wi @ G64 @ wi))
        total = sum(vars_per)
        sobol = [v / total for v in vars_per]

        # Analytic: S0 = 3.8/4.133 = 0.919, S1 = 0.333/4.133 = 0.081
        assert abs(sobol[0] - 0.919) < 0.03, f"S0={sobol[0]}"
        assert abs(sobol[1] - 0.081) < 0.03, f"S1={sobol[1]}"
        assert sobol[2] < 0.01, f"S2={sobol[2]}"


class TestHaarSobolAnalytic:
    """Sobol indices with Haar basis and tight analytic tolerances."""

    def test_haar_known_variance(self):
        """Haar Var(f_i) = sum(w_jk^2) since G=I.

        Variable 0: scale 1 coeff = 2.0 => Var = 4.0
        Variable 1: scale 1 coeff = 1.0 => Var = 1.0
        S0 = 0.8, S1 = 0.2.
        """
        D, K1 = 3, 3  # J=3, 7 features per var
        G1 = build_gram_matrix(K1, basis_name='haar')
        block = 2**K1 - 1  # = 7

        w1 = np.zeros(D * block)
        w1[0] = 2.0   # var 0, scale 1
        w1[block] = 1.0  # var 1, scale 1

        var0 = 4.0
        var1 = 1.0
        S0_expected = var0 / (var0 + var1)
        S1_expected = var1 / (var0 + var1)

        mean_model = MeanModel(
            f0=jnp.array(0.0),
            w1=jnp.array(w1, dtype=jnp.float32),
            w2=jnp.array([], dtype=jnp.float32),
            K1=K1, K2=0, D=D,
            basis_name='haar',
        )
        model = HiFiANOVA(
            mean_model=mean_model,
            K1=K1, K2=0, Kh=0, D=D,
            pair_indices=None,
            basis_name='haar',
        )
        results = compute_sobol_indices(model)
        assert abs(results['mean_sobol']['first_order'][0] - S0_expected) < 1e-6
        assert abs(results['mean_sobol']['first_order'][1] - S1_expected) < 1e-6
        assert abs(results['mean_sobol']['first_order'][2]) < 1e-10

    def test_haar_sobol_from_fit_tight(self):
        """Fit step function with Haar, verify Sobol with tight tolerance.

        y = 2*step(x0<0.5) - step(x1<0.5), all in Haar span.
        Var(f0) = 4, Var(f1) = 1, S0 = 0.8, S1 = 0.2.
        """
        from hifi_anova.core.features import build_first_order_features
        from hifi_anova.training.ridge import weighted_ridge_solve
        from hifi_anova.training.regularization import build_regularization_vector

        N, D = 10000, 3
        np.random.seed(42)
        x = jnp.array(np.random.uniform(0, 1, (N, D)))

        y = jnp.where(x[:, 0] < 0.5, 2.0, -2.0) + \
            jnp.where(x[:, 1] < 0.5, 1.0, -1.0)
        y = y - jnp.mean(y)

        phi = build_first_order_features(x, K=3, basis_name='haar')
        reg = build_regularization_vector(D=D, K1=3, K2=0, P=0,
                                           strategy='uniform',
                                           lambda_order1=0.0001,
                                           basis_name='haar')
        w = weighted_ridge_solve(phi, y, reg)
        G = build_gram_matrix(3, basis_name='haar')
        block = 7

        vars_per = []
        for i in range(D):
            wi = jnp.asarray(w[i * block:(i + 1) * block], dtype=jnp.float64)
            G64 = jnp.asarray(G, dtype=jnp.float64)
            vars_per.append(float(wi @ G64 @ wi))
        total = sum(vars_per)
        sobol = [v / total for v in vars_per]

        # Tight: exactly 4:1 ratio, Haar captures perfectly
        assert abs(sobol[0] - 0.8) < 0.02, f"S0={sobol[0]}, expected 0.8"
        assert abs(sobol[1] - 0.2) < 0.02, f"S1={sobol[1]}, expected 0.2"
        assert sobol[2] < 0.005, f"S2={sobol[2]}"


class TestMixedBasisSobol:
    """Sobol indices with mixed per-variable basis and ground truth."""

    def test_mixed_sobol_known_coefficients(self):
        """Mixed basis: var 0 Legendre, var 1 Haar, var 2 Fourier.

        var 0: Legendre K=3 (3 features), coeff[0]=3 => Var = 9*G[0,0] = 9/3 = 3.0
        var 1: Haar K=3 (7 features), coeff[0]=2 => Var = 4.0 (G=I)
        var 2: Fourier spectral K=3 (6 features), coeff[0]=1 => Var = 0.5 (G=1/2)
        Total = 7.5. S0=0.4, S1=0.5333, S2=0.0667.
        """
        from hifi_anova.core.gram import build_gram_matrix

        D = 3
        block_leg = 3   # Legendre: K=3
        block_haar = 7  # Haar: 2^3 - 1
        block_four = 6  # Fourier spectral only: 2K

        # var_specs format: (basis_name, K, include_linear, block_size, offset)
        var_specs = (
            ('legendre', 3, True, block_leg, 0),
            ('haar', 3, False, block_haar, block_leg),
            ('fourier', 3, False, block_four, block_leg + block_haar),
        )

        # Build Gram for each
        G_leg = build_gram_matrix(3, basis_name='legendre')
        G_haar = build_gram_matrix(3, basis_name='haar')
        G_four = build_gram_matrix(3, include_linear=False, basis_name='fourier')

        # Set coefficients
        w_leg = np.zeros(block_leg)
        w_leg[0] = 3.0  # degree 1
        w_haar = np.zeros(block_haar)
        w_haar[0] = 2.0  # scale 1
        w_four = np.zeros(block_four)
        w_four[0] = 1.0  # cos1

        w1 = np.concatenate([w_leg, w_haar, w_four])

        # Compute expected Sobol
        var0 = float(jnp.asarray(w_leg, dtype=jnp.float64) @
                      jnp.asarray(G_leg, dtype=jnp.float64) @
                      jnp.asarray(w_leg, dtype=jnp.float64))
        var1 = float(jnp.asarray(w_haar, dtype=jnp.float64) @
                      jnp.asarray(G_haar, dtype=jnp.float64) @
                      jnp.asarray(w_haar, dtype=jnp.float64))
        var2 = float(jnp.asarray(w_four, dtype=jnp.float64) @
                      jnp.asarray(G_four, dtype=jnp.float64) @
                      jnp.asarray(w_four, dtype=jnp.float64))
        total = var0 + var1 + var2

        mean_model = MeanModel(
            f0=jnp.array(0.0),
            w1=jnp.array(w1, dtype=jnp.float32),
            w2=jnp.array([], dtype=jnp.float32),
            K1=3, K2=0, D=D,
            basis_name='mixed',
            var_specs=var_specs,
        )
        model = HiFiANOVA(
            mean_model=mean_model,
            K1=3, K2=0, Kh=0, D=D,
            pair_indices=None,
            basis_name='mixed',
            var_specs=var_specs,
        )
        results = compute_sobol_indices(model)

        S0 = results['mean_sobol']['first_order'][0]
        S1 = results['mean_sobol']['first_order'][1]
        S2 = results['mean_sobol']['first_order'][2]
        assert abs(S0 - var0/total) < 1e-5, f"S0={S0}, expected {var0/total}"
        assert abs(S1 - var1/total) < 1e-5, f"S1={S1}, expected {var1/total}"
        assert abs(S2 - var2/total) < 1e-5, f"S2={S2}, expected {var2/total}"
        assert abs(S0 + S1 + S2 - 1.0) < 1e-6


class TestCorrelativeSobolSumToOne:
    """Correlative first-order indices sum to 1 identically (by linearity of
    covariance; Manuscript_Theoryv06 §correlative). This holds only when the
    Cov numerator and Var denominator use the same estimator — a regression
    guard for the ddof mismatch that used to inflate each index by N/(N-1)."""

    @staticmethod
    def _model(D=4, K1=4, seed=0):
        block = 2 * K1 + 1
        rng = np.random.default_rng(seed)
        w1 = rng.normal(size=D * block)            # nonzero components
        return make_test_model(D=D, K1=K1, w1=w1)

    @pytest.mark.parametrize("N", [40, 200, 1000])
    def test_sum_to_one_independent(self, N):
        from hifi_anova.analysis.sobol import compute_correlative_sobol
        model = self._model()
        rng = np.random.default_rng(1)
        x = jnp.asarray(rng.uniform(0.0, 1.0, size=(N, model.D)))
        out = compute_correlative_sobol(model, x)
        s = sum(out['first_order'].values())
        # Machine-precision identity — was N/(N-1) (e.g. ~1.026 at N=40) before
        # the estimator was unified.
        assert abs(s - 1.0) < 1e-9, f"N={N}: sum={s}"
        assert out['sum_of_correlative_indices'] == pytest.approx(1.0, abs=1e-9)

    def test_sum_to_one_correlated_inputs(self):
        """The identity is algebraic — it holds under correlated inputs too
        (where the individual shares diverge from the structural ones)."""
        from hifi_anova.analysis.sobol import compute_correlative_sobol
        model = self._model(D=3, K1=3, seed=2)
        rng = np.random.default_rng(3)
        z = rng.uniform(0.0, 1.0, size=(500, 1))
        # Strongly correlated columns.
        x = jnp.asarray(np.clip(
            z + 0.05 * rng.normal(size=(500, 3)), 0.0, 1.0))
        out = compute_correlative_sobol(model, x)
        assert abs(sum(out['first_order'].values()) - 1.0) < 1e-9

    def test_sum_to_one_float32_model(self):
        """A float32 fit (the DEC-035 default) must still sum to 1 to machine
        precision: the analytics cast component outputs to float64 like the
        structural path, so the identity does not inherit the fit's ~1e-7
        float32 round-off (regression guard — was ~6e-8 off before the cast)."""
        from hifi_anova.analysis.sobol import compute_correlative_sobol
        block = 2 * 4 + 1
        rng = np.random.default_rng(4)
        # float32 weights AND float32 inputs — the exact default-fit dtype.
        w1 = jnp.asarray(rng.normal(size=3 * block), dtype=jnp.float32)
        model = make_test_model(D=3, K1=4, w1=w1)
        x = jnp.asarray(rng.uniform(0.0, 1.0, size=(600, 3)), dtype=jnp.float32)
        out = compute_correlative_sobol(model, x)
        assert abs(sum(out['first_order'].values()) - 1.0) < 1e-9


def _pair_model(D=3, K1=3, K2=2, w1=None, w2=None):
    """Test model with a retained second-order block (all pairs of D vars)."""
    block1 = 2 * K1 + 1
    block2 = (2 * K2 + 1) ** 2
    pm = PairManager(D)
    if w1 is None:
        w1 = np.zeros(D * block1)
    if w2 is None:
        w2 = np.zeros(pm.P * block2)
    mean_model = MeanModel(
        f0=jnp.array(0.0),
        w1=jnp.array(w1, dtype=jnp.float32),
        w2=jnp.array(w2, dtype=jnp.float32),
        K1=K1, K2=K2, D=D,
    )
    model = HiFiANOVA(
        mean_model=mean_model,
        K1=K1, K2=K2, Kh=0, D=D,
        pair_indices=pm.pair_indices,
    )
    return model, pm, block1, block2


class TestCorrelativeAllComponents:
    """Correlative attribution ranges over ALL retained structured components
    (Manuscript_Theoryv06 §11.6): f_tot = Σ_{u≠∅} f̂_u over mains + pairs +
    triples. The complete collection sums to 1 identically; a first-order-only
    subset need not, when interactions are retained."""

    def test_pure_pair_component_carries_the_mass(self):
        """A pure pair model must attribute ~all mass to the pair component, not
        silently report first-order zeros summing to 0 (the old failure mode)."""
        from hifi_anova.analysis.sobol import compute_correlative_sobol
        D, K1, K2 = 3, 3, 2
        block2 = (2 * K2 + 1) ** 2
        w2 = np.zeros(3 * block2)          # PairManager(3).P == 3
        w2[block2 // 2] = 1.0              # activate pair 0 == (0, 1)
        model, pm, _, _ = _pair_model(D=D, K1=K1, K2=K2, w2=w2)
        rng = np.random.default_rng(0)
        x = jnp.asarray(rng.uniform(0.0, 1.0, size=(500, D)))
        out = compute_correlative_sobol(model, x)
        assert (0, 1) in out['second_order']
        assert out['second_order'][(0, 1)] == pytest.approx(1.0, abs=1e-9)
        assert out['first_order_sum'] == pytest.approx(0.0, abs=1e-9)
        assert out['sum_of_correlative_indices'] == pytest.approx(1.0, abs=1e-9)
        assert out['scope'] == 'all_retained_structured_components'
        assert out['residual_excluded'] is True

    def test_main_plus_pair_full_collection_sums_to_one(self):
        """With mains AND a pair retained, the complete collection sums to 1 but
        the first-order subset is a partial collection (< full)."""
        from hifi_anova.analysis.sobol import compute_correlative_sobol
        D, K1, K2 = 3, 3, 2
        block1 = 2 * K1 + 1
        block2 = (2 * K2 + 1) ** 2
        w1 = np.zeros(D * block1)
        w1[1] = 1.0                        # main effect on var 0
        w1[block1 + 1] = 0.7              # main effect on var 1
        w2 = np.zeros(3 * block2)
        w2[block2 // 2] = 0.8            # pair (0, 1)
        model, *_ = _pair_model(D=D, K1=K1, K2=K2, w1=w1, w2=w2)
        rng = np.random.default_rng(1)
        x = jnp.asarray(rng.uniform(0.0, 1.0, size=(800, D)))
        out = compute_correlative_sobol(model, x)
        assert out['sum_of_correlative_indices'] == pytest.approx(1.0, abs=1e-9)
        assert out['second_order']            # pair present
        # First-order-only subset is a strict partial collection here.
        assert out['first_order_sum'] < 1.0 - 1e-3
        full = (sum(out['first_order'].values())
                + sum(out['second_order'].values())
                + sum(out['third_order'].values()))
        assert full == pytest.approx(1.0, abs=1e-9)

    def test_diagnostic_no_false_do_not_sum_claim(self):
        """correlation_diagnostic on an interaction model flags higher-order
        structure and never claims the correlative indices 'do not sum to 1'."""
        from hifi_anova.analysis.diagnostics import correlation_diagnostic
        D, K1, K2 = 3, 3, 2
        block1 = 2 * K1 + 1
        block2 = (2 * K2 + 1) ** 2
        w1 = np.zeros(D * block1); w1[1] = 1.0
        w2 = np.zeros(3 * block2); w2[block2 // 2] = 0.9
        model, *_ = _pair_model(D=D, K1=K1, K2=K2, w1=w1, w2=w2)
        rng = np.random.default_rng(2)
        x = jnp.asarray(rng.uniform(0.0, 1.0, size=(600, D)))
        diag = correlation_diagnostic(model, x)
        assert diag['has_higher_order'] is True
        assert diag['sum_correlative'] == pytest.approx(1.0, abs=1e-9)
        assert 'do not sum to 1' not in diag['recommendation'].lower()

    def test_degenerate_zero_variance(self):
        """A zero model (all weights 0) has Var(f_tot)=0: shares are zero-filled
        and the collection sum is 0 (distinct from the sum-to-1 identity)."""
        from hifi_anova.analysis.sobol import compute_correlative_sobol
        model = make_test_model(D=3, K1=3)              # w1 defaults to zeros
        rng = np.random.default_rng(5)
        x = jnp.asarray(rng.uniform(0.0, 1.0, size=(200, 3)))
        out = compute_correlative_sobol(model, x)
        assert out['sum_of_correlative_indices'] == pytest.approx(0.0, abs=1e-12)
        assert all(v == 0.0 for v in out['first_order'].values())

    def test_negative_share_under_correlation(self):
        """Individual correlative shares may be negative under dependence, while
        the complete collection still sums to 1 (Manuscript_Theoryv06 §11.6).
        Construct perfect anti-correlation: x1 = 1 - x0 with sin1 components, so
        f_1 = -2·f_0 and S_0 = 1/(1-2) = -1, S_1 = 2."""
        from hifi_anova.analysis.sobol import compute_correlative_sobol
        D, K1 = 2, 3
        block = 2 * K1 + 1
        w1 = np.zeros(D * block)
        w1[2] = 1.0                       # var 0: sin1 coeff = 1
        w1[block + 2] = 2.0             # var 1: sin1 coeff = 2
        model = make_test_model(D=D, K1=K1, w1=w1)
        rng = np.random.default_rng(6)
        u = rng.uniform(0.0, 1.0, size=(500, 1))
        x = jnp.asarray(np.hstack([u, 1.0 - u]))   # perfectly anti-correlated
        out = compute_correlative_sobol(model, x)
        assert min(out['first_order'].values()) < -0.5    # a negative share
        assert out['sum_of_correlative_indices'] == pytest.approx(1.0, abs=1e-9)

    def test_third_order_component_allocation(self):
        """A retained triple gets its own correlative share; mains + triple form
        the complete collection and sum to 1."""
        from hifi_anova.analysis.sobol import compute_correlative_sobol
        D, K1, K3 = 3, 2, 2
        block1 = 2 * K1 + 1
        block3 = (2 * K3 + 1) ** 3
        w1 = np.zeros(D * block1)
        w1[1] = 1.0                       # a main effect on var 0
        w3 = np.zeros(block3)
        w3[block3 // 2] = 1.0           # activate the (0,1,2) triple block
        mean_model = MeanModel(
            f0=jnp.array(0.0),
            w1=jnp.array(w1, dtype=jnp.float32),
            w2=jnp.array([], dtype=jnp.float32),
            w3=jnp.array(w3, dtype=jnp.float32),
            K1=K1, K2=0, K3=K3, D=D,
        )
        model = HiFiANOVA(
            mean_model=mean_model, K1=K1, K2=0, K3=K3, Kh=0, D=D,
            pair_indices=None, triple_indices=jnp.array([[0, 1, 2]]),
        )
        rng = np.random.default_rng(7)
        x = jnp.asarray(rng.uniform(0.0, 1.0, size=(600, D)))
        out = compute_correlative_sobol(model, x)
        assert (0, 1, 2) in out['third_order']
        assert out['third_order'][(0, 1, 2)] != 0.0
        assert out['sum_of_correlative_indices'] == pytest.approx(1.0, abs=1e-9)

    def test_compute_sobol_indices_attaches_full_correlative(self):
        """Automatic assembly attaches the complete correlative collection (all
        orders, sum 1) for an interaction model — never a misleading partial."""
        D, K1, K2 = 3, 3, 2
        block1 = 2 * K1 + 1
        block2 = (2 * K2 + 1) ** 2
        w1 = np.zeros(D * block1); w1[1] = 1.0
        w2 = np.zeros(3 * block2); w2[block2 // 2] = 0.8
        model, *_ = _pair_model(D=D, K1=K1, K2=K2, w1=w1, w2=w2)
        rng = np.random.default_rng(8)
        x = jnp.asarray(rng.uniform(0.0, 1.0, size=(500, D)))
        res = compute_sobol_indices(model, x)
        cs = res['correlative_sobol']
        assert cs['scope'] == 'all_retained_structured_components'
        assert cs['second_order']
        assert cs['sum_of_correlative_indices'] == pytest.approx(1.0, abs=1e-9)

    def test_verify_model_reports_descriptive_input_correlation(self):
        """verify_model's 'Input independence' info reports the descriptive max
        ordinary |Pearson| when x_train is given (x0 = 1-x1 → ~1.00), without
        running the experimental nonlinear test."""
        from hifi_anova.analysis.diagnostics import verify_model
        D, K1, K2 = 3, 3, 2
        block1 = 2 * K1 + 1
        block2 = (2 * K2 + 1) ** 2
        w1 = np.zeros(D * block1); w1[2] = 1.0; w1[block1 + 2] = 1.0
        w2 = np.zeros(3 * block2); w2[block2 // 2] = 0.8
        model, *_ = _pair_model(D=D, K1=K1, K2=K2, w1=w1, w2=w2)
        rng = np.random.default_rng(9)
        u = rng.uniform(0.0, 1.0, size=(400, 1))
        x = np.hstack([u, 1.0 - u, rng.uniform(0.0, 1.0, size=(400, 1))])
        y = np.asarray(model.predict_mean_only(jnp.asarray(x)))
        rep = verify_model(model, jnp.asarray(x), y, x_train=jnp.asarray(x),
                           verbose=False)
        chk = [c for c in rep['checks'] if c['name'] == 'Input independence']
        assert chk and chk[0]['status'] == 'info'
        assert '1.00' in chk[0]['detail']    # descriptive max |Pearson| ~ 1.0


class TestCorrelativeTrainedIntegration:
    """At least one trained-model integration test: capability metadata and the
    sum-to-1 identity must hold against REAL fitted structure, not just
    hand-built coefficient vectors."""

    def test_trained_second_order_model(self):
        from hifi_anova.api import hifi_anova
        from hifi_anova.analysis.sobol import compute_correlative_sobol
        rng = np.random.default_rng(0)
        N = 400
        X = rng.uniform(0.0, 1.0, size=(N, 3))
        # Main effect on x0 + a genuine x0*x1 interaction.
        y = (np.sin(2 * np.pi * X[:, 0]) + 1.5 * X[:, 0] * X[:, 1]
             + 0.05 * rng.normal(size=N))
        res = hifi_anova(X, y, K1=4, K2=3, mode='second',
                         variable_selection=None, seed=0, verbose=False)
        model = res.model
        xt = jnp.asarray(np.asarray(res._data['x_train']), dtype=jnp.float64)
        out = compute_correlative_sobol(model, xt)
        assert out['scope'] == 'all_retained_structured_components'
        assert len(out['first_order']) == model.D
        assert out['sum_of_correlative_indices'] == pytest.approx(1.0, abs=1e-6)
        # Component keys must match the model's actually-retained structure.
        if model.pair_indices is not None and out['second_order']:
            retained = {(int(a), int(b)) for a, b in np.asarray(model.pair_indices)}
            assert set(out['second_order']).issubset(retained)

    def test_residual_model_excludes_residual_from_denominator(self):
        """A fitted Stage-C residual model: the correlative shares must use only
        the STRUCTURED prediction (residual excluded from f_tot / denominator),
        while fidelity 𝔉 remains available and < 1 because the residual carries
        genuine variance (reviewer: explicit residual-contract coverage)."""
        from hifi_anova.api import hifi_anova
        from hifi_anova.analysis.sobol import (compute_correlative_sobol,
                                               compute_sobol_indices)
        # Structured signal + a 3-D radial bump the <=2nd-order basis cannot
        # represent, so a Stage-C residual reliably has genuine variance.
        rng = np.random.default_rng(0)
        N, D = 500, 3
        X = rng.uniform(0.0, 1.0, (N, D))
        struct = np.sin(2 * np.pi * X[:, 0]) + 0.6 * X[:, 1]
        bump = 2.0 * np.exp(-8.0 * ((X[:, 0] - 0.5) ** 2 + (X[:, 1] - 0.5) ** 2
                                    + (X[:, 2] - 0.5) ** 2))
        y = struct + bump + 0.03 * rng.standard_normal(N)
        res = hifi_anova(X, y, mode='second', residual='rbf', verbose=False)
        model = res.model
        assert model.residual_net is not None            # residual genuinely present
        xt = jnp.asarray(np.asarray(res._data['x_train']), dtype=jnp.float64)

        # (3) Fidelity available and strictly below 1 (residual carries variance).
        sob = compute_sobol_indices(model, xt)
        assert sob['variance_accounting']['residual'] > 0.0
        assert 0.0 < sob['fidelity']['value'] < 1.0

        # (1)+(2) Correlative shares use the structured prediction only, and the
        # residual is excluded from the denominator.
        out = compute_correlative_sobol(model, xt)
        assert out['residual_excluded'] is True
        # Structured-only closure: shares sum to 1 ⇒ denominator = Var(structured
        # sum), not Var(structured + residual).
        assert out['sum_of_correlative_indices'] == pytest.approx(1.0, abs=1e-6)

        # Exact proof the share uses the STRUCTURED prediction and excludes the
        # residual from the denominator — robust to the (small) residual size.
        # Reconstruct component 0's share two ways and show the code uses the
        # structured-sum variance, NOT the residual-inclusive variance.
        from hifi_anova.analysis.sobol import _mean_component_outputs
        orders, keys, comp_out = _mean_component_outputs(model, xt)
        comp_out = np.asarray(comp_out, dtype=np.float64)
        tot_c = comp_out.sum(0)
        tot_c = tot_c - tot_c.mean()                     # centered structured f_tot
        struct_denom = float(np.mean(tot_c * tot_c))     # Var(Σ structured f_u)
        full_pred = np.asarray(model.predict_mean_only(xt), dtype=np.float64)
        full_c = full_pred - full_pred.mean()            # structured + residual
        full_denom = float(np.mean(full_c * full_c))
        assert abs(full_denom - struct_denom) > 1e-8     # residual changes the total
        idx0 = next(n for n, (o, k) in enumerate(zip(orders, keys))
                    if o == 1 and k == 0)
        f0c = comp_out[idx0] - comp_out[idx0].mean()
        cov0 = float(np.mean(f0c * tot_c))               # numerator: structured only
        assert out['first_order'][0] == pytest.approx(cov0 / struct_denom, abs=1e-9)
        assert abs(out['first_order'][0] - cov0 / full_denom) > 1e-6


class TestIndependenceAssumptionDiagnostic:
    """The correlative path is an independence-assumption diagnostic, not an
    official correlated-attribution estimand; its gate must catch nonlinear
    (uncorrelated-but-dependent) dependence, not only linear correlation."""

    def test_role_metadata(self):
        from hifi_anova.analysis.sobol import compute_correlative_sobol
        model = make_test_model(D=3, K1=3,
                                w1=np.random.default_rng(0).normal(size=3 * 7))
        rng = np.random.default_rng(1)
        x = jnp.asarray(rng.uniform(0.0, 1.0, size=(200, 3)))
        out = compute_correlative_sobol(model, x)
        assert out['role'] == 'independence_assumption_diagnostic'
        assert out['official_correlated_estimand'] is False

    def test_default_is_descriptive_no_auto_test(self):
        """By default correlation_diagnostic is DESCRIPTIVE: it records the
        independence assumption (unverified), a descriptive Pearson level, and does
        NOT run the experimental nonlinear test."""
        from hifi_anova.analysis.diagnostics import correlation_diagnostic
        model = make_test_model(D=3, K1=3,
                                w1=np.random.default_rng(2).normal(size=3 * 7))
        rng = np.random.default_rng(3)
        x = jnp.asarray(rng.uniform(0.0, 1.0, size=(1500, 3)))   # independent
        diag = correlation_diagnostic(model, x)
        assert diag['input_assumption'] == 'independent_product_measure'
        assert diag['input_assumption_verified'] is False
        assert diag['independence_test'] is None            # not auto-run
        assert 'dependence_level' not in diag              # no verdict by default
        assert diag['correlation_level'] == 'clean'         # descriptive Pearson
        assert diag['role'] == 'independence_assumption_diagnostic'

    def test_inputs_independent_by_design_recorded(self):
        from hifi_anova.analysis.diagnostics import correlation_diagnostic
        from hifi_anova.analysis.sobol import compute_sobol_indices
        model = make_test_model(D=3, K1=3,
                                w1=np.random.default_rng(2).normal(size=3 * 7))
        x = jnp.asarray(np.random.default_rng(3).uniform(0.0, 1.0, size=(300, 3)))
        diag = correlation_diagnostic(model, x, inputs_independent_by_design=True)
        assert diag['input_assumption_verified'] is True
        sob = compute_sobol_indices(model, x, inputs_independent_by_design=True)
        assert sob['input_assumption'] == 'independent_product_measure'
        assert sob['input_assumption_verified'] is True
        # Default (no assertion) stays unverified.
        assert compute_sobol_indices(model, x)['input_assumption_verified'] is False

    def test_nonlinear_dependence_caught_only_when_opted_in(self):
        """x1 = 2|x0-0.5| has ~zero linear correlation but genuine nonlinear
        dependence. The DEFAULT descriptive linear level misses it ('clean'); the
        opt-in experimental test catches it as the nonlinear-dominant channel."""
        from hifi_anova.analysis.diagnostics import correlation_diagnostic
        model = make_test_model(D=2, K1=3,
                                w1=np.random.default_rng(4).normal(size=2 * 7))
        rng = np.random.default_rng(5)
        u = rng.uniform(0.0, 1.0, size=(1200, 1))
        x = jnp.asarray(np.hstack([u, 2.0 * np.abs(u - 0.5)]))
        # Default: descriptive linear only -> reads clean (documents the blind spot).
        base = correlation_diagnostic(model, x)
        assert base['correlation_level'] == 'clean'
        assert base['independence_test'] is None
        # Opt-in experimental nonlinear test catches it.
        diag = correlation_diagnostic(model, x, run_independence_test=True)
        assert diag['max_abs_input_correlation'] < 0.15
        assert diag['nonlinear_significant'] is True
        assert diag['linear_significant'] is False
        assert diag['dependence_level'] in ('mild', 'strong')
        assert diag['nonlinear_dominant'] is True
        assert 'nonlinear' in diag['recommendation'].lower()

    def test_pure_interaction_strong_correlation_not_called_clean(self):
        """Reviewer regression: a pure-pair model (zero first-order coeffs) with
        corr(x0,x1) = -1. The descriptive Pearson level catches the LINEAR
        dependence ('strong'); the opt-in test confirms linear significance."""
        from hifi_anova.analysis.diagnostics import correlation_diagnostic
        K2 = 2
        block2 = (2 * K2 + 1) ** 2
        w2 = np.zeros(1 * block2)         # PairManager(2).P == 1 → pair (0,1)
        w2[block2 // 2] = 1.0
        model, *_ = _pair_model(D=2, K1=3, K2=K2, w2=w2)
        rng = np.random.default_rng(10)
        u = rng.uniform(0.0, 1.0, size=(600, 1))
        x = np.hstack([u, 1.0 - u])       # corr(x0, x1) = -1
        diag = correlation_diagnostic(model, jnp.asarray(x))
        assert diag['max_abs_input_correlation'] > 0.99
        assert diag['correlation_level'] == 'strong'   # descriptive Pearson catches it
        # The component-output metric is fooled (0) — documents the old bug.
        assert diag['component_output_correlation_level'] == 'clean'
        assert 'negligible' not in diag['recommendation'].lower()
        diag2 = correlation_diagnostic(model, jnp.asarray(x),
                                       run_independence_test=True)
        assert diag2['linear_significant'] is True
        assert diag2['dependence_level'] == 'strong'

    def test_verify_model_does_not_auto_test_independence(self):
        """verify_model records the independence assumption as INFO (unverified),
        never runs the nonlinear test, and exposes the assumption metadata."""
        from hifi_anova.analysis.diagnostics import verify_model
        model = make_test_model(D=3, K1=3,
                                w1=np.random.default_rng(11).normal(size=3 * 7))
        rng = np.random.default_rng(12)
        x = jnp.asarray(rng.uniform(0.0, 1.0, size=(400, 3)))
        y = np.asarray(model.predict_mean_only(x))
        rep = verify_model(model, x, y, x_train=x, verbose=False)
        assert rep['input_assumption'] == 'independent_product_measure'
        assert rep['input_assumption_verified'] is False
        chk = [c for c in rep['checks'] if c['name'] == 'Input independence']
        assert chk and chk[0]['status'] == 'info'
        assert 'Pearson' in chk[0]['detail']              # descriptive only
        rep2 = verify_model(model, x, y, x_train=x, verbose=False,
                            inputs_independent_by_design=True)
        assert rep2['input_assumption_verified'] is True

    @pytest.mark.parametrize("n,D", [(25, 3), (50, 3), (100, 3), (80, 6)])
    def test_independent_null_false_positive_rate(self, n, D):
        """Reviewer regression: genuinely independent uniform inputs must NOT be
        flagged as dependent across sample sizes and dimensions. With the biased
        distance correlation vs fixed thresholds this hit ~100%; the unbiased
        estimator + max-statistic permutation test must control it near the 5%
        nominal level. We assert the not-'clean' rate stays low over many seeds."""
        from hifi_anova.analysis.diagnostics import _input_dependence
        K = 40
        not_clean = 0
        for s in range(K):
            x = np.random.default_rng(1000 + s).uniform(0.0, 1.0, size=(n, D))
            dep = _input_dependence(x, seed=s)
            if dep['dependence_level'] != 'clean':
                not_clean += 1
        rate = not_clean / K
        # Nominal FWER is 0.05; allow generous slack for the finite null sample.
        assert rate <= 0.15, f"n={n} D={D}: false-positive rate {rate:.2%}"

    def test_full_covariance_is_opt_in(self):
        """The O(C^2) component covariance matrix is not built by default."""
        from hifi_anova.analysis.sobol import compute_correlative_sobol
        model, *_ = _pair_model(D=3, K1=3, K2=2,
                                w1=np.random.default_rng(0).normal(size=3 * 7))
        rng = np.random.default_rng(1)
        x = jnp.asarray(rng.uniform(0.0, 1.0, size=(200, 3)))
        out = compute_correlative_sobol(model, x)
        assert 'component_covariance_matrix' not in out
        out2 = compute_correlative_sobol(model, x, return_full_covariance=True)
        C = len(out2['component_keys'])
        assert out2['component_covariance_matrix'].shape == (C, C)
