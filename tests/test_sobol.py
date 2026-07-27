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
        G1=G1, G2=None,
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
            G1=G1, G2=G2,
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
            pair_indices=None, G1=G1, G2=None,
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
            pair_indices=None, G1=G1, G2=None,
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
            pair_indices=None, G1=G1, G2=None,
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
            pair_indices=None, G1=None, G2=None,
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
