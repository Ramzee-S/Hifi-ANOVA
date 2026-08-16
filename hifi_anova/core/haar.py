"""Haar wavelet basis on [0,1] for the Hoeffding decomposition.

The Haar system detects LOCALIZED effects — step changes, threshold effects,
regime boundaries — that global bases (Fourier, Legendre) approximate poorly.

Wavelet system:
  Scale j=1: 1 function  (half-domain contrast)
  Scale j=2: 2 functions (quarter-domain contrasts)
  Scale j=J: 2^{j-1} functions at resolution 2^{-j+1}
  Total through scale J: 2^J - 1 basis functions.

General formula:
  psi_{j,k}(x) = 2^{(j-1)/2} * psi(2^{j-1} x - k)
  where psi(t) = +1 on [0, 0.5), -1 on [0.5, 1), 0 elsewhere
  k = 0, 1, ..., 2^{j-1}-1

Key properties:
  - Vanishing integral: int_0^1 psi_{j,k}(x) dx = 0 (Hoeffding condition)
  - Orthonormality: Gram matrix = Identity
  - Sobol: Var(f_i) = sum w_{j,k}^2 (just sum of squared coefficients)
  - Piecewise constant: no classical derivatives, use Besov penalty instead
"""

from ..array_backend import xp as jnp  # switchable array backend (numpy exact core)
from typing import Tuple


class HaarBasis:
    """Haar wavelet basis on [0,1].

    Implements the same interface pattern as the Fourier and Legendre bases
    in features.py, but as a standalone class with additional interpretation
    methods (scale_of_index, position_of_index).

    Properties used by the framework:
      - vanishing_integral: True
      - gram_diagonal: True (Gram = Identity)
      - includes_linear: False
      - derivative_penalty_available: False (use scale-based Besov penalty)
    """

    def __init__(self, max_scale: int = 4):
        """
        Args:
            max_scale: J. Gives 2^J - 1 basis functions per variable.
                J=3: 7 functions (coarse)
                J=4: 15 functions (moderate)
                J=5: 31 functions (fine)
                J=6: 63 functions (very fine)
        """
        self.J = max_scale
        self.n_basis = 2 ** max_scale - 1

    @property
    def includes_linear(self) -> bool:
        return False

    @property
    def gram_is_identity(self) -> bool:
        return True

    def evaluate(self, x: jnp.ndarray, max_scale: int = None) -> jnp.ndarray:
        """Evaluate all Haar wavelets for a single variable.

        Args:
            x: (N,) array of values in [0, 1].
            max_scale: override self.J if desired.

        Returns:
            (N, 2^J - 1) array of Haar wavelet evaluations.
        """
        J = self.J if max_scale is None else max_scale
        features = []

        for j in range(1, J + 1):
            scale_factor = 2.0 ** ((j - 1) / 2.0)
            n_positions = 2 ** (j - 1)
            interval_width = 1.0 / n_positions

            for k in range(n_positions):
                left = k * interval_width
                mid = left + interval_width / 2.0
                right = left + interval_width

                # Rightmost cell is closed at right == 1.0 so x == 1.0 (the
                # max sample after min-max scaling) doesn't produce an
                # all-zero Haar row. Matches _build_haar_basis in features.py.
                if k == n_positions - 1:
                    in_right_half = (x >= mid) & (x <= right)
                else:
                    in_right_half = (x >= mid) & (x < right)
                psi = jnp.where(
                    (x >= left) & (x < mid),
                    scale_factor,
                    jnp.where(in_right_half, -scale_factor, 0.0)
                )
                features.append(psi)

        return jnp.stack(features, axis=-1)  # (N, 2^J - 1)

    def evaluate_batch(self, x: jnp.ndarray,
                       max_scale: int = None) -> jnp.ndarray:
        """Evaluate Haar basis for all D variables.

        Args:
            x: (N, D) array.
            max_scale: override self.J.

        Returns:
            (N, D, n_basis) per-variable basis array.
        """
        J = self.J if max_scale is None else max_scale
        N, D = x.shape
        # Evaluate each variable
        blocks = [self.evaluate(x[:, i], J) for i in range(D)]
        return jnp.stack(blocks, axis=1)  # (N, D, n_basis)

    def gram_matrix(self, max_scale: int = None) -> jnp.ndarray:
        """Gram matrix = Identity (orthonormal basis)."""
        n = 2 ** (self.J if max_scale is None else max_scale) - 1
        return jnp.eye(n, dtype=jnp.float64)

    def complexity_weights(self, max_scale: int = None,
                           penalty_exponent: float = 2.0) -> jnp.ndarray:
        """Per-basis-function complexity weights for regularization.

        Uses scale-based (Besov norm) penalty: weight = 4^{alpha * j}
        where alpha = penalty_exponent.

        penalty_exponent=1: mild scale penalty (allows fine-scale features)
        penalty_exponent=2: strong scale penalty (suppresses fine-scale)

        This is the Besov-norm regularization B^alpha_{2,2}, the correct
        smoothness measure for piecewise-constant function spaces.
        """
        J = self.J if max_scale is None else max_scale
        weights = []
        for j in range(1, J + 1):
            n_at_scale = 2 ** (j - 1)
            w = 4.0 ** (penalty_exponent * j)
            weights.extend([w] * n_at_scale)
        return jnp.array(weights, dtype=jnp.float64)

    def scale_of_index(self, idx: int) -> int:
        """Return the scale j for basis function at index idx."""
        cumulative = 0
        for j in range(1, self.J + 1):
            n_at_scale = 2 ** (j - 1)
            if idx < cumulative + n_at_scale:
                return j
            cumulative += n_at_scale
        raise IndexError(f"Index {idx} out of range for J={self.J}")

    def position_of_index(self, idx: int) -> Tuple[int, int, float, float]:
        """Return (scale j, position k, interval_start, interval_end)
        for basis function at index idx.

        Useful for reporting WHERE a localized feature was detected.
        """
        cumulative = 0
        for j in range(1, self.J + 1):
            n_at_scale = 2 ** (j - 1)
            if idx < cumulative + n_at_scale:
                k = idx - cumulative
                interval_width = 1.0 / n_at_scale
                return (j, k, k * interval_width, (k + 1) * interval_width)
            cumulative += n_at_scale
        raise IndexError(f"Index {idx} out of range for J={self.J}")

    def scale_slice(self, j: int) -> slice:
        """Return the slice into the coefficient vector for scale j.

        Scale j has 2^{j-1} coefficients starting at index 2^{j-1} - 1.
        """
        start = 2 ** (j - 1) - 1
        end = 2 ** j - 1
        return slice(start, end)


def build_haar_features(x: jnp.ndarray, J: int) -> jnp.ndarray:
    """Convenience function matching the build_first_order_features signature.

    Args:
        x: (N, D) in [0, 1]
        J: max wavelet scale (mapped from K parameter in the framework)

    Returns:
        (N, D * (2^J - 1)) feature matrix
    """
    basis = HaarBasis(J)
    per_var = basis.evaluate_batch(x, J)  # (N, D, n_basis)
    N = x.shape[0]
    return per_var.reshape(N, -1)
