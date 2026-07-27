"""Precomputed matrices encoding the geometry of the basis functions.

Built once per K value. Tiny matrices - no performance concerns.
All functions are pure: no state, no side effects, easily testable.

Three basis families:
  basis_name='fourier': lin-sin cross-terms, (2K+1)×(2K+1) or (2K)×(2K)
  basis_name='legendre': perfectly diagonal, K×K
  basis_name='haar': identity matrix, (2^K-1)×(2^K-1)
"""

import jax.numpy as jnp
import numpy as np


def build_gram_matrix(K: int, include_linear: bool = True,
                       basis_name: str = 'fourier') -> jnp.ndarray:
    """Gram matrix G encoding basis inner products on [0,1].

    Args:
        K: max harmonic/degree.
        include_linear: for Fourier, whether to include linear term.
        basis_name: 'fourier' or 'legendre'

    Returns:
        G of shape (B, B) where B = basis_size(K, include_linear, basis_name).

    Fourier full: G[0,0]=1/12, G[2k-1,2k-1]=G[2k,2k]=1/2, G[0,2k]=-1/(2πk)
    Fourier spectral: diagonal, all 1/2
    Legendre: diagonal, G[j,j] = 1/(2(j+1)+1) = 1/(2j+3) for j=0..K-1
    """
    if basis_name == 'haar':
        return _build_gram_haar(K)
    if basis_name == 'legendre':
        return _build_gram_legendre(K)

    # Fourier
    if not include_linear:
        size = 2 * K
        if size == 0:
            return jnp.zeros((0, 0), dtype=jnp.float64)
        return jnp.eye(size, dtype=jnp.float64) * 0.5

    size = 2 * K + 1
    G = np.zeros((size, size), dtype=np.float64)
    G[0, 0] = 1.0 / 12.0
    for k in range(1, K + 1):
        G[2 * k - 1, 2 * k - 1] = 0.5
        G[2 * k, 2 * k] = 0.5
        G[0, 2 * k] = -1.0 / (2.0 * np.pi * k)
        G[2 * k, 0] = -1.0 / (2.0 * np.pi * k)
    return jnp.array(G)


def _build_gram_legendre(K: int) -> jnp.ndarray:
    """Gram matrix for shifted Legendre polynomials P̃₁..P̃ₖ on [0,1].

    Perfectly diagonal: G[j,j] = ∫₀¹ P̃_{j+1}(x)² dx = 1/(2(j+1)+1)

    For shifted Legendre P̃ₖ(x) = Pₖ(2x-1):
    ∫₀¹ P̃ₖ² dx = ½ ∫₋₁¹ Pₖ² dt = ½ · 2/(2k+1) = 1/(2k+1)
    """
    if K == 0:
        return jnp.zeros((0, 0), dtype=jnp.float64)
    diag = np.array([1.0 / (2 * (j + 1) + 1) for j in range(K)], dtype=np.float64)
    return jnp.diag(jnp.array(diag))


def _build_gram_haar(J: int) -> jnp.ndarray:
    """Gram matrix for Haar wavelets: Identity.

    Haar wavelets are orthonormal on [0,1], so G = I.
    J is the max scale (K parameter in the framework).
    Size: (2^J - 1) x (2^J - 1).
    """
    if J == 0:
        return jnp.zeros((0, 0), dtype=jnp.float64)
    n = 2 ** J - 1
    return jnp.eye(n, dtype=jnp.float64)


def build_gram_matrix_2d(G1: jnp.ndarray) -> jnp.ndarray:
    """Second-order Gram: G ⊗ G (Kronecker product)."""
    return jnp.kron(G1, G1)


def build_gram_matrix_3d(G1: jnp.ndarray) -> jnp.ndarray:
    """Third-order Gram: G ⊗ G ⊗ G (triple Kronecker product)."""
    return jnp.kron(G1, jnp.kron(G1, G1))


def build_derivative_penalty(K: int, p: int = 2,
                              include_linear: bool = True,
                              basis_name: str = 'fourier') -> jnp.ndarray:
    """Diagonal penalty vector for integrated squared p-th derivative.

    Args:
        K: max harmonic/degree.
        p: derivative order (1=smoothness, 2=curvature).
        include_linear: for Fourier, whether to include linear entry.
        basis_name: 'fourier' or 'legendre'

    Returns:
        D of shape (B,) where B = basis_size(K, include_linear, basis_name).
    """
    if basis_name == 'haar':
        return _build_derivative_penalty_haar(K, p)
    if basis_name == 'legendre':
        return _build_derivative_penalty_legendre(K, p)

    # Fourier
    if not include_linear:
        size = 2 * K
        if size == 0:
            return jnp.zeros(0, dtype=jnp.float64)
        D = np.zeros(size, dtype=np.float64)
        for k in range(1, K + 1):
            val = (2.0 * np.pi * k) ** (2 * p) / 2.0
            D[2 * k - 2] = val
            D[2 * k - 1] = val
        return jnp.array(D)

    size = 2 * K + 1
    D = np.zeros(size, dtype=np.float64)
    if p == 1:
        D[0] = 1.0
    for k in range(1, K + 1):
        val = (2.0 * np.pi * k) ** (2 * p) / 2.0
        D[2 * k - 1] = val
        D[2 * k] = val
    return jnp.array(D)


def _build_derivative_penalty_legendre(K: int, p: int) -> jnp.ndarray:
    """Integrated squared p-th derivative for shifted Legendre on [0,1].

    D[j] = ∫₀¹ [(d/dx)ᵖ P̃_{j+1}(x)]² dx

    Uses numpy.polynomial.legendre for exact computation.
    For p=1: D[j] = 2(j+1)(j+2) (analytically known).
    For p≥2: computed via polynomial arithmetic (exact for polynomials).
    """
    if K == 0:
        return jnp.zeros(0, dtype=jnp.float64)

    D = np.zeros(K, dtype=np.float64)

    if p == 1:
        # Analytical: D₁[j] = 2·k·(k+1) where k=j+1
        for j in range(K):
            k = j + 1
            D[j] = 2.0 * k * (k + 1)
        return jnp.array(D)

    # General p: use numpy.polynomial.legendre
    import numpy.polynomial.legendre as leg
    for j in range(K):
        k = j + 1  # polynomial degree
        if k < p:
            D[j] = 0.0
            continue
        # Legendre coefficients for Pₖ: unit vector at position k
        c = np.zeros(k + 1)
        c[k] = 1.0
        # Take p-th derivative (in t ∈ [-1,1])
        c_deriv = c
        for _ in range(p):
            c_deriv = leg.legder(c_deriv)
        # ||c_deriv||²_{L²[-1,1]} via Legendre orthogonality:
        # ∫₋₁¹ [Σ aₘ Pₘ(t)]² dt = Σ aₘ² · 2/(2m+1)
        norm_sq = sum(c_deriv[m] ** 2 * 2.0 / (2 * m + 1)
                      for m in range(len(c_deriv)))
        # Chain rule (x→t: dx=dt/2) and interval mapping:
        # D[j] = 2^(2p-1) · ||Pₖ^(p)||²_{[-1,1]}
        D[j] = 2.0 ** (2 * p - 1) * norm_sq

    return jnp.array(D)


def _build_derivative_penalty_haar(J: int, p: int) -> jnp.ndarray:
    """Scale-based Besov penalty for Haar wavelets.

    Haar wavelets are piecewise constant — classical derivatives are zero
    everywhere except at jump points. The Besov norm B^alpha_{2,2} is the
    correct smoothness measure. We map derivative order p to Besov exponent:
      p=1 (smoothness) → alpha=1: weight = 4^j
      p=2 (curvature)  → alpha=2: weight = 4^{2j} = 16^j

    This gives analogous behavior to the Fourier/Legendre derivative penalties:
    higher p suppresses fine-scale (high-frequency) features more strongly.

    Returns (2^J - 1,) penalty vector.
    """
    if J == 0:
        return jnp.zeros(0, dtype=jnp.float64)

    alpha = float(p)  # p=1 → mild, p=2 → strong
    D = np.zeros(2 ** J - 1, dtype=np.float64)
    idx = 0
    for j in range(1, J + 1):
        n_at_scale = 2 ** (j - 1)
        weight = 4.0 ** (alpha * j)
        for _ in range(n_at_scale):
            D[idx] = weight
            idx += 1

    return jnp.array(D)
