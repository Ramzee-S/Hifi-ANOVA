"""Build feature matrices from batched inputs.

ORDERING CONTRACT (all downstream code depends on this):

basis_name='fourier' (default):
  First-order: [var1: lin,cos1,sin1,...,cosK,sinK | var2: ...] — (2K+1) per var
  Second/third-order: outer products of per-variable basis vectors
  include_linear controls whether orders 2+ include the linear term

basis_name='legendre':
  First-order: [var1: P1,P2,...,PK | var2: ...] — K per var (degrees 1..K)
  P̃ₖ(x) = Pₖ(2x-1) shifted Legendre, all with zero integral on [0,1]
  P̃₁ = 2(x-½) is the linear term (always included)
  Gram matrix is perfectly diagonal: G[j,j] = 1/(2j+3)
  include_linear is irrelevant (always True for Legendre)

basis_name='haar':
  First-order: [var1: ψ₁₀,ψ₂₀,ψ₂₁,...,ψ_J,2^{J-1}-1 | var2: ...] — (2^K-1) per var
  K is reinterpreted as J (max wavelet scale).
  Haar wavelets: piecewise-constant, localized, orthonormal.
  Gram matrix is the identity. Sobol = sum of squared coefficients.
  include_linear is irrelevant (Haar has no linear term).

ALL FEATURE CONSTRUCTION IS VECTORIZED.
"""

import jax
import jax.numpy as jnp


def basis_size(K: int, include_linear: bool = True,
               basis_name: str = 'fourier') -> int:
    """Number of basis functions per variable.

    Args:
        K: max harmonic (Fourier), max polynomial degree (Legendre),
           or max wavelet scale J (Haar).
        include_linear: for Fourier, whether to include (x-0.5) term.
                        Ignored for Legendre and Haar.
        basis_name: 'fourier', 'legendre', or 'haar'

    Returns:
        Fourier: 2K+1 (full) or 2K (spectral-only)
        Legendre: K (degrees 1..K)
        Haar: 2^K - 1 (scales 1..K)
    """
    if basis_name == 'haar':
        return 2 ** K - 1 if K > 0 else 0
    if basis_name == 'legendre':
        return K
    return 2 * K + 1 if include_linear else 2 * K


def build_per_variable_basis(x: jnp.ndarray, K: int,
                              include_linear: bool = True,
                              basis_name: str = 'fourier') -> jnp.ndarray:
    """Build basis vectors per variable.

    Args:
        x: (N, D) inputs in [0, 1].
        K: max harmonic/degree.
        include_linear: for Fourier, whether to include linear term.
        basis_name: 'fourier' or 'legendre'

    Returns:
        (N, D, B) where B = basis_size(K, include_linear, basis_name).
    """
    if basis_name == 'haar':
        return _build_haar_basis(x, K)
    if basis_name == 'legendre':
        return _build_legendre_basis(x, K)
    return _build_fourier_basis(x, K, include_linear)


def _build_fourier_basis(x: jnp.ndarray, K: int,
                          include_linear: bool = True) -> jnp.ndarray:
    """Fourier basis: [lin, cos1, sin1, ..., cosK, sinK] or [cos1, sin1, ...]."""
    N, D = x.shape

    if K == 0:
        if include_linear:
            return (x - 0.5)[:, :, None]
        else:
            return jnp.zeros((N, D, 0))

    k = jnp.arange(1, K + 1)
    args = 2.0 * jnp.pi * x[:, :, None] * k[None, None, :]
    cos_f = jnp.cos(args)
    sin_f = jnp.sin(args)
    fourier = jnp.stack([cos_f, sin_f], axis=-1).reshape(N, D, 2 * K)

    if include_linear:
        linear = (x - 0.5)[:, :, None]
        return jnp.concatenate([linear, fourier], axis=-1)
    else:
        return fourier


def _build_legendre_basis(x: jnp.ndarray, K: int) -> jnp.ndarray:
    """Shifted Legendre polynomials P̃ₖ(x) = Pₖ(2x-1), k=1..K.

    All have zero integral on [0,1] (Hoeffding condition).
    P̃₁(x) = 2x-1 = 2(x-½) is the linear term.

    Returns (N, D, K). For K=0, returns (N, D, 0).
    """
    N, D = x.shape
    if K == 0:
        return jnp.zeros((N, D, 0))

    t = 2.0 * x - 1.0  # map [0,1] → [-1,1]

    # Three-term recurrence: P₀=1, P₁=t, (k+1)Pₖ₊₁ = (2k+1)tPₖ − kPₖ₋₁
    polys = []
    P_prev = jnp.ones((N, D))   # P₀(t) = 1
    P_curr = t                    # P₁(t) = t

    polys.append(P_curr)          # k=1: P̃₁(x) = 2x-1

    for k in range(1, K):
        P_next = ((2 * k + 1) * t * P_curr - k * P_prev) / (k + 1)
        polys.append(P_next)
        P_prev = P_curr
        P_curr = P_next

    return jnp.stack(polys, axis=-1)  # (N, D, K)


def _build_haar_basis(x: jnp.ndarray, J: int) -> jnp.ndarray:
    """Haar wavelet basis psi_{j,k}(x) for scales j=1..J.

    Each wavelet is piecewise constant with support on an interval of
    width 2^{-(j-1)}, normalized by 2^{(j-1)/2} for orthonormality.

    Returns (N, D, 2^J - 1). For J=0, returns (N, D, 0).
    """
    N, D = x.shape
    if J == 0:
        return jnp.zeros((N, D, 0))

    features = []
    for j in range(1, J + 1):
        scale_factor = 2.0 ** ((j - 1) / 2.0)
        n_positions = 2 ** (j - 1)
        interval_width = 1.0 / n_positions

        for k in range(n_positions):
            left = k * interval_width
            mid = left + interval_width / 2.0
            right = left + interval_width

            # +scale_factor on left half, -scale_factor on right half, 0 elsewhere
            psi = jnp.where(
                (x >= left) & (x < mid),
                scale_factor,
                jnp.where(
                    (x >= mid) & (x < right),
                    -scale_factor,
                    0.0
                )
            )
            features.append(psi)

    return jnp.stack(features, axis=-1)  # (N, D, 2^J - 1)


def build_first_order_features(x: jnp.ndarray, K: int,
                                include_linear: bool = True,
                                basis_name: str = 'fourier') -> jnp.ndarray:
    """Build first-order feature matrix.

    Args:
        x: (N, D) inputs in [0, 1].
        K: max harmonic number.
        include_linear: if False, pure harmonics only (no linear term).
            For Fourier: [cos1, sin1, ..., cosK, sinK] — 2K features.
            Ignored for Legendre (always includes P̃₁) and Haar.
        basis_name: 'fourier', 'legendre', or 'haar'.

    Returns: (N, D*B) where B = basis_size(K, include_linear, basis_name).
    """
    N, D = x.shape
    basis = build_per_variable_basis(x, K, include_linear=include_linear,
                                      basis_name=basis_name)
    return basis.reshape(N, -1)


def build_second_order_features(x: jnp.ndarray, K: int,
                                pair_indices: jnp.ndarray,
                                include_linear: bool = True,
                                basis_name: str = 'fourier') -> jnp.ndarray:
    """Build second-order feature matrix using outer products.

    Returns: (N, P*B²) where B = basis_size(K, include_linear, basis_name).
    """
    N, D = x.shape
    basis = build_per_variable_basis(x, K, include_linear, basis_name)

    left = basis[:, pair_indices[:, 0], :]
    right = basis[:, pair_indices[:, 1], :]
    products = left[:, :, :, None] * right[:, :, None, :]

    return products.reshape(N, -1)


# ─────────────────────────────────────────────────────────────
# Mixed per-variable basis construction
# ─────────────────────────────────────────────────────────────

def _mixed_include_linear(basis_name: str) -> bool:
    """In mixed mode, each basis has a fixed include_linear rule:
    Legendre: True (P̃₁ is the linear term)
    Fourier: False (no linear — Legendre owns it)
    Haar: False (no linear by nature)
    """
    return basis_name == 'legendre'


def build_mixed_first_order_features(
    x: jnp.ndarray,
    var_specs: list,
) -> tuple:
    """Build first-order features with per-variable basis assignment.

    In mixed mode, each variable uses its own basis family and K:
      - Legendre: K features (P̃₁..P̃_K), includes linear
      - Fourier: 2K features (cos,sin only), NO linear
      - Haar: 2^K-1 features (wavelets), no linear

    Args:
        x: (N, D) inputs in [0, 1].
        var_specs: list of D dicts, each with 'basis' and 'K'.
            Example: [{'basis': 'legendre', 'K': 5},
                      {'basis': 'fourier', 'K': 8},
                      {'basis': 'haar', 'K': 4}]

    Returns:
        phi: (N, F_total) concatenated feature matrix.
        block_info: tuple of D tuples (basis, K, include_linear, block_size, offset).
    """
    N, D = x.shape
    blocks = []
    info_list = []
    offset = 0

    for i in range(D):
        spec = var_specs[i]
        bn = spec['basis']
        K = spec['K']
        il = _mixed_include_linear(bn)
        B = basis_size(K, include_linear=il, basis_name=bn)

        # Build basis for single variable
        basis_i = build_per_variable_basis(
            x[:, i:i + 1], K, include_linear=il, basis_name=bn)
        blocks.append(basis_i[:, 0, :])  # (N, B)

        info_list.append((bn, K, il, B, offset))
        offset += B

    phi = jnp.concatenate(blocks, axis=1) if blocks else jnp.zeros((N, 0))
    return phi, tuple(info_list)


def build_mixed_second_order_features(
    x: jnp.ndarray,
    pair_indices: jnp.ndarray,
    var_specs: list,
) -> tuple:
    """Build second-order features with per-variable basis.

    For pair (i, j), uses basis_i ⊗ basis_j (outer product of potentially
    different-sized basis vectors). The Gram matrix for this pair is
    G_i ⊗ G_j, NOT the uniform G₁ ⊗ G₁.

    Args:
        x: (N, D) inputs in [0, 1].
        pair_indices: (P, 2) array of (i, j) pairs.
        var_specs: list of D dicts with 'basis' and 'K'.

    Returns:
        phi2: (N, sum of Bi*Bj for each pair) feature matrix.
        pair_block_info: tuple of P tuples
            (var_i, var_j, Bi, Bj, block_size, offset).
    """
    N, D = x.shape
    P = pair_indices.shape[0] if len(pair_indices) > 0 else 0
    blocks = []
    info_list = []
    offset = 0

    for p in range(P):
        i = int(pair_indices[p, 0])
        j = int(pair_indices[p, 1])
        spec_i, spec_j = var_specs[i], var_specs[j]

        il_i = _mixed_include_linear(spec_i['basis'])
        il_j = _mixed_include_linear(spec_j['basis'])

        basis_i = build_per_variable_basis(
            x[:, i:i + 1], spec_i['K'], il_i, spec_i['basis'])[:, 0, :]
        basis_j = build_per_variable_basis(
            x[:, j:j + 1], spec_j['K'], il_j, spec_j['basis'])[:, 0, :]

        products = basis_i[:, :, None] * basis_j[:, None, :]  # (N, Bi, Bj)
        Bi = basis_i.shape[1]
        Bj = basis_j.shape[1]
        blocks.append(products.reshape(N, -1))

        info_list.append((i, j, Bi, Bj, Bi * Bj, offset))
        offset += Bi * Bj

    if blocks:
        phi2 = jnp.concatenate(blocks, axis=1)
    else:
        phi2 = jnp.zeros((N, 0))
    return phi2, tuple(info_list)


def build_third_order_features(x: jnp.ndarray, K: int,
                                triple_indices: jnp.ndarray,
                                include_linear: bool = True,
                                basis_name: str = 'fourier') -> jnp.ndarray:
    """Build third-order feature matrix using triple outer products.

    Returns: (N, T*B³) where B = basis_size(K, include_linear, basis_name).
    """
    N, D = x.shape
    basis = build_per_variable_basis(x, K, include_linear, basis_name)

    left = basis[:, triple_indices[:, 0], :]
    mid = basis[:, triple_indices[:, 1], :]
    right = basis[:, triple_indices[:, 2], :]

    products = (left[:, :, :, None, None]
                * mid[:, :, None, :, None]
                * right[:, :, None, None, :])

    return products.reshape(N, -1)
