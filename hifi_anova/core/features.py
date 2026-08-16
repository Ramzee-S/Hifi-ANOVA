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

from ..array_backend import xp as jnp  # switchable array backend (numpy exact core)


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

            # +scale_factor on left half, -scale_factor on right half, 0 elsewhere.
            # The rightmost cell (right == 1.0) is closed at its right edge:
            # with a strict `x < right`, every wavelet vanished at x == 1.0
            # exactly, and min-max scaling places the max sample there — an
            # all-zero Haar row. (Measure-zero point: the analytic Gram = I
            # is unaffected.)
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


def build_first_order_features_subset(
    x: jnp.ndarray, K: int, included: list,
    include_linear: bool = True,
    basis_name: str = 'fourier',
) -> jnp.ndarray:
    """First-order features for a SUBSET of variables (uniform basis/K).

    Column layout is ``[var included[0] | var included[1] | ...]`` with
    ``basis_size(K, ...)`` columns per included variable — the uniform layout
    restricted to ``included`` (ascending order expected). Used by the
    order-selective mean path (``variable_orders``) and the variance-variable
    subset (``variance_variables``); with ``included == range(D)`` it equals
    :func:`build_first_order_features` exactly.
    """
    N, D = x.shape
    basis = build_per_variable_basis(x, K, include_linear=include_linear,
                                     basis_name=basis_name)
    idx = jnp.asarray(list(included), dtype=jnp.int32)
    return basis[:, idx, :].reshape(N, -1)


def build_second_order_features_per_pair(
    x: jnp.ndarray, K2_list, pair_indices: jnp.ndarray,
    include_linear: bool = True,
    basis_name: str = 'fourier',
) -> tuple:
    """Second-order features with a per-pair harmonic order (uniform basis).

    Pair ``p = (i, j)`` gets its own order ``K2_list[p]``: the block is the
    outer product of the two variables' order-``K2_list[p]`` bases, so blocks
    are ragged (``B_p²`` columns each with ``B_p = basis_size(K2_list[p], …)``).

    Args:
        x: (N, D) inputs in [0, 1].
        K2_list: sequence of P positive ints, aligned with ``pair_indices``.
        pair_indices: (P, 2) array of (i, j) pairs.
        include_linear / basis_name: shared basis flags (uniform family).

    Returns:
        ``(phi2, pair_block_info)`` — ``phi2`` is (N, Σ_p B_p²);
        ``pair_block_info`` is a tuple of ``(i, j, B_p, B_p, B_p², offset)``
        per pair (offsets relative to the start of ``phi2``), the same tuple
        shape the mixed-basis path uses, so downstream slicing is shared.
    """
    N, D = x.shape
    P = pair_indices.shape[0] if len(pair_indices) > 0 else 0
    # One basis build per distinct order (typically few).
    basis_by_k = {}
    for Kp in set(int(k) for k in K2_list):
        basis_by_k[Kp] = build_per_variable_basis(
            x, Kp, include_linear=include_linear, basis_name=basis_name)

    blocks = []
    info_list = []
    offset = 0
    for p in range(P):
        i = int(pair_indices[p, 0])
        j = int(pair_indices[p, 1])
        Kp = int(K2_list[p])
        basis = basis_by_k[Kp]
        bi = basis[:, i, :]
        bj = basis[:, j, :]
        B = bi.shape[1]
        products = bi[:, :, None] * bj[:, None, :]
        blocks.append(products.reshape(N, -1))
        info_list.append((i, j, B, B, B * B, offset))
        offset += B * B

    phi2 = (jnp.concatenate(blocks, axis=1) if blocks
            else jnp.zeros((N, 0)))
    return phi2, tuple(info_list)


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


# =============================================================================
# Solved-layout mean-design composition (BR-11)
# =============================================================================
# The ONE implementation of the mean design's layout branching (uniform /
# mixed per-variable basis / per-pair K2 / order-selective first-order
# subset). ``HiFiANOVA.build_phi1/build_phi2/build_phi_all_fit`` and the
# linear residual's prediction-time projection rebuild all delegate here, so
# the layout a residual reproduces can never drift from the layout the model
# was solved on.

def _var_specs_as_dicts(var_specs):
    """Accept either the model's static tuple-of-tuples ``(basis, K, ...)``
    or the feature builders' list-of-dicts form."""
    specs = list(var_specs)
    if specs and isinstance(specs[0], dict):
        return specs
    return [{'basis': s[0], 'K': s[1]} for s in specs]


def build_mean_phi1(x, K1, include_linear=True, basis_name='fourier',
                    var_specs=None, fo_included=None):
    """First-order mean block for a design layout.

    ``var_specs`` (mixed per-variable basis) wins; else ``fo_included``
    restricts the uniform layout to the solved subset (BR-06; ``None`` = all
    D variables, ``()`` = empty block — an intercept-only mean design); else
    the full uniform layout.
    """
    if var_specs is not None:
        phi, _ = build_mixed_first_order_features(
            x, _var_specs_as_dicts(var_specs))
        return phi
    if fo_included is not None:
        return build_first_order_features_subset(
            x, K1, list(fo_included), include_linear=include_linear,
            basis_name=basis_name)
    return build_first_order_features(
        x, K1, include_linear=include_linear, basis_name=basis_name)


def build_mean_phi2(x, K2, pair_indices, include_linear=True,
                    basis_name='fourier', var_specs=None, pair_k2=None):
    """Second-order mean block for a design layout, or ``None`` if absent.

    Branch order mirrors ``HiFiANOVA.build_phi2``: mixed per-variable basis,
    then per-pair K2 (ragged blocks), then the uniform shared-K2 layout.
    """
    if var_specs is not None and pair_indices is not None:
        phi2, _ = build_mixed_second_order_features(
            x, pair_indices, _var_specs_as_dicts(var_specs))
        return phi2 if phi2.shape[1] > 0 else None
    if pair_k2 is not None and pair_indices is not None:
        phi2, _ = build_second_order_features_per_pair(
            x, pair_k2, pair_indices, include_linear=include_linear,
            basis_name=basis_name)
        return phi2 if phi2.shape[1] > 0 else None
    if K2 > 0 and pair_indices is not None:
        return build_second_order_features(
            x, K2, pair_indices, include_linear=include_linear,
            basis_name=basis_name)
    return None


def build_mean_design(x, *, K1, K2=0, K3=0, pair_indices=None,
                      triple_indices=None, include_linear_1=True,
                      include_linear_2=True, include_linear_3=True,
                      basis_name='fourier', var_specs=None, pair_k2=None,
                      fo_included=None):
    """Concatenated mean design ``[phi1 | phi2 | phi3]`` for a layout.

    With ``fo_included`` set this is the FITTED-DESIGN layout (the columns the
    model was actually solved on — matches ``record.Phi``); with everything
    default it equals the uniform full layout. ``fo_included=()`` with no
    pairs/triples yields an (N, 0) design: the intercept-only mean, whose
    orthogonal projection is a no-op (the complement-only limit).
    """
    if var_specs is not None and fo_included is not None:
        raise ValueError(
            "variable_orders first-order subsets are not supported with a "
            "mixed per-variable basis (var_specs); the trainer never "
            "produces this combination.")
    parts = [build_mean_phi1(x, K1, include_linear=include_linear_1,
                             basis_name=basis_name, var_specs=var_specs,
                             fo_included=fo_included)]
    phi2 = build_mean_phi2(x, K2, pair_indices,
                           include_linear=include_linear_2,
                           basis_name=basis_name, var_specs=var_specs,
                           pair_k2=pair_k2)
    if phi2 is not None:
        parts.append(phi2)
    if K3 > 0 and triple_indices is not None:
        parts.append(build_third_order_features(
            x, K3, triple_indices, include_linear=include_linear_3,
            basis_name=basis_name))
    return jnp.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]
