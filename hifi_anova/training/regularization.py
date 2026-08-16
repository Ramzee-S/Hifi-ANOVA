"""Build per-feature regularization vectors for all strategies.

The regularization vector r has one entry per feature (coefficient).
The ridge penalty is w^T diag(r) w.

Strategies:
  uniform:    r[j] = lambda for all j. Standard ridge — treats all coefficients
              equally regardless of their variance contribution. Useful as a
              reference but not recommended for Fourier bases.

  variance:   r[j] = lambda * G[j,j]. Penalizes each coefficient proportionally
              to the variance it contributes per unit squared coefficient.
              linear: lambda * 1/12, harmonics: lambda * 1/2.
              This is the principled "equal impact" penalty — lambda directly
              controls the variance budget per feature. DEFAULT strategy.

  smoothness: r[j] = lambda * (2*pi*k)^2 / 2. Integrated squared first
              derivative (Sobolev H1 penalty). Penalizes rate of change.
              Standard in smoothing splines (Wahba 1990).

  curvature:  r[j] = lambda * (2*pi*k)^4 / 2. Integrated squared second
              derivative. Penalizes curvature, leaves linear effects free.
              Standard cubic spline penalty (Green & Silverman 1994).

  sobolev:    r[j] = lambda * (1 + (2*pi*k)^2)^s. Sobolev H^s norm.
              The most principled option: smoothly interpolates between
              uniform (s=0), smoothness-like (s=1), and curvature-like (s=2).
              The "+1" naturally regularizes the linear term (r = lambda)
              without any ad-hoc fix. Default s=1.
              References: Adams & Fournier (2003), Wahba (1990).

  spectral:   r[j] = lambda * k^alpha. Direct frequency weighting.
              alpha=0: uniform, alpha=2: ~smoothness, alpha=4: ~curvature.
              Simpler parameterization, intuitive frequency control.
              Linear term gets small stability ridge (like curvature).
"""

from ..array_backend import xp as jnp  # switchable array backend (numpy exact core)
import numpy as np

from ..core.gram import build_derivative_penalty, build_gram_matrix


def _build_single_order_reg(K: int, n_blocks: int, lam: float,
                            strategy: str,
                            include_linear: bool = True,
                            basis_name: str = 'fourier') -> jnp.ndarray:
    """Build regularization for a single order (first or second).

    Args:
        K: max harmonic/degree for this order
        n_blocks: number of blocks (D for first-order, P for second-order)
        lam: base regularization strength
        strategy: 'uniform', 'variance', 'smoothness', 'curvature', 'sobolev[_s]', 'spectral[_a]'
        include_linear: whether basis includes linear term
        basis_name: 'fourier' or 'legendre'

    Returns:
        Regularization vector of shape (n_blocks * block_size,)
    """
    from ..core.features import basis_size as _bs
    block_size = _bs(K, include_linear=include_linear, basis_name=basis_name)

    if block_size == 0:
        return jnp.zeros(0, dtype=jnp.float64)

    if strategy == 'uniform':
        per_basis = np.full(block_size, lam, dtype=np.float64)
    elif strategy == 'variance':
        G = np.array(build_gram_matrix(K, include_linear=include_linear, basis_name=basis_name))
        per_basis = lam * np.diag(G)
    elif strategy == 'smoothness':
        D2 = np.array(build_derivative_penalty(K, p=1, include_linear=include_linear, basis_name=basis_name))
        per_basis = lam * D2
    elif strategy == 'curvature':
        D4 = np.array(build_derivative_penalty(K, p=2, include_linear=include_linear, basis_name=basis_name))
        per_basis = lam * D4
        # For Fourier: linear term has zero curvature, add stability ridge
        # For Legendre: P̃₁ has zero second derivative, same issue
        if block_size > 0:
            per_basis[0] = max(per_basis[0], lam * 1e-6)
    elif strategy.startswith('sobolev'):
        parts = strategy.split('_')
        s = float(parts[1]) if len(parts) > 1 else 1.0
        per_basis = np.zeros(block_size, dtype=np.float64)
        if basis_name == 'haar':
            # Haar: scale j maps to frequency 2^{j-1}.
            # Sobolev analog: r = λ·(1 + 4^{j-1})^s
            idx = 0
            for j in range(1, K + 1):
                n_at_scale = 2 ** (j - 1)
                val = lam * (1.0 + 4.0 ** (j - 1)) ** s
                for _ in range(n_at_scale):
                    per_basis[idx] = val
                    idx += 1
        elif basis_name == 'legendre':
            n_polys = K if include_linear else K - 1
            for j in range(n_polys):
                k = (j + 1) if include_linear else (j + 2)
                per_basis[j] = lam * (1.0 + k * (k + 1)) ** s
        else:
            # Fourier: r = λ·(1 + (2πk)²)^s
            offset = 0
            if include_linear:
                per_basis[0] = lam * 1.0
                offset = 1
            for k in range(1, K + 1):
                val = (1.0 + (2.0 * np.pi * k) ** 2) ** s
                per_basis[offset + 2 * (k - 1)] = lam * val
                per_basis[offset + 2 * (k - 1) + 1] = lam * val
    elif strategy.startswith('spectral'):
        parts = strategy.split('_')
        alpha = float(parts[1]) if len(parts) > 1 else 2.0
        per_basis = np.zeros(block_size, dtype=np.float64)
        if basis_name == 'haar':
            # Haar: scale j maps to frequency 2^{j-1}.
            # Spectral: r = λ·(2^{j-1})^alpha
            idx = 0
            for j in range(1, K + 1):
                n_at_scale = 2 ** (j - 1)
                val = lam * float(2 ** (j - 1)) ** alpha
                for _ in range(n_at_scale):
                    per_basis[idx] = val
                    idx += 1
        elif basis_name == 'legendre':
            n_polys = K if include_linear else K - 1
            for j in range(n_polys):
                k = (j + 1) if include_linear else (j + 2)
                per_basis[j] = lam * float(k) ** alpha
        else:
            # Fourier
            offset = 0
            if include_linear:
                per_basis[0] = lam * 1.0  # linear term: k=0, r = λ·0^α = 0 → use λ for stability
                offset = 1
            for k in range(1, K + 1):
                val = float(k) ** alpha
                per_basis[offset + 2 * (k - 1)] = lam * val
                per_basis[offset + 2 * (k - 1) + 1] = lam * val
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    reg = np.tile(per_basis, n_blocks)
    return jnp.array(reg)


def _build_second_order_reg_block(K: int, lam: float, strategy: str,
                                   include_linear: bool = True,
                                   basis_name: str = 'fourier') -> np.ndarray:
    """Build regularization for one second-order pair block.

    For second-order terms, the block size is B^2 where B = basis_size(K, include_linear, basis_name).
    """
    from ..core.features import basis_size as _bs
    block_1d = _bs(K, include_linear, basis_name)
    block_2d = block_1d ** 2

    if block_2d == 0:
        return np.array([], dtype=np.float64)

    if strategy == 'uniform':
        return np.full(block_2d, lam, dtype=np.float64)
    elif strategy == 'variance':
        G1 = np.array(build_gram_matrix(K, include_linear, basis_name))
        G1_diag = np.diag(G1)
        G2_diag = np.outer(G1_diag, G1_diag).ravel()
        return lam * G2_diag
    elif strategy in ('curvature', 'smoothness'):
        p = 2 if strategy == 'curvature' else 1
        D_raw = np.array(build_derivative_penalty(K, p=p, include_linear=include_linear,
                                                    basis_name=basis_name))
        if len(D_raw) > 0:
            D_raw[0] = max(D_raw[0], 1e-6)  # stability ridge on linear/P̃₁

        reg_2d = np.zeros(block_2d, dtype=np.float64)
        for a in range(block_1d):
            for b in range(block_1d):
                reg_2d[a * block_1d + b] = lam * (D_raw[a] + D_raw[b])
                if reg_2d[a * block_1d + b] < lam * 1e-6:
                    reg_2d[a * block_1d + b] = lam * 1e-6
        return reg_2d
    elif strategy.startswith('sobolev') or strategy.startswith('spectral'):
        r_1d = np.array(_build_single_order_reg(K, 1, lam, strategy,
                                                 include_linear=include_linear,
                                                 basis_name=basis_name))
        reg_2d = np.zeros(block_2d, dtype=np.float64)
        for a in range(block_1d):
            for b in range(block_1d):
                reg_2d[a * block_1d + b] = r_1d[a] + r_1d[b]
                if reg_2d[a * block_1d + b] < lam * 1e-6:
                    reg_2d[a * block_1d + b] = lam * 1e-6
        return reg_2d
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def _build_third_order_reg_block(K: int, lam: float, strategy: str,
                                  include_linear: bool = True,
                                  basis_name: str = 'fourier') -> np.ndarray:
    """Build regularization for one third-order triple block.

    Block size is B^3 where B = basis_size(K, include_linear).
    Penalty on triple (a,b,c) = lam * (D_raw[a] + D_raw[b] + D_raw[c]).
    """
    from ..core.features import basis_size as _bs
    block_1d = _bs(K, include_linear, basis_name)
    block_3d = block_1d ** 3

    if block_3d == 0:
        return np.array([], dtype=np.float64)

    if strategy == 'uniform':
        return np.full(block_3d, lam, dtype=np.float64)
    elif strategy == 'variance':
        G1 = np.array(build_gram_matrix(K, include_linear, basis_name))
        G1_diag = np.diag(G1)
        G3_diag = np.einsum('i,j,k->ijk', G1_diag, G1_diag, G1_diag).ravel()
        return lam * G3_diag
    elif strategy in ('curvature', 'smoothness'):
        p = 2 if strategy == 'curvature' else 1
        D_raw = np.array(build_derivative_penalty(K, p=p, include_linear=include_linear,
                                                    basis_name=basis_name))
        if len(D_raw) > 0:
            D_raw[0] = max(D_raw[0], 1e-6)
        reg_3d = np.zeros(block_3d, dtype=np.float64)
        for a in range(block_1d):
            for b in range(block_1d):
                for c in range(block_1d):
                    idx = a * block_1d ** 2 + b * block_1d + c
                    reg_3d[idx] = lam * (D_raw[a] + D_raw[b] + D_raw[c])
                    if reg_3d[idx] < lam * 1e-6:
                        reg_3d[idx] = lam * 1e-6
        return reg_3d
    elif strategy.startswith('sobolev') or strategy.startswith('spectral'):
        r_1d = np.array(_build_single_order_reg(K, 1, lam, strategy,
                                                 include_linear=include_linear,
                                                 basis_name=basis_name))
        reg_3d = np.zeros(block_3d, dtype=np.float64)
        for a in range(block_1d):
            for b in range(block_1d):
                for c in range(block_1d):
                    idx = a * block_1d ** 2 + b * block_1d + c
                    reg_3d[idx] = r_1d[a] + r_1d[b] + r_1d[c]
                    if reg_3d[idx] < lam * 1e-6:
                        reg_3d[idx] = lam * 1e-6
        return reg_3d
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def build_regularization_vector(
    D: int, K1: int, K2: int, P: int,
    strategy='variance',
    lambda_order1: float = 0.001,
    lambda_order2: float = 0.01,
    K3: int = 0, T: int = 0,
    lambda_order3: float = 0.1,
    M_residual: int = 0,
    lambda_residual: float = 1.0,
    include_linear_1: bool = True,
    include_linear_2: bool = True,
    include_linear_3: bool = True,
    basis_name: str = 'fourier',
) -> jnp.ndarray:
    """Build the full regularization vector for [w1 | w2 | w3 | alpha_res].

    Args:
        D: number of variables
        K1: max harmonic, first order
        K2: max harmonic, second order (0 for no second order), OR a per-pair
            sequence of P orders (ragged pair blocks, one per pair)
        P: number of pairs
        strategy: regularization strategy — either a single string applied to
            all orders (e.g. 'curvature') or a dict with per-order strategies:
            {'order1': 'curvature', 'order2': 'smoothness', 'order3': 'uniform'}.
            Missing keys fall back to the 'default' key or 'variance'.
        lambda_order1: regularization for first-order terms
        lambda_order2: regularization for second-order terms
        K3: max harmonic, third order (0 for no third order)
        T: number of triples
        lambda_order3: regularization for third-order terms
        M_residual: number of linear residual features (0 for none)
        lambda_residual: regularization for linear residual
        include_linear_1: whether first-order basis includes linear term
        include_linear_2: whether second-order basis includes linear term
        include_linear_3: whether third-order basis includes linear term
        basis_name: 'fourier' or 'legendre'

    Returns:
        Regularization vector matching the concatenated feature matrix.
    """
    # Resolve per-order strategies
    if isinstance(strategy, dict):
        default_strat = strategy.get('default', 'variance')
        strat1 = strategy.get('order1', default_strat)
        strat2 = strategy.get('order2', default_strat)
        strat3 = strategy.get('order3', default_strat)
    else:
        strat1 = strat2 = strat3 = strategy

    # First-order regularization
    reg1 = _build_single_order_reg(K1, D, lambda_order1, strat1,
                                    include_linear=include_linear_1, basis_name=basis_name)

    parts = [reg1]

    # Second-order regularization. ``K2`` may be a per-pair sequence of P
    # orders (the per-pair-K2 term-structure path): each pair block is then
    # built at its own order and the blocks are ragged. A scalar K2 keeps the
    # uniform tiled path byte-identical.
    if not isinstance(K2, int) and hasattr(K2, '__len__'):
        if len(K2) != P:
            raise ValueError(
                f"per-pair K2 sequence has {len(K2)} entries but P={P} pairs.")
        for K2_p in K2:
            block_p = _build_second_order_reg_block(
                int(K2_p), lambda_order2, strat2,
                include_linear=include_linear_2, basis_name=basis_name)
            if len(block_p) > 0:
                parts.append(jnp.array(block_p))
    elif K2 > 0 and P > 0:
        single_block = _build_second_order_reg_block(
            K2, lambda_order2, strat2, include_linear=include_linear_2,
            basis_name=basis_name)
        if len(single_block) > 0:
            reg2 = jnp.array(np.tile(single_block, P))
            parts.append(reg2)

    # Third-order regularization
    if K3 > 0 and T > 0:
        single_block_3d = _build_third_order_reg_block(
            K3, lambda_order3, strat3, include_linear=include_linear_3,
            basis_name=basis_name)
        if len(single_block_3d) > 0:
            reg3 = jnp.array(np.tile(single_block_3d, T))
            parts.append(reg3)

    # Linear residual regularization (uniform)
    if M_residual > 0:
        reg_res = jnp.full(M_residual, lambda_residual)
        parts.append(reg_res)

    return jnp.concatenate(parts)


def build_mixed_regularization_vector(
    var_specs: list,
    strategy: str = 'variance',
    lambda_order1: float = 0.001,
    pair_indices=None,
    lambda_order2: float = 0.01,
) -> jnp.ndarray:
    """Build regularization for mixed per-variable basis models.

    Each variable gets its own regularization block sized to its basis.
    For second-order pairs, penalties combine additively from both
    variables' 1D penalties.

    Args:
        var_specs: list of D dicts with 'basis' and 'K'.
        strategy: regularization strategy (applied per-variable with its basis).
        lambda_order1: first-order regularization.
        pair_indices: (P, 2) pair indices for second-order, or None.
        lambda_order2: second-order regularization.

    Returns:
        Regularization vector matching [w1_mixed | w2_mixed].
    """
    from ..core.features import basis_size, _mixed_include_linear

    # First-order: per-variable blocks
    parts = []
    for spec in var_specs:
        bn = spec['basis']
        K = spec['K']
        il = _mixed_include_linear(bn)
        block_reg = _build_single_order_reg(K, 1, lambda_order1, strategy,
                                             include_linear=il, basis_name=bn)
        parts.append(block_reg)

    # Second-order: per-pair blocks with mixed G_i, G_j
    if pair_indices is not None and len(pair_indices) > 0:
        for p in range(len(pair_indices)):
            i = int(pair_indices[p, 0])
            j = int(pair_indices[p, 1])
            spec_i, spec_j = var_specs[i], var_specs[j]
            il_i = _mixed_include_linear(spec_i['basis'])
            il_j = _mixed_include_linear(spec_j['basis'])
            Bi = basis_size(spec_i['K'], il_i, spec_i['basis'])
            Bj = basis_size(spec_j['K'], il_j, spec_j['basis'])

            # Get 1D penalties for each variable
            r_i = np.array(_build_single_order_reg(
                spec_i['K'], 1, lambda_order2, strategy,
                include_linear=il_i, basis_name=spec_i['basis']))
            r_j = np.array(_build_single_order_reg(
                spec_j['K'], 1, lambda_order2, strategy,
                include_linear=il_j, basis_name=spec_j['basis']))

            # Additive penalty: r(a,b) = r_i[a] + r_j[b]
            reg_pair = np.zeros(Bi * Bj, dtype=np.float64)
            for a in range(Bi):
                for b in range(Bj):
                    val = r_i[a] + r_j[b]
                    if val < lambda_order2 * 1e-6:
                        val = lambda_order2 * 1e-6
                    reg_pair[a * Bj + b] = val
            parts.append(jnp.array(reg_pair))

    return jnp.concatenate(parts)


def build_variance_regularization_vector(
    D: int, Kh: int,
    strategy: str = 'variance',
    lambda_h: float = 0.1,
    K2h: int = 0, Ph: int = 0,
    lambda_h2: float = 1.0,
    K3h: int = 0, Th: int = 0,
    lambda_h3: float = 1.0,
    M_h_residual: int = 0,
    lambda_h_res: float = 1.0,
    include_linear_h1: bool = True,
    include_linear_h2: bool = True,
    include_linear_h3: bool = True,
    basis_name: str = 'fourier',
) -> jnp.ndarray:
    """Build regularization for variance coefficients [w1_h | w2_h | w3_h | w_res_h].

    Args:
        D: number of variables
        Kh: max harmonic, first-order variance
        strategy: regularization strategy
        lambda_h: first-order variance regularization
        K2h: max harmonic, second-order variance (0 = none)
        Ph: number of variance pairs
        lambda_h2: second-order variance regularization
        K3h: max harmonic, third-order variance (0 = none)
        Th: number of variance triples
        lambda_h3: third-order variance regularization
        M_h_residual: number of variance residual features (0 = none)
        lambda_h_res: variance residual regularization

    Returns:
        Shape (D*(2Kh+1) + Ph*(2K2h+1)^2 + Th*(2K3h+1)^3 + M_h_residual,)
    """
    parts = [_build_single_order_reg(Kh, D, lambda_h, strategy,
                                      include_linear=include_linear_h1, basis_name=basis_name)]

    if K2h > 0 and Ph > 0:
        single_block = _build_second_order_reg_block(K2h, lambda_h2, strategy,
                                                       include_linear=include_linear_h2, basis_name=basis_name)
        parts.append(jnp.array(np.tile(single_block, Ph)))

    if K3h > 0 and Th > 0:
        single_block_3d = _build_third_order_reg_block(K3h, lambda_h3, strategy,
                                                         include_linear=include_linear_h3, basis_name=basis_name)
        parts.append(jnp.array(np.tile(single_block_3d, Th)))

    if M_h_residual > 0:
        parts.append(jnp.full(M_h_residual, lambda_h_res))

    return jnp.concatenate(parts)
