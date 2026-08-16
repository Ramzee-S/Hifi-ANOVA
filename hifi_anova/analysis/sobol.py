"""Analytic and correlative Sobol indices.

Two types of indices are computed:
  - Structural (analytic): w^T G w, assuming independent inputs.
    Sum to 1. Characterize the function's intrinsic sensitivity.
  - Correlative (empirical): joint-law covariance allocation over ALL retained
    structured components, from covariance of component outputs on actual data.
    The complete collection sums to 1 identically (by linearity of covariance,
    regardless of dependence); individual shares may be negative or exceed 1, and
    a first-order-only subset need not sum to 1 when interactions are retained.
    This is an independence-assumption diagnostic, not an official
    correlated-attribution estimand.

For independent inputs, both types agree.
The divergence between them diagnoses the impact of input dependence.
"""

from ..array_backend import xp as jnp  # switchable array backend (numpy exact core)
import numpy as np
from typing import Optional

from ..core.gram import build_gram_matrix, build_gram_matrix_2d, build_gram_matrix_3d
from ..core.features import basis_size
from ..model.linear_residual import predict_residual_batch
from .._result_aliases import ResultAliasDict


def _block_variances(w, gram, n_blocks: int, block: int) -> np.ndarray:
    """Batched per-block variances ``max(0, w_b^T G w_b)`` for contiguous blocks.

    Replaces a Python loop of ``n_blocks`` tiny JAX quadratic forms (each of which
    forced a device sync via ``float(...)``) with a single numpy einsum. Numerics
    are identical to the per-block ``w_b @ G @ w_b`` up to float64 round-off.

    G is a Gram matrix (PSD), so each quadratic form is >= 0 up to round-off;
    the clip only absorbs that round-off. A MATERIALLY negative value means a
    mis-built Gram or wrong block slicing, which a silent clip would mask —
    warn instead of hiding it.
    """
    W = np.asarray(w, dtype=np.float64).reshape(n_blocks, block)
    G = np.asarray(gram, dtype=np.float64)
    raw = np.einsum('bi,ij,bj->b', W, G, W)
    neg_tol = 1e-8 * max(1.0, float(np.max(np.abs(raw))) if raw.size else 1.0)
    if raw.size and float(np.min(raw)) < -neg_tol:
        import warnings
        warnings.warn(
            "Block variance w^T G w is materially negative "
            f"(min={float(np.min(raw)):.3e}, tol={-neg_tol:.1e}); the Gram "
            "matrix or block slicing is likely inconsistent. The value is "
            "clipped to 0, but the resulting Sobol indices should not be "
            "trusted until this is resolved.", stacklevel=2)
    return np.maximum(0.0, raw)


def _total_order_variances(D: int, first: dict, second: dict,
                           third: dict) -> dict:
    """Per-variable total-order variances: ``V_i + Σ_{i∈pair} + Σ_{i∈triple}``.

    Shared by every Sobol normalization (mean, variance, core). Keys of
    ``second``/``third`` are index tuples; membership of ``i`` in the tuple
    decides inclusion.
    """
    totals = {}
    for i in range(D):
        t = first.get(i, 0.0)
        for key, v in second.items():
            if i in key:
                t += v
        for key, v in third.items():
            if i in key:
                t += v
        totals[i] = t
    return totals


def _normalized_sobol_block(D: int, first: dict, second: dict, third: dict,
                            residual_var: float, denom: float) -> dict:
    """Sobol shares ``V_u/denom`` incl. the residual share (4d normalization).

    The single normalization routine behind both ``mean_sobol`` and
    ``log_variance_sobol`` (their legacy code was verbatim-duplicated). Degenerate
    ``denom <= 0``: first_order and total_order zero-filled, residual 0.0,
    second/third left empty — the exact legacy shape.
    """
    block = {'first_order': {}, 'second_order': {}, 'third_order': {},
             'total_order': {}}
    if denom > 0:
        for i, v in first.items():
            block['first_order'][i] = v / denom
        for key, v in second.items():
            block['second_order'][key] = v / denom
        for key, v in third.items():
            block['third_order'][key] = v / denom
        block['residual'] = residual_var / denom
    else:
        for i in range(D):
            block['first_order'][i] = 0.0
        block['residual'] = 0.0
    totals = _total_order_variances(D, first, second, third)
    for i in range(D):
        block['total_order'][i] = totals[i] / denom if denom > 0 else 0.0
    return block


def _core_sobol_block(D: int, first: dict, second: dict, third: dict,
                      v_core: float) -> dict:
    """Core Sobol shares ``V_u/V_core`` (no residual in the denominator).

    Degenerate ``v_core <= 0``: only first_order is zero-filled; total_order
    stays EMPTY — the legacy asymmetry, pinned by
    tests/test_characterization_pins.py, preserved verbatim.
    """
    block = {'first_order': {}, 'second_order': {}, 'third_order': {},
             'total_order': {}}
    if v_core > 0:
        for i, v in first.items():
            block['first_order'][i] = v / v_core
        for key, v in second.items():
            block['second_order'][key] = v / v_core
        for key, v in third.items():
            block['third_order'][key] = v / v_core
        totals = _total_order_variances(D, first, second, third)
        for i in range(D):
            block['total_order'][i] = totals[i] / v_core
    else:
        for i in range(D):
            block['first_order'][i] = 0.0
    return block


def _mean_component_variances(model):
    """Analytic per-component variances ``w_b^T G w_b`` of the mean model.

    Returns ``(first_order_vars, second_order_vars, third_order_vars)`` —
    dicts keyed by variable index / pair tuple / triple tuple. Handles both
    the uniform-basis path (batched block quadratic forms) and the mixed
    per-variable path (per-block Gram via the model accessors).
    """
    G1 = jnp.asarray(model.G1, dtype=jnp.float64) if model.G1 is not None else None
    D = model.D
    K1 = model.K1
    K2 = model.K2

    # First-order variances — handle both mixed and uniform modes
    _bn = getattr(model, 'basis_name', 'fourier')
    _il1 = getattr(model, 'include_linear_1', True)
    _is_mixed = getattr(model, 'var_specs', None) is not None
    w1 = jnp.asarray(model.mean_model.w1, dtype=jnp.float64)
    first_order_vars = {}

    if _is_mixed:
        # Mixed mode: per-variable Gram and block sizes
        for i in range(D):
            wi = jnp.asarray(model.mean_model.get_coefficients_for_variable(i),
                             dtype=jnp.float64)
            Gi = jnp.asarray(model.mean_model.get_var_gram(i), dtype=jnp.float64)
            var_i = jnp.maximum(0.0, wi @ Gi @ wi)
            first_order_vars[i] = float(var_i)
    else:
        block1 = basis_size(K1, _il1, _bn)
        v1 = _block_variances(w1, G1, D, block1)
        for i in range(D):
            first_order_vars[i] = float(v1[i])

    # Second-order variances — handle ragged pairs (mixed G_i ⊗ G_j, or
    # per-pair-K2 G(K2_p) ⊗ G(K2_p)) via the model accessors
    second_order_vars = {}
    _has_pairs = model.pair_indices is not None and (
        K2 > 0 or _is_mixed)
    if _has_pairs and model.pair_indices is not None:
        w2 = jnp.asarray(model.mean_model.w2, dtype=jnp.float64)
        P = model.pair_indices.shape[0]

        if model.mean_model.pair_block_info is not None:
            # Ragged pair blocks (mixed per-variable basis OR per-pair K2):
            # per-pair Gram via get_pair_gram
            for p in range(P):
                wp = jnp.asarray(model.mean_model.get_coefficients_for_pair(p),
                                 dtype=jnp.float64)
                Gp = jnp.asarray(model.mean_model.get_pair_gram(p),
                                 dtype=jnp.float64)
                var_p = jnp.maximum(0.0, wp @ Gp @ wp)
                i, j = int(model.pair_indices[p, 0]), int(model.pair_indices[p, 1])
                second_order_vars[(i, j)] = float(var_p)
        elif K2 > 0:
            incl_lin_2 = getattr(model, 'include_linear_2', True)
            G2 = jnp.asarray(model.G2, dtype=jnp.float64) if model.G2 is not None else build_gram_matrix_2d(build_gram_matrix(K2, incl_lin_2, _bn))
            G2 = jnp.asarray(G2, dtype=jnp.float64)
            block2 = basis_size(K2, incl_lin_2, _bn) ** 2

            v2 = _block_variances(w2, G2, P, block2)
            for p in range(P):
                i, j = int(model.pair_indices[p, 0]), int(model.pair_indices[p, 1])
                second_order_vars[(i, j)] = float(v2[p])

    # Third-order variances
    third_order_vars = {}
    K3 = getattr(model, 'K3', 0)
    if K3 > 0 and model.triple_indices is not None:
        if _is_mixed:
            # Uniform block3 slicing below would read wrong offsets for
            # per-variable block sizes. Unreachable today (the mixed path
            # forces K3=0), but guard rather than silently mis-slice.
            raise NotImplementedError(
                "third-order Sobol variances are not supported for mixed "
                "per-variable bases (uniform block slicing).")
        incl_lin_3 = getattr(model, 'include_linear_3', True)
        G3 = jnp.asarray(model.G3, dtype=jnp.float64) if model.G3 is not None else build_gram_matrix_3d(build_gram_matrix(K3, incl_lin_3, _bn))
        G3 = jnp.asarray(G3, dtype=jnp.float64)
        w3 = jnp.asarray(model.mean_model.w3, dtype=jnp.float64)
        block3 = basis_size(K3, incl_lin_3, _bn) ** 3
        T = model.triple_indices.shape[0]

        v3 = _block_variances(w3, G3, T, block3)
        for t in range(T):
            i, j, k = (int(model.triple_indices[t, pos]) for pos in range(3))
            third_order_vars[(i, j, k)] = float(v3[t])

    return first_order_vars, second_order_vars, third_order_vars


def _mean_residual_variance(model, residual_measure, x_data, cube_points):
    """Mean residual-net variance under the chosen measure.

    Returns ``(residual_var, res_pred, res_points)`` where ``res_pred`` is ĝ
    on ``res_points`` (kept for the fidelity cross-term) — both ``None`` when
    no residual net ran or no evaluation points are available. QMC (default)
    uses the shared cube sample; 'empirical' uses ``x_data``.
    """
    residual_var = 0.0
    res_pred = None
    res_points = None
    if model.residual_net is not None:
        if residual_measure == 'qmc':
            res_points = cube_points()
            rp = predict_residual_batch(model.residual_net, res_points)
        elif x_data is not None:
            res_points = x_data
            rp = predict_residual_batch(model.residual_net, res_points)
        else:
            rp = None
        if rp is not None:
            if rp.ndim > 1:
                rp = rp.squeeze(-1)
            residual_var = float(jnp.var(rp))
            res_pred = rp
    return residual_var, res_pred, res_points


def _fidelity_and_core_total(model, first, second, third, residual_var,
                             res_pred, res_points):
    """Structural fidelity 𝔉, orthogonality cross-term, and the labeled
    core/total Sobol blocks (v06 §3.2/§8; M3/DEC-032).

    ``V_core`` sums the SAME first/second/third variances the legacy fractions
    use — one shared source of truth for the retained-order set, so the two
    normalizations cannot drift apart. The total block is built as ``𝔉·core``
    so ``S^total = 𝔉·S^core`` holds by construction. Returns
    ``(fidelity_dict, mean_sobol_core, mean_sobol_total)``.
    """
    D = model.D
    v_core = (sum(first.values()) + sum(second.values()) + sum(third.values()))

    # Orthogonality cross-term Ĉov(f̂_core, ĝ) on the SAME residual-measure points
    # (§8 ~1122-1128): one extra evaluation of the structured mean on a sample
    # already in memory; identically 0 when no residual stage ran. Reported
    # beside 𝔉, never folded into it.
    cross_cov = 0.0
    if res_pred is not None and res_points is not None:
        phi1_c = model.build_phi1(res_points)
        phi2_c = model.build_phi2(res_points)
        phi3_c = model.build_phi3(res_points)
        f_core = np.asarray(model.mean_model.predict(phi1_c, phi2_c, phi3_c),
                            dtype=np.float64)
        g_hat = np.asarray(res_pred, dtype=np.float64)
        cross_cov = float(np.mean((f_core - f_core.mean()) * (g_hat - g_hat.mean())))

    fidelity = compute_fidelity(v_core, residual_var, cross_cov)
    F_fid = fidelity['value']

    mean_sobol_core = _core_sobol_block(D, first, second, third, v_core)
    mean_sobol_total = {'first_order': {}, 'second_order': {}, 'third_order': {},
                        'total_order': {}}
    for order in ('first_order', 'second_order', 'third_order', 'total_order'):
        for key, v in mean_sobol_core[order].items():
            mean_sobol_total[order][key] = F_fid * v
    return fidelity, mean_sobol_core, mean_sobol_total


def compute_fidelity(v_core: float, residual_var: float,
                     cross_cov: float = 0.0) -> dict:
    """Structural fidelity 𝔉 and the orthogonality-defect diagnostic (v06 §8).

    Single source of truth for the core→total bridge (M3/DEC-032). With
    ``V_core = Σ_{u≠∅} Var(f̂_u)`` (the retained structured orders) and
    ``Var(ĝ)`` the residual/NN variance,

        𝔉 = V_core / (V_core + Var(ĝ)),        Ŝ_u^total = 𝔉 · Ŝ_u^core

    (Eq. ``fidelity``; §3.2 ``S_total = ρ_k·S_core``). ``1 − 𝔉`` is the honest
    interpretability gap. ``𝔉 ≡ 1`` when no residual stage ran.

    ``cross_cov`` is ``Ĉov(f̂_core, ĝ)`` under the shared (QMC-cube) measure. The
    manuscript's decomposition is exact only when core and residual are orthogonal;
    when orthogonality is empirical, ``Var(f̂) = V_core + Var(ĝ) + 2·Ĉov`` and the
    reported ``orthogonality_defect = 2·Ĉov / Var(f̂)`` measures how far the identity
    is from exact. Per §8 the cross term is reported *beside* 𝔉, never folded into it
    (folding would destroy its meaning: 𝔉 is the exact quantity under the orthogonal
    measure, the defect is the departure from that measure).

    𝔉 is *model-internal*: a decomposition of the FITTED function's variance, not of
    the data. Do not read it as R² (fitted-vs-data) nor as ρ_k (the population estimand
    it estimates). "Internal R² of the interpretable core against the full fit" is the
    accurate one-liner — the share of the fit ``f̂`` that the interpretable structure
    ``f̂_core`` carries, with ``1−𝔉`` the share living in the residual ``ĝ`` (DEC-034).

    Returns a dict with ``value`` (𝔉), ``var_core``, ``var_residual``,
    ``cross_covariance``, ``orthogonality_defect``, and the machine-readable
    ``conditional_on_residual_variance`` flag (the total interval treats Var(ĝ) as
    fixed — the same conditionality convention the core CI already carries).
    """
    denom = v_core + residual_var
    fid = float(v_core / denom) if denom > 0 else 1.0
    var_total = v_core + residual_var + 2.0 * cross_cov
    defect = float(2.0 * cross_cov / var_total) if var_total > 0 else 0.0
    return {
        'value': fid,
        'var_core': float(v_core),
        'var_residual': float(residual_var),
        'cross_covariance': float(cross_cov),
        'orthogonality_defect': defect,
        'conditional_on_residual_variance': True,
    }


def compute_sobol_indices(
    model,
    x_data: Optional[jnp.ndarray] = None,
    residual_measure: str = 'qmc',
    qmc_n: int = 1 << 16,
    qmc_seed: int = 0,
    inputs_independent_by_design: bool = False,
) -> dict:
    """Compute the full dual Sobol spectrum.

    Args:
        model: HiFiANOVA instance
        x_data: (N, D) data — only used for correlative indices and (when
            ``residual_measure='empirical'``) the legacy residual variance.
        residual_measure: measure under which the residual-network variance
            (mean model and Stage-D variance model) is estimated.
            ``'qmc'`` (default) — ``Var_uniform(NN)`` via a deterministic Sobol
            sample of the cube [0,1]^D, i.e. the SAME uniform measure the
            analytic Gram forms ``w^T G w`` use, so numerator and denominator are
            measure-coherent. ``'empirical'`` — legacy behaviour: variance of the
            residual over ``x_data`` (the data measure). See DEC-021.
        qmc_n: QMC sample size for ``residual_measure='qmc'`` (rounded up to a
            power of two; 2^16 by default).
        qmc_seed: QMC scramble seed (fixes the estimate for reproducibility).

    Returns:
        dict with mean_sobol, log_variance_sobol, variance_accounting. The
        deprecated ``variance_sobol`` read alias returns the same log-scale block.

    Measure (residual variance): the parametric component variances are analytic
    Gram quadratic forms ``w_b^T G w_b`` — the exact variance under a uniform
    input measure on [0,1]^D. With ``residual_measure='qmc'`` (default) the
    residual-network term is ``Var_uniform(NN)`` estimated by QMC on the cube, so
    every term in ``total_var`` lives under the *same* uniform measure and the
    reported Sobol fractions ``S_i = V_i / V_tot`` are a coherent structural
    decomposition of the learned function. (With ``'empirical'`` the residual
    term reverts to the variance over ``x_data`` — a *different* measure for
    non-uniform inputs — so the fractions then mix an analytic numerator with an
    empirical one; this is retained only for backward comparison.) The
    unnormalised ``w^T G w`` (``variance_accounting``) is the primary,
    measure-consistent attribution quantity; normalization is a presentation
    choice. For fully parametric models (no residual_net) all terms are analytic.
    """
    D = model.D

    # Canonical keys are physically stored. ResultAliasDict retains the
    # one-release ``variance_sobol`` read alias without duplicating persistence.
    results = ResultAliasDict()

    # --- Mean Sobol indices ---
    _bn = getattr(model, 'basis_name', 'fourier')
    first_order_vars, second_order_vars, third_order_vars = (
        _mean_component_variances(model))

    # Residual variance. Default: Var_uniform(NN) via QMC on the cube — the same
    # uniform measure as the analytic Gram forms above (measure-coherent). The
    # QMC cube sample is drawn once and reused for the variance-model residual.
    _qmc_pts = None

    def _cube_points():
        nonlocal _qmc_pts
        if _qmc_pts is None:
            from .qmc import sobol_cube_sample
            _qmc_pts = jnp.asarray(sobol_cube_sample(D, qmc_n, qmc_seed),
                                   dtype=jnp.float64)
        return _qmc_pts

    residual_var, _res_pred, _res_points = _mean_residual_variance(
        model, residual_measure, x_data, _cube_points)

    # Total variance
    total_var = (sum(first_order_vars.values()) +
                 sum(second_order_vars.values()) +
                 sum(third_order_vars.values()) +
                 residual_var)

    mean_sobol = _normalized_sobol_block(
        D, first_order_vars, second_order_vars, third_order_vars,
        residual_var, total_var)
    results['mean_sobol'] = mean_sobol

    # --- Structural fidelity 𝔉 + labeled core/total shares (v06 §3.2/§8; M3/DEC-032) ---
    # ``mean_sobol`` (legacy) divides by ``total_var = V_core + Var(ĝ)`` — the TOTAL
    # shares. The labeled core/total blocks and the 𝔉 bridge come from the shared
    # helper (one source of truth for V_core, so the normalizations cannot drift).
    # ``mean_sobol`` is the back-compat alias of ``mean_sobol_total``.
    fidelity, mean_sobol_core, mean_sobol_total = _fidelity_and_core_total(
        model, first_order_vars, second_order_vars, third_order_vars,
        residual_var, _res_pred, _res_points)
    results['fidelity'] = fidelity
    results['mean_sobol_core'] = mean_sobol_core
    results['mean_sobol_total'] = mean_sobol_total

    # --- Log-variance Sobol indices (if heteroscedastic) ---
    if model.variance_model is not None:
        results['log_variance_sobol'] = _variance_sobol_block(
            model, _bn, residual_measure, x_data, _cube_points)

    # --- Variance accounting ---
    results['variance_accounting'] = {
        'first_order_total': sum(first_order_vars.values()),
        'second_order_total': sum(second_order_vars.values()),
        'third_order_total': sum(third_order_vars.values()),
        'residual': residual_var,
        'residual_nn': residual_var,  # backward compat alias
        'total_model_variance': total_var,
        'per_variable_first_order': first_order_vars,
        'per_pair_second_order': second_order_vars,
        'per_triple_third_order': third_order_vars,
        # Measure under which `residual` was estimated: 'qmc' (uniform cube,
        # coherent with the analytic w^T G w forms) or 'empirical' (data measure).
        'residual_measure': (residual_measure if model.residual_net is not None
                             else 'none'),
    }

    # --- Correlative indices (if data provided) ---
    if x_data is not None:
        corr_results = compute_correlative_sobol(model, x_data)
        results['correlative_sobol'] = corr_results

    # Input-measure contract: the structural indices describe the fitted function
    # under the reference INDEPENDENT product measure. Independence is an
    # assumption, verified only if the caller asserts it (e.g. a controlled
    # experiment with independently generated inputs); for observational data it
    # must be justified externally, and dependent-input attribution is out of scope.
    results['input_assumption'] = 'independent_product_measure'
    results['input_assumption_verified'] = bool(inputs_independent_by_design)

    return results


def _variance_sobol_block(model, _bn, residual_measure, x_data, cube_points):
    """Variance-model Sobol spectrum: analytic per-block Gram variances,
    the (projected) variance-residual term, shared normalization, and the
    per-block accounting dict. Pure move from ``compute_sobol_indices``;
    ``cube_points`` is the memoized QMC-cube closure shared with the mean
    residual so both residual terms use the same sample.
    """
    vm = model.variance_model
    D = model.D
    Kh = model.Kh
    _ilh1 = getattr(model, 'include_linear_h1', getattr(vm, 'include_linear_h1', True))
    _ilh2 = getattr(model, 'include_linear_h2', getattr(vm, 'include_linear_h2', True))
    _ilh3 = getattr(model, 'include_linear_h3', getattr(vm, 'include_linear_h3', True))
    _bn_h = getattr(vm, 'basis_name', _bn)
    Gh = build_gram_matrix(Kh, _ilh1, _bn_h)
    Gh = jnp.asarray(Gh, dtype=jnp.float64)
    wh = jnp.asarray(vm.w1, dtype=jnp.float64)
    block_h = basis_size(Kh, _ilh1, _bn_h)

    # First-order variance. With a variance-variable subset (BR-01), w1 holds
    # only the included variables' blocks; excluded variables report exactly 0
    # (sigma² is flat along them by user assertion) and keys still span all D.
    var_h_first = {}
    _vv = getattr(vm, 'variance_variables', None)
    if _vv is not None:
        vh1 = _block_variances(wh, Gh, len(_vv), block_h)
        for i in range(D):
            var_h_first[i] = 0.0
        for pos, i in enumerate(_vv):
            var_h_first[int(i)] = float(vh1[pos])
    else:
        vh1 = _block_variances(wh, Gh, D, block_h)
        for i in range(D):
            var_h_first[i] = float(vh1[i])

    # Second-order variance (if present)
    var_h_second = {}
    K2h = getattr(vm, 'K2h', 0)
    if K2h > 0 and hasattr(vm, 'w2') and len(vm.w2) > 0:
        G2h = build_gram_matrix_2d(build_gram_matrix(K2h, _ilh2, _bn_h))
        G2h = jnp.asarray(G2h, dtype=jnp.float64)
        w2h = jnp.asarray(vm.w2, dtype=jnp.float64)
        block_h2 = basis_size(K2h, _ilh2, _bn_h) ** 2
        pair_idx_h = vm.pair_indices_h
        if pair_idx_h is not None:
            Ph = pair_idx_h.shape[0]
            vh2 = _block_variances(w2h, G2h, Ph, block_h2)
            for p in range(Ph):
                i, j = int(pair_idx_h[p, 0]), int(pair_idx_h[p, 1])
                var_h_second[(i, j)] = float(vh2[p])

    # Third-order variance (if present)
    var_h_third = {}
    K3h = getattr(vm, 'K3h', 0)
    if K3h > 0 and hasattr(vm, 'w3') and len(vm.w3) > 0:
        G3h = build_gram_matrix_3d(build_gram_matrix(K3h, _ilh3, _bn_h))
        G3h = jnp.asarray(G3h, dtype=jnp.float64)
        w3h = jnp.asarray(vm.w3, dtype=jnp.float64)
        block_h3 = basis_size(K3h, _ilh3, _bn_h) ** 3
        triple_idx_h = getattr(vm, 'triple_indices_h', None)
        if triple_idx_h is not None:
            Th = triple_idx_h.shape[0]
            vh3 = _block_variances(w3h, G3h, Th, block_h3)
            for t in range(Th):
                i, j, k = (int(triple_idx_h[t, pos]) for pos in range(3))
                var_h_third[(i, j, k)] = float(vh3[t])

    # Variance residual contribution. Same measure treatment as the mean
    # residual above: default QMC under the uniform cube measure so this
    # empirically-estimated term is coherent with the analytic var-Sobol Gram
    # forms in the shared total_var_h denominator (DEC-021). 'empirical'
    # reverts to the variance over x_data.
    var_h_residual = 0.0
    if hasattr(vm, 'has_variance_residual') and vm.has_variance_residual:
        x_res = None
        if residual_measure == 'qmc':
            x_res = cube_points()
        elif x_data is not None:
            x_res = x_data
        if x_res is not None:
            z_h = vm.variance_residual.build_features(x_res)
            if (vm.variance_residual.proj_coeffs.ndim >= 2 and
                    vm.variance_residual.proj_coeffs.shape[0] > 0):
                # proj_coeffs was fitted against the FULL variance design
                # [psi1|psi2|psi3] (trainer Stage D), so the projection here
                # must rebuild the same design — psi1 alone mismatches
                # whenever K2h/K3h > 0.
                from ..core.features import (build_second_order_features,
                                             build_third_order_features)
                psi_parts = [model.build_psi1(x_res)]
                _pair_idx_h = getattr(vm, 'pair_indices_h', None)
                if K2h > 0 and _pair_idx_h is not None:
                    psi_parts.append(build_second_order_features(
                        x_res, K2h, _pair_idx_h,
                        include_linear=_ilh2, basis_name=_bn_h))
                _triple_idx_h = getattr(vm, 'triple_indices_h', None)
                if K3h > 0 and _triple_idx_h is not None:
                    psi_parts.append(build_third_order_features(
                        x_res, K3h, _triple_idx_h,
                        include_linear=_ilh3, basis_name=_bn_h))
                psi_fourier = (jnp.concatenate(psi_parts, axis=1)
                               if len(psi_parts) > 1 else psi_parts[0])
                z_h_proj = z_h - psi_fourier @ vm.variance_residual.proj_coeffs
            else:
                z_h_proj = z_h
            h_res = z_h_proj @ vm.w_var_residual
            var_h_residual = float(jnp.var(h_res))

    total_var_h = (sum(var_h_first.values()) +
                   sum(var_h_second.values()) +
                   sum(var_h_third.values()) +
                   var_h_residual)

    log_variance_sobol = _normalized_sobol_block(
        D, var_h_first, var_h_second, var_h_third,
        var_h_residual, total_var_h)

    log_variance_sobol['variance_accounting'] = {
        'first_order_total': sum(var_h_first.values()),
        'second_order_total': sum(var_h_second.values()),
        'third_order_total': sum(var_h_third.values()),
        'residual': var_h_residual,
        'total': total_var_h,
        'per_variable_first_order': var_h_first,
        'per_pair_second_order': var_h_second,
        'per_triple_third_order': var_h_third,
        'residual_measure': (residual_measure
                             if (hasattr(vm, 'has_variance_residual')
                                 and vm.has_variance_residual) else 'none'),
    }

    return log_variance_sobol


def _mean_component_outputs(model, x_data):
    """Per-component fitted outputs ``f̂_u(x)`` on ``x_data`` — every retained
    STRUCTURED component (first-, second-, and third-order).

    The orthogonal residual ``ĝ`` (residual network) is deliberately excluded:
    per Manuscript_Theoryv06 §11.6 the correlative total
    ``f̂_tot = Σ_{u≠∅} f̂_u`` ranges over the structured ANOVA components, and
    the residual is accounted for separately through the structural fidelity 𝔉
    (§8, Eq. ``fidelity``) rather than folded into the correlative denominator.

    Returns ``(orders, keys, outputs)``: ``orders`` a list of 1/2/3 tagging each
    component's interaction order, ``keys`` the matching variable index / pair
    tuple / triple tuple, ``outputs`` an ``(n_comp, N)`` float64 array of the
    component outputs (uncentered; the caller centers).
    """
    D = model.D
    mm = model.mean_model
    _bn = getattr(model, 'basis_name', 'fourier')
    orders, keys, outs = [], [], []

    # --- First order (all D components are always present) ---
    phi1 = np.asarray(model.build_phi1(x_data), dtype=np.float64)
    _is_mixed = getattr(model, 'var_specs', None) is not None
    if _is_mixed:
        for i in range(D):
            _, _, _, block_i, offset_i = mm.var_specs[i]
            wi = np.asarray(mm.get_coefficients_for_variable(i), dtype=np.float64)
            orders.append(1)
            keys.append(i)
            outs.append(phi1[:, offset_i: offset_i + block_i] @ wi)
    else:
        block1 = basis_size(model.K1, getattr(model, 'include_linear_1', True), _bn)
        for i in range(D):
            wi = np.asarray(mm.get_coefficients_for_variable(i), dtype=np.float64)
            orders.append(1)
            keys.append(i)
            outs.append(phi1[:, i * block1: (i + 1) * block1] @ wi)

    # --- Second order (retained pairs) ---
    if model.pair_indices is not None:
        phi2 = model.build_phi2(x_data)
        if phi2 is not None:
            phi2 = np.asarray(phi2, dtype=np.float64)
            for p in range(model.pair_indices.shape[0]):
                wp = np.asarray(mm.get_coefficients_for_pair(p), dtype=np.float64)
                if mm.pair_block_info is not None:
                    _, _, _, _, block, offset = mm.pair_block_info[p]
                else:
                    block = basis_size(
                        model.K2, getattr(model, 'include_linear_2', True), _bn) ** 2
                    offset = p * block
                i, j = int(model.pair_indices[p, 0]), int(model.pair_indices[p, 1])
                orders.append(2)
                keys.append((i, j))
                outs.append(phi2[:, offset: offset + block] @ wp)

    # --- Third order (retained triples) ---
    K3 = getattr(model, 'K3', 0)
    if K3 > 0 and getattr(model, 'triple_indices', None) is not None:
        phi3 = model.build_phi3(x_data)
        if phi3 is not None:
            phi3 = np.asarray(phi3, dtype=np.float64)
            block3 = basis_size(
                K3, getattr(model, 'include_linear_3', True), _bn) ** 3
            for t in range(model.triple_indices.shape[0]):
                wt = np.asarray(mm.get_coefficients_for_triple(t), dtype=np.float64)
                i, j, k = (int(model.triple_indices[t, pos]) for pos in range(3))
                orders.append(3)
                keys.append((i, j, k))
                outs.append(phi3[:, t * block3: (t + 1) * block3] @ wt)

    outputs = np.asarray(outs, dtype=np.float64)  # (n_comp, N)
    return orders, keys, outputs


def compute_correlative_sobol(model, x_data: jnp.ndarray,
                              return_full_covariance: bool = False) -> dict:
    """Correlative Sobol indices — joint-law covariance allocation over ALL
    retained structured components (Manuscript_Theoryv06 §11.6).

    ROLE: this is an **independence-assumption diagnostic**, not an official
    supported estimand for correlated-input attribution. HiFi-ANOVA's reported
    attribution is the *structural* (reference-measure) spectrum, which assumes
    independent inputs. When inputs are dependent, these correlative shares
    measure how far the attribution moves under the observed joint law — a guard
    on the independence assumption. Principled dependent-input attribution
    (Shapley effects / generalized hierarchically-orthogonal ANOVA) is out of
    scope (manuscript outlook). See ``correlation_diagnostic`` for the paired
    linear + nonlinear (distance-correlation) dependence check.

    For every retained nonconstant structured component ``u`` (first-, second-,
    and third-order), with ``f̂_tot = Σ_{u≠∅} f̂_u`` the sum of all such
    components evaluated on the data,

        S^corr_u = Cov(f̂_u, f̂_tot) / Var(f̂_tot).

    By linearity of covariance ``Σ_{u≠∅} S^corr_u = 1`` **identically**,
    regardless of input dependence, because
    ``Σ_u Cov(f̂_u, f̂_tot) = Cov(f̂_tot, f̂_tot) = Var(f̂_tot)``. What input
    dependence changes is (a) the *individual* terms, which may be negative or
    exceed 1, and (b) the value of any *partial* collection: a sum over
    first-order components alone, when interaction components are also retained,
    need not equal 1. For a purely first-order model the mains ARE the complete
    collection, so the first-order indices sum to 1 exactly.

    Unlike the structural indices (analytic Gram forms ``w^T G w`` under the
    product-of-marginals reference measure), these are joint-law quantities:
    they use the actual covariance of component outputs on ``x_data`` and so
    account for input correlations. For independent inputs correlative ≈
    structural.

    The orthogonal residual ``ĝ`` is NOT part of ``f̂_tot`` (§8/§11.6): it is
    reported separately through the structural fidelity 𝔉. The correlative
    spectrum therefore allocates the *structured* part of the fit among its
    components; when a large residual is present, read it alongside 𝔉.

    The identity holds only when the Cov numerator and the Var denominator use
    the same estimator, so both use the ddof=0 (÷N) empirical covariance built
    below (a mismatched ddof=1 numerator used to inflate each index by N/(N-1)).
    Analytics run in float64 like the structural path, so the identity holds to
    machine precision rather than to a float32 fit's ~1e-7 round-off.

    Cost is O(N·C) time and memory for C = D + P + T retained components — each
    share needs only Cov(f̂_u, f̂_tot), not the full C×C covariance. The full
    matrix is O(N·C²) and is built only on request (``return_full_covariance``).

    Args:
        model: fitted HiFiANOVA
        x_data: (N, D) input data (transformed to [0,1])
        return_full_covariance: if True, also return the (C, C)
            ``component_covariance_matrix`` (O(N·C²)); default False.

    Returns:
        dict with:
          first_order:  {i: S_i^corr}
          second_order: {(i,j): S_ij^corr}
          third_order:  {(i,j,k): S_ijk^corr}
          sum_of_correlative_indices: Σ over the complete reported collection
              (≡1 up to round-off; 0.0 in the degenerate zero-variance case)
          first_order_sum: Σ over first-order components only (a partial
              collection — need not equal 1 when interactions are retained)
          cross_correlation_matrix / covariance_matrix: (D, D) FIRST-ORDER
              component-output blocks
          max_abs_cross_correlation: max |off-diagonal| of the first-order
              component-output correlation — a COMPONENT-OUTPUT quantity, NOT an
              input-correlation measure (it is 0 when first-order coeffs are 0
              even if inputs are perfectly correlated). For an input-dependence
              gate use ``correlation_diagnostic`` (linear + distance correlation).
          component_output_correlation_level: 'clean'/'mild'/'strong' from
              max_abs_cross_correlation (component-output, same caveat)
          component_covariance_matrix: (C, C) — only if return_full_covariance
          component_keys: ordered component keys (int / tuple)
          scope / denominator / residual_excluded / role /
              official_correlated_estimand: machine-readable contract
    """
    D = model.D
    orders, keys, outputs = _mean_component_outputs(model, x_data)
    outputs = np.asarray(outputs, dtype=np.float64)  # (C, N)
    N = outputs.shape[1]

    # Center once (ddof=0). Components are ~zero mean by construction, but we
    # center explicitly so the covariances are exact.
    centered = outputs - outputs.mean(axis=1, keepdims=True)  # (C, N)

    # f̂_tot = Σ_u f̂_u; denominator Var_N(f̂_tot) matches the ddof=0 numerator.
    total_c = centered.sum(axis=0)             # centered f̂_tot, (N,)
    total_var = float(np.mean(total_c * total_c))  # ddof=0 Var(f̂_tot)

    # S^corr_u = Cov_N(f̂_u, f̂_tot)/Var_N(f̂_tot). Cov with the total is one
    # matrix-vector product — O(N·C), no C×C matrix (reviewer: quadratic scaling).
    cov_with_total = (centered @ total_c) / N  # (C,)
    per_component = (cov_with_total / total_var) if total_var > 0 else np.zeros(len(keys))

    first_order, second_order, third_order = {}, {}, {}
    for s, order, key in zip(per_component, orders, keys):
        s = float(s)
        if order == 1:
            first_order[key] = s
        elif order == 2:
            second_order[key] = s
        else:
            third_order[key] = s

    # First-order (D, D) component-output covariance/correlation block. This is a
    # COMPONENT-OUTPUT diagnostic (0 when first-order coeffs are 0); it is NOT a
    # substitute for an input-correlation gate — see correlation_diagnostic.
    fo_idx = [i for i, o in enumerate(orders) if o == 1]
    fo_c = centered[fo_idx]                    # (D, N)
    fo_cov = (fo_c @ fo_c.T) / N               # (D, D)
    fo_std = np.sqrt(np.maximum(np.diag(fo_cov), 1e-20))
    corr_matrix = fo_cov / (fo_std[:, None] * fo_std[None, :])
    np.fill_diagonal(corr_matrix, 1.0)
    mask = ~np.eye(D, dtype=bool)
    max_abs_cross = float(np.max(np.abs(corr_matrix[mask]))) if D > 1 else 0.0

    if max_abs_cross < 0.1:
        comp_level = 'clean'
    elif max_abs_cross < 0.3:
        comp_level = 'mild'
    else:
        comp_level = 'strong'

    result = {
        'first_order': first_order,
        'second_order': second_order,
        'third_order': third_order,
        'sum_of_correlative_indices': float(sum(per_component)),
        'first_order_sum': float(sum(first_order.values())),
        'cross_correlation_matrix': corr_matrix,
        'covariance_matrix': fo_cov,
        'component_keys': list(keys),
        'max_abs_cross_correlation': max_abs_cross,
        'component_output_correlation_level': comp_level,
        'scope': 'all_retained_structured_components',
        'denominator': 'var_of_summed_retained_structured_components',
        'residual_excluded': True,
        'role': 'independence_assumption_diagnostic',
        'official_correlated_estimand': False,
    }
    if return_full_covariance:
        result['component_covariance_matrix'] = (centered @ centered.T) / N
    return result
