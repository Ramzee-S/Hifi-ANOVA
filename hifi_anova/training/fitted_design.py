"""Fitted-design record: the design the trainer actually solved.

The one-call API used to *rebuild* an independent, unweighted, config-penalty
ridge for its post-fit diagnostics (``api.py``), so the reported ``sigma_hat`` /
``df`` / LOO / Sobol CIs could describe a *different model* than the one returned
(an internal design note motivates this). This module carries the trainer's real
fit — design matrix, penalty, centered target, block layout, and (for Stage-D)
the GLS precision weights — so every diagnostic is computed from the fit the user
gets back.

Two-fit convention (theory manuscript, Theorem ``projection`` Part ii):
  * **predict / diagnose** (σ̂, df, LOO, epistemic CI) from the precision-weighted
    (GLS) fit — ``sample_weights`` carries ``W = diag(1/σ²(xₙ))``;
  * **attribute** (Sobol point indices + their HC3 CI) from the *unit-weight*
    (``W = I``) companion in ``interpretive`` — the HC3 sandwich is already
    heteroscedasticity-robust, so attribution CIs are *not* reweighted.
In the homoscedastic case the two coincide: ``sample_weights is None`` and
``interpretive is None`` mean "this record is both fits".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


# Mean-estimator convention tags (DEC-039 provenance). A saved artifact or a
# downstream consumer needs to distinguish *how* the fitted mean intercept /
# features were estimated, because the Stage-D default flipped estimator vintages
# (legacy fixed-intercept/uncentered solve → profiled joint-GLS). These record
# the EFFECTIVE convention actually used, not merely the requested flag. Single
# source of truth — imported by the trainer (results/record) and by model.io
# (save/load metadata) so the strings are not duplicated across modules.
MEAN_INTERCEPT_PROFILED_JOINT_GLS = "profiled_joint_gls"
"""Stage-D weighted mean solved as the penalized-GLS optimum: weighted-center
BOTH y and Φ (profiled unpenalized intercept). The DEC-039 default
(``stage_d_joint_gls_mean=True``)."""
MEAN_INTERCEPT_LEGACY_FIXED = "legacy_fixed_intercept_uncentered_features"
"""Legacy Stage-D mean: fixed intercept ``f0 = Σwₙyₙ/Σwₙ`` with a weighted ridge
on *uncentered* Φ (NOT the penalized-GLS optimum). Only under the non-default
``stage_d_joint_gls_mean=False`` compatibility flag."""
MEAN_INTERCEPT_UNWEIGHTED = "unweighted_centered"
"""Ordinary unit-weight centered ridge mean: the homoscedastic (Stage A/B/C,
mixed) fit, and the mean a Stage-D fit ships when it reverts to constant variance
or falls back to the unit-weight (Hoeffding-projection) mean."""
MEAN_INTERCEPT_LEGACY_UNKNOWN = "legacy_unknown"
"""Interpretation for a loaded heteroscedastic artifact that predates this field:
its Stage-D mean vintage (legacy vs joint-GLS) cannot be recovered from metadata.
Never written by a fresh save."""


@dataclass
class DesignBlock:
    """One ANOVA order's contiguous column block inside ``FittedDesign.Phi``.

    A block spans *all* groups of a single interaction order (order 1: the ``D``
    first-order variables; order 2: the ``P`` retained pairs; order 3: the ``T``
    retained triples), laid out group-major with ``basis_size(K, ...)`` (order 1)
    or its square/cube (orders 2/3) columns per group — exactly how the trainer
    concatenates ``[phi1 | phi2 | phi3]``.

    Attributes:
        order: interaction order (1, 2, or 3).
        K: harmonic/degree/scale parameter for this order (K1 / K2 / K3).
        basis_name: 'fourier' | 'legendre' | 'haar' for this order.
        include_linear: whether this order's per-variable basis keeps the linear
            term (Fourier only; ignored for Legendre/Haar).
        columns: column ``slice`` of this order within ``Phi``.
        gram: the order's Gram matrix (G1 (B,B); G2 (B²,B²); G3 (B³,B³)) used to
            turn coefficients into component variances for Sobol.
        n_groups: number of groups in this order (D, P, or T).
        indices: (n_groups, order) integer array naming each group's variables —
            ``pair_indices`` (order 2) / ``triple_indices`` (order 3); ``None``
            for order 1 (group i is simply variable i).
    """

    order: int
    K: int
    basis_name: str
    include_linear: bool
    columns: slice
    gram: np.ndarray
    n_groups: int
    indices: Optional[np.ndarray] = None


@dataclass
class VarianceDesign:
    """The fitted log-variance sub-problem of a Stage-D (heteroscedastic) fit.

    Carries the variance design and penalty the trainer's Newton solve actually
    used, so the one-call API can compute the manuscript's Tier-II one-step LOO
    jackknife of the *joint* heteroscedastic model — and, on request, the
    Tier-III exact nested refit — from the design the trainer solved (the same
    fitted-design-record discipline as the mean block; see
    :func:`hifi_anova.analysis.automl.joint_loo`). ``None`` on a homoscedastic fit.

    The log-variance model is ``h(x) = h0 + psi(x)^T w_h`` with
    ``sigma^2(x) = exp(h(x))`` (clipped to ``[-LOG_VAR_CLIP, LOG_VAR_CLIP]`` in
    prediction). ``Psi`` / ``reg_var`` / ``w_h`` are the NON-augmented variance
    columns (the intercept ``h0`` is a separate, unpenalized fixed effect —
    ``newton_solve_log_variance`` augments ``[1, Psi]`` internally). All arrays
    are float64; ``Psi`` is ``[psi1 | psi2 | psi3 | z_h_proj]`` exactly as the
    Newton solve saw it (any variance-residual projection columns included).

    Attributes:
        Psi: (N, F_h) variance features the Newton solve used.
        reg_var: (F_h,) variance ridge penalty (``lambda_h`` times the shape).
        w_h: (F_h,) fitted log-variance coefficients (no ``h0``).
        h0: fitted log-variance intercept.
    """

    Psi: np.ndarray
    reg_var: np.ndarray
    w_h: np.ndarray
    h0: float


@dataclass
class FittedDesign:
    """The penalized design the trainer solved, plus its block layout.

    Everything a linear diagnostic needs is here, so ``ridge_analytics`` /
    ``sobol_confidence_intervals`` / the epistemic interval each become one call
    on the *real* fit rather than a rebuild.

    Attributes:
        Phi: (N, F) design matrix, float64. Columns are laid out
            ``[order-1 | order-2 | order-3 | residual…]``; ``blocks`` describe the
            structured prefix, any residual/NN tail is penalty-only.
        w: (F,) fitted coefficients (float64) — the trainer's solve. Diagnostics
            re-solve from ``Phi``/``reg_diag`` for a single float64 convention, so
            ``w`` is carried for provenance/consistency checks, not required.
        reg_diag: (F,) ridge penalty diagonal actually used (the real per-order
            penalties, *not* config padding).
        y_centered: (N,) target minus ``f0``.
        f0: intercept (unweighted mean for homoscedastic; the GLS weighted mean
            ``Σwₙyₙ/Σwₙ`` for Stage-D).
        D: number of input variables.
        sample_weights: (N,) GLS precision weights ``W = diag(1/σ²(xₙ))`` for a
            Stage-D fit; ``None`` ⇒ homoscedastic (unit weight).
        blocks: ordered list of :class:`DesignBlock` (orders present, ascending).
        interpretive: the unit-weight companion used for attribution; ``None`` ⇒
            this record is already unit-weight and serves both roles.
        mean_intercept_mode: the mean-estimator convention this record's ``f0`` /
            coefficients were solved under (DEC-039 provenance) — one of the
            ``MEAN_INTERCEPT_*`` tags. Defaults to the ordinary unit-weight
            centered ridge; a Stage-D caller sets it to the effective weighted
            convention (profiled joint-GLS or the legacy fixed-intercept solve).
    """

    Phi: np.ndarray
    w: np.ndarray
    reg_diag: np.ndarray
    y_centered: np.ndarray
    f0: float
    D: int
    sample_weights: Optional[np.ndarray] = None
    blocks: List[DesignBlock] = field(default_factory=list)
    interpretive: Optional["FittedDesign"] = None
    mean_intercept_mode: str = MEAN_INTERCEPT_UNWEIGHTED
    # Explicit per-group Sobol layout for a *mixed per-variable basis*, where
    # groups within an order have different column sizes and Gram matrices so the
    # uniform ``[phi1|phi2|phi3]`` reconstruction does not apply. Each entry is a
    # ``(order, key, columns, gram)`` tuple: ``key`` is the variable index (order
    # 1) or the ``(i, j)`` pair (order 2); ``columns`` is the group's column
    # ``slice`` in ``Phi``; ``gram`` its Gram. ``None`` ⇒ uniform layout (use
    # ``blocks`` / ``sobol_ci_kwargs`` the usual way).
    sobol_groups: Optional[List[tuple]] = None

    @property
    def is_weighted(self) -> bool:
        """True iff Stage-D precision weights are attached (GLS fit)."""
        return self.sample_weights is not None

    def block(self, order: int) -> Optional[DesignBlock]:
        """Return the :class:`DesignBlock` of the given order, or ``None``."""
        for b in self.blocks:
            if b.order == order:
                return b
        return None

    def attribution_record(self) -> "FittedDesign":
        """The record to attribute (Sobol) from: the unit-weight companion.

        For a homoscedastic fit that is ``self``; for a Stage-D fit it is
        ``interpretive`` (the ``W = I`` re-solve). Falls back to ``self`` if a
        companion was not attached.
        """
        return self.interpretive if self.interpretive is not None else self

    def sobol_ci_kwargs(self) -> dict:
        """Assemble the block-structure kwargs for ``sobol_confidence_intervals``.

        Mirrors the uniform ``[phi1|phi2|phi3]`` column layout the CI routine
        reconstructs internally from ``K``/Gram/counts, so passing these is
        equivalent to the trainer's real layout (residual/NN tail excluded — it
        does not enter a Sobol component).

        For a *mixed per-variable basis* (``sobol_groups`` set) the uniform
        reconstruction does not apply, so the explicit per-group layout is passed
        through as ``groups=`` instead.
        """
        if self.sobol_groups is not None:
            return dict(D=self.D, groups=self.sobol_groups)
        b1 = self.block(1)
        if b1 is None:
            raise ValueError("FittedDesign has no first-order block")
        kw = dict(
            D=self.D,
            K1=b1.K,
            G1=b1.gram,
            include_linear_1=b1.include_linear,
            basis_name=b1.basis_name,
        )
        b2 = self.block(2)
        if b2 is not None and b2.n_groups > 0:
            kw.update(K2=b2.K, P=b2.n_groups, G2=b2.gram, pair_indices=b2.indices)
        b3 = self.block(3)
        if b3 is not None and b3.n_groups > 0:
            kw.update(K3=b3.K, T=b3.n_groups, G3=b3.gram,
                      triple_indices=b3.indices)
        return kw


def _as_f64(a):
    return None if a is None else np.asarray(a, dtype=np.float64)


def build_record(
    Phi, w, reg_diag, y_train, D,
    K1, G1, include_linear_1, basis_name,
    *,
    f0=None, sample_weights=None,
    mean_intercept_mode=MEAN_INTERCEPT_UNWEIGHTED,
    K2=0, P=0, G2=None, pair_indices=None, include_linear_2=True,
    K3=0, T=0, G3=None, triple_indices=None, include_linear_3=True,
    fo_included=None, pair_block_info=None, pair_grams=None,
) -> FittedDesign:
    """Assemble a :class:`FittedDesign` from a trainer stage's design locals.

    Builds the ordered ``blocks`` list from the uniform ``[phi1|phi2|phi3]``
    column layout (order-1: ``D`` groups of ``basis_size(K1,…)`` columns; order-2:
    ``P`` groups of ``G2.shape[0]``; order-3: ``T`` groups of ``G3.shape[0]``),
    matching how the trainer concatenates the design. Any residual/NN columns of
    ``Phi`` beyond the third-order block are penalty-only (not part of any Sobol
    component) and simply left out of ``blocks``. All arrays are stored float64.

    Centering matches the API's float64 convention: ``y_centered = y64 − f0`` in
    float64. For a homoscedastic fit ``f0`` defaults to ``mean(y_train)`` computed
    the same way the API does (``np.mean`` on the native dtype); a Stage-D caller
    passes the GLS weighted intercept explicitly via ``f0``.

    Term-structure layouts (order-selective / per-pair-K2 fits): pass
    ``fo_included`` (ascending TRUE variable indices whose first-order blocks
    are in ``Phi``; ``None`` = all ``D``) and/or ``pair_block_info`` (the
    ragged per-pair layout ``(i, j, B, B, block, offset)`` with per-pair Grams
    in ``pair_grams``). Either one switches the record to the explicit
    ``sobol_groups`` per-group layout (the same mechanism mixed-basis fits
    use), since the uniform ``blocks`` reconstruction no longer applies. The
    default (both ``None``) path is byte-identical to before.
    """
    from ..core.features import basis_size

    Phi = _as_f64(Phi)
    reg_diag = _as_f64(reg_diag)
    y64 = _as_f64(y_train)
    if f0 is None:
        # Mirror api.py: np.mean on the original (float32) target, then center in
        # float64. Reproduces the reported σ̂/df/LOO bit-for-bit on the golden
        # (hypersensitive overfit) path.
        f0 = float(np.mean(np.asarray(y_train)))
    else:
        f0 = float(f0)
    y_centered = y64 - f0

    if fo_included is not None or pair_block_info is not None:
        if K3 > 0 and T > 0:
            raise NotImplementedError(
                "term-structure records (fo_included / pair_block_info) do "
                "not support third-order blocks.")
        b1 = basis_size(K1, include_linear_1, basis_name)
        inc = (list(fo_included) if fo_included is not None
               else list(range(D)))
        groups = []
        for pos, i in enumerate(inc):
            groups.append((1, int(i), slice(pos * b1, (pos + 1) * b1),
                           _as_f64(G1)))
        F1 = len(inc) * b1
        if pair_block_info is not None:
            if pair_grams is None or len(pair_grams) != len(pair_block_info):
                raise ValueError(
                    "pair_block_info requires a matching pair_grams sequence "
                    "(one Gram per ragged pair block).")
            for p, (i, j, Bi, Bj, block, offset) in enumerate(pair_block_info):
                groups.append((2, (int(i), int(j)),
                               slice(F1 + offset, F1 + offset + block),
                               _as_f64(pair_grams[p])))
        elif K2 > 0 and P > 0 and G2 is not None:
            G2_64 = _as_f64(G2)
            b2 = G2_64.shape[0]
            for p in range(P):
                i, j = int(pair_indices[p, 0]), int(pair_indices[p, 1])
                groups.append((2, (i, j),
                               slice(F1 + p * b2, F1 + (p + 1) * b2), G2_64))
        return FittedDesign(
            Phi=Phi,
            w=_as_f64(w),
            reg_diag=reg_diag,
            y_centered=y_centered,
            f0=f0,
            D=D,
            sample_weights=_as_f64(sample_weights),
            blocks=[],
            mean_intercept_mode=mean_intercept_mode,
            sobol_groups=groups,
        )

    b1 = basis_size(K1, include_linear_1, basis_name)
    F1 = D * b1
    blocks = [DesignBlock(
        order=1, K=K1, basis_name=basis_name, include_linear=include_linear_1,
        columns=slice(0, F1), gram=_as_f64(G1), n_groups=D)]
    off = F1

    if K2 > 0 and P > 0 and G2 is not None:
        G2 = _as_f64(G2)
        b2 = G2.shape[0]
        F2 = P * b2
        blocks.append(DesignBlock(
            order=2, K=K2, basis_name=basis_name, include_linear=include_linear_2,
            columns=slice(off, off + F2), gram=G2, n_groups=P,
            indices=(np.asarray(pair_indices) if pair_indices is not None
                     else None)))
        off += F2

    if K3 > 0 and T > 0 and G3 is not None:
        G3 = _as_f64(G3)
        b3 = G3.shape[0]
        F3 = T * b3
        blocks.append(DesignBlock(
            order=3, K=K3, basis_name=basis_name, include_linear=include_linear_3,
            columns=slice(off, off + F3), gram=G3, n_groups=T,
            indices=(np.asarray(triple_indices) if triple_indices is not None
                     else None)))
        off += F3

    return FittedDesign(
        Phi=Phi,
        w=_as_f64(w),
        reg_diag=reg_diag,
        y_centered=y_centered,
        f0=f0,
        D=D,
        sample_weights=_as_f64(sample_weights),
        blocks=blocks,
        mean_intercept_mode=mean_intercept_mode,
    )


def build_mixed_record(
    Phi, w, reg_diag, y_train, D,
    block_info, pair_block_info=None, pair_indices=None,
    *, f0=None,
) -> FittedDesign:
    """Assemble a :class:`FittedDesign` for a *mixed per-variable basis* fit.

    Unlike :func:`build_record`, each first-order variable (and each pair) has its
    own basis family / size, so groups within an order are not uniform. This
    builds the explicit ``sobol_groups`` per-group layout (column slice + Gram)
    that :func:`~hifi_anova.analysis.automl.sobol_confidence_intervals` consumes
    via its ``groups=`` path.

    Args:
        Phi, w, reg_diag, y_train, D: as in :func:`build_record`.
        block_info: order-1 layout from ``build_mixed_first_order_features`` —
            a tuple of ``(basis, K, include_linear, block_size, offset)`` per
            variable.
        pair_block_info: order-2 layout from ``build_mixed_second_order_features``
            — a tuple of ``(i, j, Bi, Bj, block_size, offset)`` per pair; the
            offsets are relative to the start of the order-2 sub-block. ``None``
            for a first-order-only (Stage-A) fit.
        pair_indices: unused for layout (pairs come from ``pair_block_info``);
            accepted for symmetry with the caller.
        f0: intercept; defaults to ``mean(y_train)`` (mixed uses the plain,
            unweighted mean).
    """
    from ..core.gram import build_gram_matrix

    Phi = _as_f64(Phi)
    reg_diag = _as_f64(reg_diag)
    y64 = _as_f64(y_train)
    if f0 is None:
        f0 = float(np.mean(np.asarray(y_train)))
    else:
        f0 = float(f0)
    y_centered = y64 - f0

    groups = []
    gram1 = {}   # cache per-variable Gram for reuse in pair Grams
    F1 = 0
    for i, (bn, K, il, B, offset) in enumerate(block_info):
        Gi = np.asarray(build_gram_matrix(K, il, bn), dtype=np.float64)
        gram1[i] = Gi
        groups.append((1, i, slice(offset, offset + B), Gi))
        F1 = max(F1, offset + B)

    if pair_block_info is not None:
        for (i, j, Bi, Bj, block_size, offset) in pair_block_info:
            Gi = gram1[i]
            Gj = gram1[j]
            # Order-2 Gram for the (basis_i ⊗ basis_j) product columns, laid out
            # row-major over (Bi, Bj) exactly as build_mixed_second_order_features
            # reshapes them — this is the Kronecker product G_i ⊗ G_j.
            Gij = np.kron(Gi, Gj)
            col = slice(F1 + offset, F1 + offset + block_size)
            groups.append((2, (int(i), int(j)), col, Gij))

    return FittedDesign(
        Phi=Phi,
        w=_as_f64(w),
        reg_diag=reg_diag,
        y_centered=y_centered,
        f0=f0,
        D=D,
        sample_weights=None,
        blocks=[],
        sobol_groups=groups,
    )
