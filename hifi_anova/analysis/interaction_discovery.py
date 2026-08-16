"""Projection-based residual sieve: flag possible missing structure.

The residual sieve projects the model residual onto feature subspaces at each
interaction level to measure how much variance each subspace would capture —
without fitting it. This is exact for linear models.

Available scans:
  - scan_missing_pairs:          per-pair score for mean residual
  - scan_missing_variance_pairs: per-pair fitted log-residual-scale score
  - scan_missing_triples:        per-triple score for mean residual
  - scan_residual_subspace:      score for RBF/RFF residual subspace
  - unified_residual_sieve:      decompose residual across ALL levels at once
  - auto_decide_stages:          use sieve scores to decide which stages to add

Each per-group scan is a ranking heuristic, not a calibrated hypothesis test.
These scans — like the package's BIC, group-lasso, 1-SE, and pruning choices —
are model-selection heuristics, **not** the manuscript's FDR-controlled
Theorem-2 procedures. Threshold crossings are therefore called "flagged" or
"above threshold", never statistically significant.

Usage:
    from hifi_anova.analysis.interaction_discovery import unified_residual_sieve

    # After fitting order 1 (or 1+2):
    sieve = unified_residual_sieve(model, x_data, y_data)
    print(sieve)
    # Shows: how much residual is second-order? third-order? smooth (RBF)? noise?
"""

import warnings

import numpy as np
from ..array_backend import xp as jnp  # switchable array backend (numpy exact core)
from typing import Dict, List, Optional, Tuple
from itertools import combinations
from dataclasses import dataclass, field


_ALIAS_UNSET = object()


def _resolve_flag_threshold(flag_threshold, significance_threshold, default):
    """Resolve the one-release deprecated threshold keyword alias."""
    if significance_threshold is not _ALIAS_UNSET:
        warnings.warn(
            "significance_threshold is deprecated; use flag_threshold. "
            "Residual-sieve threshold crossings are heuristic flags, not "
            "significance tests.",
            DeprecationWarning, stacklevel=3)
        if flag_threshold is not _ALIAS_UNSET:
            raise TypeError(
                "Pass only flag_threshold; significance_threshold is its "
                "deprecated alias.")
        return float(significance_threshold)
    return default if flag_threshold is _ALIAS_UNSET else float(flag_threshold)


class _DiscoveryResult(dict):
    """Canonical discovery mapping with warning-emitting legacy key aliases."""
    _ALIASES = {
        'n_significant': 'n_flagged',
        'significance_threshold': 'flag_threshold',
    }

    def _canonical_key(self, key):
        canonical = self._ALIASES.get(key)
        if canonical is not None:
            warnings.warn(
                f"{key} is deprecated; use {canonical}. Residual-sieve "
                "threshold crossings are heuristic flags, not significance "
                "tests.",
                DeprecationWarning, stacklevel=3)
            return canonical
        return key

    def __getitem__(self, key):
        return super().__getitem__(self._canonical_key(key))

    def get(self, key, default=None):
        return super().get(self._canonical_key(key), default)

    def __contains__(self, key):
        return super().__contains__(self._canonical_key(key))


@dataclass(init=False)
class MissingPairResult:
    """Exploratory ranking from scanning unselected pairs (not an FDR test)."""
    # Per-pair variance captured from residual
    pair_scores: Dict[tuple, float]
    # Ranked list: [(pair, score), ...] highest first
    ranked_pairs: List[Tuple[tuple, float]]
    # Summary
    n_scanned: int
    n_flagged: int
    total_residual_variance: float
    total_captured_by_missing: float
    flag_threshold: float = 0.001
    # Per-pair details
    pair_details: Dict[tuple, dict] = field(default_factory=dict)

    def __init__(
        self,
        pair_scores,
        ranked_pairs,
        n_scanned,
        n_flagged=_ALIAS_UNSET,
        total_residual_variance=_ALIAS_UNSET,
        total_captured_by_missing=_ALIAS_UNSET,
        pair_details=_ALIAS_UNSET,
        *,
        flag_threshold=0.001,
        n_significant=_ALIAS_UNSET,
    ):
        """Build a result while preserving the pre-DEC-050 constructor.

        The first seven positional slots retain their historical order; the
        canonical ``flag_threshold`` is keyword-only. ``n_significant=`` is a
        one-release warning alias for ``n_flagged=``.
        """
        if n_significant is not _ALIAS_UNSET:
            warnings.warn(
                "n_significant is deprecated; use n_flagged. Residual-sieve "
                "threshold crossings are heuristic flags, not significance "
                "tests.",
                DeprecationWarning, stacklevel=2)
            if n_flagged is not _ALIAS_UNSET:
                raise TypeError(
                    "Pass only n_flagged; n_significant is its deprecated alias.")
            n_flagged = n_significant
        missing = [
            name for name, value in (
                ('n_flagged', n_flagged),
                ('total_residual_variance', total_residual_variance),
                ('total_captured_by_missing', total_captured_by_missing),
            ) if value is _ALIAS_UNSET]
        if missing:
            raise TypeError("Missing required argument(s): " + ", ".join(missing))

        self.pair_scores = pair_scores
        self.ranked_pairs = ranked_pairs
        self.n_scanned = n_scanned
        self.n_flagged = n_flagged
        self.total_residual_variance = total_residual_variance
        self.total_captured_by_missing = total_captured_by_missing
        self.flag_threshold = flag_threshold
        self.pair_details = ({} if pair_details is _ALIAS_UNSET
                             else pair_details)

    def __setstate__(self, state):
        """Migrate pre-DEC-050 pickle state to canonical field names."""
        state = dict(state)
        legacy_count = state.pop('n_significant', _ALIAS_UNSET)
        if 'n_flagged' not in state and legacy_count is not _ALIAS_UNSET:
            warnings.warn(
                "Loaded legacy MissingPairResult.n_significant; migrated it "
                "to n_flagged.",
                DeprecationWarning, stacklevel=2)
            state['n_flagged'] = legacy_count
        state.setdefault('flag_threshold', 0.001)
        state.setdefault('pair_details', {})
        self.__dict__.update(state)

    @property
    def n_significant(self):
        """Deprecated alias for :attr:`n_flagged` (one-release bridge)."""
        warnings.warn(
            "n_significant is deprecated; use n_flagged. Residual-sieve "
            "threshold crossings are heuristic flags, not significance tests.",
            DeprecationWarning, stacklevel=2)
        return self.n_flagged

    @property
    def significance_threshold(self):
        """Deprecated alias for :attr:`flag_threshold` (one-release bridge)."""
        warnings.warn(
            "significance_threshold is deprecated; use flag_threshold.",
            DeprecationWarning, stacklevel=2)
        return self.flag_threshold


def scan_missing_pairs(
    model,
    x_data: jnp.ndarray,
    y_data: jnp.ndarray,
    selected_pairs: Optional[List[tuple]] = None,
    K2: Optional[int] = None,
    flag_threshold=_ALIAS_UNSET,
    verbose: bool = True,
    *,
    significance_threshold=_ALIAS_UNSET,
) -> MissingPairResult:
    """Scan all unselected pairs for residual variance capture.

    For each unselected pair (i,j):
      1. Build second-order features Phi_pair (N, block^2)
      2. Project residual onto Phi_pair: r_proj = Phi (Phi^T Phi)^{-1} Phi^T r
      3. Score = Var(r_proj) / Var(r)

    This exploratory sieve ranks how much residual variance each pair's
    subspace captures without fitting the full model. It is a model-selection
    heuristic, not a calibrated p-value or the manuscript's FDR-controlled
    Theorem-2 procedure.

    Args:
        model: fitted HiFiANOVA (with some pairs selected)
        x_data: (N, D) input data
        y_data: (N,) targets
        selected_pairs: list of already-fitted pairs [(i,j), ...].
            If None, reads from model.pair_indices.
        K2: harmonic order for pair features. If None, uses model.K2.
        flag_threshold: pairs capturing more than this fraction of residual
            variance are flagged for inspection.
        significance_threshold: deprecated alias for ``flag_threshold``.
        verbose: print results

    Returns:
        MissingPairResult with per-pair scores and rankings
    """
    from ..core.features import build_second_order_features

    flag_threshold = _resolve_flag_threshold(
        flag_threshold, significance_threshold, 0.001)

    x_data = jnp.asarray(x_data)
    y_data = jnp.asarray(y_data)
    D = model.D
    K2_use = K2 if K2 is not None else model.K2
    if K2_use == 0:
        K2_use = 3  # default for discovery

    # Get selected pairs from model
    if selected_pairs is None:
        if model.pair_indices is not None:
            selected_pairs = [(int(model.pair_indices[p, 0]),
                               int(model.pair_indices[p, 1]))
                              for p in range(model.pair_indices.shape[0])]
        else:
            selected_pairs = []
    selected_set = set(selected_pairs)

    # Compute residuals from current model
    mean_pred, _ = model.predict(x_data)
    residuals = np.asarray(y_data - mean_pred, dtype=np.float64)
    residual_var = float(np.var(residuals))

    if residual_var < 1e-15:
        if verbose:
            print("  Residual variance is zero — nothing to discover.")
        return MissingPairResult(
            pair_scores={}, ranked_pairs=[], n_scanned=0,
            n_flagged=0, flag_threshold=flag_threshold,
            total_residual_variance=0.0,
            total_captured_by_missing=0.0)

    # All possible pairs
    all_pairs = list(combinations(range(D), 2))
    missing_pairs = [p for p in all_pairs if p not in selected_set]

    if not missing_pairs:
        if verbose:
            print("  All pairs already selected — nothing to scan.")
        return MissingPairResult(
            pair_scores={}, ranked_pairs=[], n_scanned=0,
            n_flagged=0, flag_threshold=flag_threshold,
            total_residual_variance=residual_var,
            total_captured_by_missing=0.0)

    pair_scores = {}
    pair_details = {}

    for (i, j) in missing_pairs:
        # Build features for just this pair
        pair_idx = jnp.array([[i, j]], dtype=jnp.int32)
        Phi_pair = np.asarray(
            build_second_order_features(x_data, K2_use, pair_idx),
            dtype=np.float64)  # (N, block^2)

        frac_captured, coeffs = _project_residual_score(
            Phi_pair, residuals, residual_var)

        pair_scores[(i, j)] = frac_captured
        pair_details[(i, j)] = {
            'variance_captured': frac_captured * residual_var,
            'fraction_of_residual': frac_captured,
            'coefficients': coeffs,
        }

    # Rank by captured variance
    ranked = sorted(pair_scores.items(), key=lambda x: -x[1])
    n_flagged = sum(1 for _, s in ranked if s > flag_threshold)
    total_captured = sum(pair_scores.values())

    if verbose:
        print(f"  Scanned {len(missing_pairs)} unselected pairs "
              f"(residual var = {residual_var:.4f})")
        print("  Exploratory selection heuristic (not an FDR-controlled test).")
        print(f"  {n_flagged} flagged (>{flag_threshold:.1%} of residual):")
        for (i, j), score in ranked[:min(10, len(ranked))]:
            flag = " *** above threshold" if score > flag_threshold else ""
            print(f"    ({i},{j}): {score:.4f} "
                  f"({score * residual_var:.4f} variance){flag}")

    return MissingPairResult(
        pair_scores=pair_scores,
        ranked_pairs=ranked,
        n_scanned=len(missing_pairs),
        n_flagged=n_flagged,
        flag_threshold=flag_threshold,
        total_residual_variance=residual_var,
        total_captured_by_missing=total_captured,
        pair_details=pair_details,
    )


def scan_missing_variance_pairs(
    model,
    x_data: jnp.ndarray,
    y_data: jnp.ndarray,
    Kh: Optional[int] = None,
    flag_threshold=_ALIAS_UNSET,
    verbose: bool = True,
    *,
    significance_threshold=_ALIAS_UNSET,
) -> MissingPairResult:
    """Scan for missing variance (noise) interactions.

    Same as scan_missing_pairs but applied to the VARIANCE residual:
    does the noise structure depend on pair (i,j)?

    Projects ``log(r^2) - h_fitted`` onto pair features as a noisy
    log-residual-moment ranking proxy. Under an ideal known mean it contains
    the additive ``log chi^2_1`` term (and hence a constant expectation shift);
    fitted residuals add mean-estimation bias. It is not an unbiased estimator
    of log-variance, a calibrated test, or the manuscript's FDR procedure.

    Args:
        model: fitted HiFiANOVA with variance_model
        x_data: (N, D) inputs
        y_data: (N,) targets
        Kh: harmonic order for variance pairs (default: model.Kh or 2)
        flag_threshold: heuristic fraction threshold for flagging
        significance_threshold: deprecated alias for ``flag_threshold``
        verbose: print results

    Returns:
        MissingPairResult for variance interactions
    """
    from ..core.features import build_second_order_features

    flag_threshold = _resolve_flag_threshold(
        flag_threshold, significance_threshold, 0.001)

    x_data = jnp.asarray(x_data)
    y_data = jnp.asarray(y_data)
    D = model.D
    Kh_use = Kh if Kh is not None else getattr(model, 'Kh', 2)
    if Kh_use == 0:
        Kh_use = 2

    # Compute mean prediction and squared residuals
    mean_pred, var_pred = model.predict(x_data)
    residuals = np.asarray(y_data - mean_pred, dtype=np.float64)
    r2 = residuals ** 2

    # Compute variance residual: what the current variance model doesn't explain
    if model.variance_model is not None:
        log_var_pred = np.asarray(jnp.log(var_pred), dtype=np.float64)
        # Variance residual = log(r^2) - log(sigma^2_predicted)
        # Noisy log-residual-moment proxy: includes log(chi^2_1), its constant
        # expectation shift, and contamination from estimating the mean.
        log_r2 = np.log(np.maximum(r2, 1e-20))
        var_residual = log_r2 - log_var_pred
    else:
        # No variance model — use log(r^2) - log(mean(r^2))
        log_r2 = np.log(np.maximum(r2, 1e-20))
        var_residual = log_r2 - np.mean(log_r2)

    var_residual_var = float(np.var(var_residual))

    # Get current variance pairs
    var_pairs_set = set()
    if hasattr(model, 'variance_model') and model.variance_model is not None:
        vm = model.variance_model
        if hasattr(vm, 'pair_indices_h') and vm.pair_indices_h is not None:
            for p in range(vm.pair_indices_h.shape[0]):
                var_pairs_set.add((int(vm.pair_indices_h[p, 0]),
                                   int(vm.pair_indices_h[p, 1])))

    all_pairs = list(combinations(range(D), 2))
    missing_pairs = [p for p in all_pairs if p not in var_pairs_set]

    pair_scores = {}
    pair_details = {}

    for (i, j) in missing_pairs:
        pair_idx = jnp.array([[i, j]], dtype=jnp.int32)
        Psi_pair = np.asarray(
            build_second_order_features(x_data, Kh_use, pair_idx),
            dtype=np.float64)

        frac, _ = _project_residual_score(Psi_pair, var_residual, var_residual_var)

        pair_scores[(i, j)] = frac
        pair_details[(i, j)] = {
            'variance_captured': frac * var_residual_var,
            'fraction_of_var_residual': frac,
        }

    ranked = sorted(pair_scores.items(), key=lambda x: -x[1])
    n_flagged = sum(1 for _, s in ranked if s > flag_threshold)

    if verbose:
        print(f"  Variance pair scan: {len(missing_pairs)} pairs "
              f"(var residual var = {var_residual_var:.4f})")
        print("  Exploratory selection heuristic (not an FDR-controlled test).")
        if n_flagged > 0:
            print(f"  {n_flagged} variance interactions flagged:")
        for (i, j), score in ranked[:min(8, len(ranked))]:
            if score > flag_threshold:
                print(f"    ({i},{j}): {score:.4f} *** above threshold")

    return MissingPairResult(
        pair_scores=pair_scores,
        ranked_pairs=ranked,
        n_scanned=len(missing_pairs),
        n_flagged=n_flagged,
        flag_threshold=flag_threshold,
        total_residual_variance=var_residual_var,
        total_captured_by_missing=sum(pair_scores.values()),
        pair_details=pair_details,
    )


def iterative_pair_discovery(
    model,
    x_train: jnp.ndarray,
    y_train: jnp.ndarray,
    x_val: jnp.ndarray,
    y_val: jnp.ndarray,
    max_rounds: int = 5,
    pairs_per_round: int = 3,
    flag_threshold=_ALIAS_UNSET,
    K2: Optional[int] = None,
    verbose: bool = True,
    *,
    significance_threshold=_ALIAS_UNSET,
) -> Dict:
    """Iterative pair discovery: fit → scan → add → refit.

    A cheap alternative to fitting all C(D,2) pairs: start with selected
    pairs, scan for missed ones, add the highest-ranked flagged ones, and refit.
    Continues until no pairs are above threshold or max_rounds is reached. This
    is exploratory model selection, not an FDR-controlled test.

    Args:
        model: fitted HiFiANOVA (with some pairs)
        x_train, y_train: training data
        x_val, y_val: validation data
        max_rounds: maximum discovery rounds
        pairs_per_round: how many pairs to add per round
        flag_threshold: minimum fraction to flag for addition
        significance_threshold: deprecated alias for ``flag_threshold``
        K2: harmonic order for discovery
        verbose: print progress

    Returns:
        Dict with:
          discovered_pairs: list of newly discovered pairs
          rounds: per-round diagnostics
          final_model: model with discovered pairs added
          improvement: RMSE improvement from discovered pairs
    """
    from ..training.ridge import weighted_ridge_solve
    from ..training.regularization import build_regularization_vector
    from ..core.features import build_second_order_features, basis_size
    from ..core.pairs import PairManager

    flag_threshold = _resolve_flag_threshold(
        flag_threshold, significance_threshold, 0.005)

    K2_use = K2 if K2 is not None else model.K2
    D = model.D
    K1 = model.K1

    # Current pairs
    current_pairs = []
    if model.pair_indices is not None:
        for p in range(model.pair_indices.shape[0]):
            current_pairs.append((int(model.pair_indices[p, 0]),
                                  int(model.pair_indices[p, 1])))

    initial_rmse = float(jnp.sqrt(jnp.mean((y_val - model.predict_mean_only(x_val)) ** 2)))
    rounds = []
    discovered = []

    current_model = model

    for rnd in range(max_rounds):
        if verbose:
            print(f"\n--- Discovery round {rnd + 1}/{max_rounds} ---")

        # Scan missing pairs
        scan = scan_missing_pairs(
            current_model, x_train, y_train,
            selected_pairs=current_pairs, K2=K2_use,
            flag_threshold=flag_threshold,
            verbose=verbose,
        )

        if scan.n_flagged == 0:
            if verbose:
                print("  No missing pairs above threshold — stopping.")
            break

        # Pick top pairs to add
        new_pairs = [pair for pair, score in scan.ranked_pairs[:pairs_per_round]
                     if score > flag_threshold]
        if not new_pairs:
            break

        current_pairs.extend(new_pairs)
        discovered.extend(new_pairs)

        if verbose:
            print(f"  Adding {len(new_pairs)} pairs: {new_pairs}")

        # Refit with extended pairs
        pair_mgr = PairManager(D)
        pair_mgr.pair_indices = jnp.array(
            np.array(current_pairs, dtype=np.int32))
        pair_mgr.P = len(current_pairs)

        phi1_train = model.build_phi1(x_train)
        phi2_train = build_second_order_features(x_train, K2_use, pair_mgr.pair_indices)
        Phi_all = jnp.concatenate([phi1_train, phi2_train], axis=1)

        f0 = float(jnp.mean(y_train))
        y_c = y_train - f0

        reg = build_regularization_vector(D, K1, K2_use, pair_mgr.P, 'curvature',
                                           0.001, 0.01)
        w = weighted_ridge_solve(Phi_all, y_c, reg)

        # Build updated model
        from ..model.mean_model import MeanModel
        from ..model.hifi_anova import HiFiANOVA

        _bn = getattr(model, 'basis_name', 'fourier')
        _il1 = getattr(model, 'include_linear_1', True)
        F1 = D * basis_size(K1, _il1, _bn)
        w1 = w[:F1]
        w2 = w[F1:]

        mean_model = MeanModel(
            f0=jnp.array(f0, dtype=jnp.float32),
            w1=jnp.array(w1, dtype=jnp.float32),
            w2=jnp.array(w2, dtype=jnp.float32),
            K1=K1, K2=K2_use, D=D,
        )
        current_model = HiFiANOVA(
            mean_model=mean_model,
            residual_net=model.residual_net,
            K1=K1, K2=K2_use, K3=model.K3, Kh=model.Kh, D=D,
            pair_indices=np.array(pair_mgr.pair_indices),
            triple_indices=model.triple_indices,
        )

        rmse_val = float(jnp.sqrt(jnp.mean(
            (y_val - current_model.predict_mean_only(x_val)) ** 2)))

        rounds.append({
            'round': rnd + 1,
            'pairs_added': new_pairs,
            'total_pairs': len(current_pairs),
            'rmse_val': rmse_val,
            'scan': scan,
        })

        if verbose:
            print(f"  RMSE after refit: {rmse_val:.4f} "
                  f"(total {len(current_pairs)} pairs)")

    final_rmse = float(jnp.sqrt(jnp.mean(
        (y_val - current_model.predict_mean_only(x_val)) ** 2)))

    return {
        'discovered_pairs': discovered,
        'rounds': rounds,
        'final_model': current_model,
        'initial_rmse': initial_rmse,
        'final_rmse': final_rmse,
        'improvement': initial_rmse - final_rmse,
        'n_rounds': len(rounds),
        'total_pairs': len(current_pairs),
    }


# =============================================================================
# Core projection helper (used by all sieve functions)
# =============================================================================

def _project_residual_score(Phi_sub: np.ndarray, residuals: np.ndarray,
                             residual_var: float,
                             adjust_for_df: bool = True) -> Tuple[float, np.ndarray]:
    """Project residual onto a feature subspace and return variance fraction captured.

    When adjust_for_df=True (default), subtracts the expected fraction that
    pure noise would capture due to overfitting: E[fraction | noise] ≈ M/N.
    This prevents false positives when projecting noise onto many features.

    Args:
        Phi_sub: (N, M) feature matrix for the subspace
        residuals: (N,) residual vector
        residual_var: total variance of residual (for normalization)
        adjust_for_df: subtract expected null fraction M/N

    Returns:
        (fraction_captured, coefficients)
    """
    N = Phi_sub.shape[0]
    block = Phi_sub.shape[1]
    if block == 0:
        return 0.0, np.zeros(0)
    PtP = Phi_sub.T @ Phi_sub
    eps = 1e-8 * np.trace(PtP) / max(block, 1)
    A = PtP + eps * np.eye(block)
    coeffs = np.linalg.solve(A, Phi_sub.T @ residuals)
    r_proj = Phi_sub @ coeffs
    var_captured = float(np.var(r_proj))
    frac = var_captured / max(residual_var, 1e-15)

    if adjust_for_df:
        # Under the null (residual is pure noise), a projection onto M features
        # captures approximately M/N of the noise variance (overfitting).
        null_frac = block / N
        frac = max(0.0, frac - null_frac)

    return frac, coeffs


# =============================================================================
# Scan missing triples (third-order interactions)
# =============================================================================

def scan_missing_triples(
    model,
    x_data: jnp.ndarray,
    y_data: jnp.ndarray,
    selected_triples: Optional[List[tuple]] = None,
    K3: Optional[int] = None,
    flag_threshold=_ALIAS_UNSET,
    max_triples: int = 500,
    verbose: bool = True,
    *,
    significance_threshold=_ALIAS_UNSET,
) -> Dict:
    """Scan unselected triples for residual variance capture.

    Same exploratory ranking principle as scan_missing_pairs but for three-way
    interactions. This is not a calibrated p-value or FDR-controlled test.
    Each test is a (2K+1)^3 × (2K+1)^3 solve — still tiny for K3=1.

    With D=20, C(20,3)=1140 triples. At K3=1 that's 1140 × 27×27 solves — seconds.
    Set max_triples to limit scanning for very large D.

    Args:
        model: fitted HiFiANOVA
        x_data: (N, D) inputs
        y_data: (N,) targets
        selected_triples: already-fitted triples. If None, reads from model.
        K3: harmonic order (default: model.K3 or 1)
        flag_threshold: heuristic threshold for flagging
        significance_threshold: deprecated alias for ``flag_threshold``
        max_triples: cap on number of triples to scan
        verbose: print results

    Returns:
        Dict with triple_scores, ranked_triples, n_flagged, flag_threshold, etc.
    """
    from ..core.features import build_third_order_features

    flag_threshold = _resolve_flag_threshold(
        flag_threshold, significance_threshold, 0.001)

    x_data = jnp.asarray(x_data)
    y_data = jnp.asarray(y_data)
    D = model.D
    K3_use = K3 if K3 is not None else getattr(model, 'K3', 0)
    if K3_use == 0:
        K3_use = 1

    # Get currently selected triples
    if selected_triples is None:
        if getattr(model, 'triple_indices', None) is not None:
            ti = model.triple_indices
            selected_triples = [(int(ti[t, 0]), int(ti[t, 1]), int(ti[t, 2]))
                                for t in range(ti.shape[0])]
        else:
            selected_triples = []
    selected_set = set(selected_triples)

    # Compute residuals
    mean_pred, _ = model.predict(x_data)
    residuals = np.asarray(y_data - mean_pred, dtype=np.float64)
    residual_var = float(np.var(residuals))

    if residual_var < 1e-15:
        if verbose:
            print("  Residual variance is zero — nothing to discover.")
        return _DiscoveryResult(
            triple_scores={}, ranked_triples=[], n_scanned=0,
            n_flagged=0, flag_threshold=flag_threshold,
            total_residual_variance=0.0)

    # All possible triples (capped at max_triples)
    all_triples = list(combinations(range(D), 3))
    missing = [t for t in all_triples if t not in selected_set]
    if len(missing) > max_triples:
        missing = missing[:max_triples]
        if verbose:
            print(f"  Capped at {max_triples} triples (of {len(all_triples) - len(selected_set)} missing)")

    # Get basis config from model
    _bn = getattr(model, 'basis_name', 'fourier')
    _il3 = getattr(model, 'include_linear_3', True)

    triple_scores = {}
    for (i, j, k) in missing:
        triple_idx = jnp.array([[i, j, k]], dtype=jnp.int32)
        Phi_triple = np.asarray(
            build_third_order_features(x_data, K3_use, triple_idx,
                                        include_linear=_il3, basis_name=_bn),
            dtype=np.float64)
        frac, _ = _project_residual_score(Phi_triple, residuals, residual_var)
        triple_scores[(i, j, k)] = frac

    ranked = sorted(triple_scores.items(), key=lambda x: -x[1])
    n_flagged = sum(1 for _, s in ranked if s > flag_threshold)

    if verbose:
        print(f"  Triple scan: {len(missing)} triples "
              f"(residual var = {residual_var:.4f})")
        print("  Exploratory selection heuristic (not an FDR-controlled test).")
        print(f"  {n_flagged} flagged (>{flag_threshold:.1%}):")
        for triple, score in ranked[:min(8, len(ranked))]:
            if score > flag_threshold:
                print(f"    {triple}: {score:.4f} *** above threshold")

    return _DiscoveryResult({
        'triple_scores': triple_scores,
        'ranked_triples': ranked,
        'n_scanned': len(missing),
        'n_flagged': n_flagged,
        'flag_threshold': flag_threshold,
        'total_residual_variance': residual_var,
    })


# =============================================================================
# Scan RBF/RFF residual subspace
# =============================================================================

def scan_residual_subspace(
    model,
    x_data: jnp.ndarray,
    y_data: jnp.ndarray,
    residual_type: str = 'rbf',
    n_features: int = 200,
    sigma: float = 0.2,
    gamma: float = 3.0,
    verbose: bool = True,
) -> Dict:
    """Measure how much residual variance a smooth RBF/RFF subspace would capture.

    This answers: "if I add an RBF/RFF residual with these hyperparameters,
    how much of the remaining residual would it explain?"

    The answer is a single number (fraction captured), plus per-center diagnostics.

    Args:
        model: fitted HiFiANOVA
        x_data: (N, D) inputs
        y_data: (N,) targets
        residual_type: 'rbf' or 'rff'
        n_features: number of centers (RBF) or random features (RFF)
        sigma: RBF width (only for 'rbf')
        gamma: RFF frequency scale (only for 'rff')
        verbose: print result

    Returns:
        Dict with fraction_captured, n_features, residual_variance
    """
    from ..model.linear_residual import RBFResidual, RFFResidual

    x_data = jnp.asarray(x_data)
    y_data = jnp.asarray(y_data)
    D = model.D

    # Compute residuals
    mean_pred, _ = model.predict(x_data)
    residuals = np.asarray(y_data - mean_pred, dtype=np.float64)
    residual_var = float(np.var(residuals))

    if residual_var < 1e-15:
        if verbose:
            print("  Residual variance is zero.")
        return {'fraction_captured': 0.0, 'residual_variance': 0.0}

    # Build residual features
    if residual_type == 'rbf':
        res = RBFResidual.create(x_data, n_centers=min(n_features, len(x_data)),
                                   sigma=sigma, method='kmeans')
    elif residual_type == 'rff':
        import jax
        res = RFFResidual.create(D, n_features=n_features, gamma=gamma,
                                   key=jax.random.PRNGKey(42))
    else:
        raise ValueError(f"Unknown residual_type: {residual_type}")

    Z = np.asarray(res.build_features(x_data), dtype=np.float64)

    # Project residual onto Z (without orthogonalizing against Fourier first —
    # this gives an upper bound on what the residual subspace can capture)
    frac_raw, _ = _project_residual_score(Z, residuals, residual_var)

    # Also project with Fourier orthogonalization (the actual amount after projection)
    Phi = np.asarray(model.build_phi_all(x_data), dtype=np.float64)
    from ..core.projection import project_features_orthogonal
    Z_proj, _ = project_features_orthogonal(jnp.array(Z), jnp.array(Phi))
    Z_proj_np = np.asarray(Z_proj, dtype=np.float64)
    frac_orthogonal, _ = _project_residual_score(Z_proj_np, residuals, residual_var)

    if verbose:
        print(f"  Residual subspace ({residual_type}, M={Z.shape[1]}): "
              f"captures {frac_orthogonal:.1%} of residual "
              f"(raw {frac_raw:.1%}, orthogonal {frac_orthogonal:.1%})")

    return {
        'fraction_captured_raw': frac_raw,
        'fraction_captured_orthogonal': frac_orthogonal,
        'residual_variance': residual_var,
        'n_features': Z.shape[1],
        'residual_type': residual_type,
    }


# =============================================================================
# Unified residual sieve: decompose residual across ALL levels
# =============================================================================

@dataclass
class SieveResult:
    """Results from the unified residual sieve."""
    total_residual_variance: float
    # Per-level variance fractions (what each level would capture)
    first_order_fraction: float          # unexplained by current first-order
    second_order_fraction: float         # total from all missing pairs
    third_order_fraction: float          # total from all missing triples
    residual_rbf_fraction: float         # from smooth RBF subspace
    noise_fraction: float                # unexplained by any structured subspace
    # Top items per level
    top_pairs: List[Tuple[tuple, float]] = field(default_factory=list)
    top_triples: List[Tuple[tuple, float]] = field(default_factory=list)
    # Decision recommendation
    recommendation: str = ""

    def __repr__(self):
        lines = [
            f"SieveResult (total residual var = {self.total_residual_variance:.4f}):",
            f"  First-order (unfitted vars):  {self.first_order_fraction:6.1%}",
            f"  Second-order (missing pairs): {self.second_order_fraction:6.1%}",
            f"  Third-order (missing triples):{self.third_order_fraction:6.1%}",
            f"  Smooth residual (RBF):        {self.residual_rbf_fraction:6.1%}",
            f"  Noise / unexplained:          {self.noise_fraction:6.1%}",
        ]
        if self.top_pairs:
            lines.append(f"  Top pairs: {self.top_pairs[:3]}")
        if self.top_triples:
            lines.append(f"  Top triples: {self.top_triples[:3]}")
        if self.recommendation:
            lines.append(f"  Recommendation: {self.recommendation}")
        return "\n".join(lines)


def unified_residual_sieve(
    model,
    x_data: jnp.ndarray,
    y_data: jnp.ndarray,
    scan_pairs: bool = True,
    scan_triples: bool = True,
    scan_rbf: bool = True,
    K2: Optional[int] = None,
    K3: int = 1,
    rbf_n_centers: int = 200,
    rbf_sigma: float = 0.2,
    max_pairs: Optional[int] = None,
    max_triples: int = 500,
    verbose: bool = True,
) -> SieveResult:
    """Rank residual projections across interaction levels simultaneously.

    After fitting a model (any stage), this function answers:
      "Of the remaining residual, how much is..."
        - first-order (variables not fully captured)
        - second-order (missing pair interactions)
        - third-order (missing triple interactions)
        - smooth higher-order (capturable by RBF)
        - noise (not capturable by any structured subspace)

    Each level is assessed by projecting the residual onto that level's
    feature subspace. The fractions may sum to MORE than 1 (subspaces
    overlap at finite N), but the relative ranking is meaningful. This is an
    exploratory model-selection heuristic, not the manuscript's FDR-controlled
    Theorem-2 procedure.

    Args:
        model: fitted HiFiANOVA at any stage
        x_data: (N, D) inputs
        y_data: (N,) targets
        scan_pairs: whether to scan missing pairs
        scan_triples: whether to scan missing triples (skip for large D)
        scan_rbf: whether to scan RBF residual subspace
        K2: pair harmonic order (default: model.K2 or 3)
        K3: triple harmonic order (default: 1)
        rbf_n_centers: RBF centers for smooth residual scan
        rbf_sigma: RBF width
        max_pairs: cap on pairs to scan (None = all)
        max_triples: cap on triples to scan
        verbose: print decomposition

    Returns:
        SieveResult with per-level fractions and recommendation
    """

    x_data = jnp.asarray(x_data)
    y_data = jnp.asarray(y_data)
    D = model.D

    # Compute residuals
    mean_pred, _ = model.predict(x_data)
    residuals = np.asarray(y_data - mean_pred, dtype=np.float64)
    residual_var = float(np.var(residuals))

    if verbose:
        total_var_y = float(np.var(np.asarray(y_data)))
        # Explained-variance convention (the library/manuscript default; see
        # hifi_anova.analysis.metrics).
        r2 = 1.0 - residual_var / total_var_y if total_var_y > 0 else 0.0
        print(f"Residual sieve (R²[expl-var] = {r2:.4f}, "
              f"residual var = {residual_var:.4f}):")

    if residual_var < 1e-15:
        return SieveResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    # --- First-order: project residual onto first-order features ---
    # This measures how much first-order signal the current model is MISSING
    phi1 = np.asarray(model.build_phi1(x_data), dtype=np.float64)
    frac_1st, _ = _project_residual_score(phi1, residuals, residual_var)
    if verbose:
        print(f"  First-order (re-projection): {frac_1st:.1%}")

    # --- Second-order: scan missing pairs ---
    frac_2nd = 0.0
    top_pairs = []
    if scan_pairs:
        scan_2nd = scan_missing_pairs(model, x_data, y_data, K2=K2,
                                       flag_threshold=0.0, verbose=False)
        frac_2nd = scan_2nd.total_captured_by_missing
        top_pairs = scan_2nd.ranked_pairs[:5]
        if verbose:
            n_flagged = sum(1 for _, s in scan_2nd.ranked_pairs if s > 0.005)
            print(f"  Second-order ({scan_2nd.n_scanned} pairs): "
                  f"{frac_2nd:.1%} total, {n_flagged} flagged above threshold")

    # --- Third-order: scan missing triples ---
    frac_3rd = 0.0
    top_triples = []
    if scan_triples and D >= 3:
        scan_3rd = scan_missing_triples(model, x_data, y_data, K3=K3,
                                          flag_threshold=0.0,
                                          max_triples=max_triples, verbose=False)
        frac_3rd = sum(scan_3rd['triple_scores'].values())
        top_triples = scan_3rd['ranked_triples'][:5]
        if verbose:
            n_flagged = sum(1 for _, s in scan_3rd['ranked_triples'] if s > 0.005)
            print(f"  Third-order ({scan_3rd['n_scanned']} triples): "
                  f"{frac_3rd:.1%} total, {n_flagged} flagged above threshold")

    # --- Smooth residual (RBF): how much can RBF capture? ---
    frac_rbf = 0.0
    if scan_rbf:
        rbf_result = scan_residual_subspace(
            model, x_data, y_data, residual_type='rbf',
            n_features=rbf_n_centers, sigma=rbf_sigma, verbose=False)
        frac_rbf = rbf_result['fraction_captured_orthogonal']
        if verbose:
            print(f"  Smooth residual (RBF, M={rbf_n_centers}): {frac_rbf:.1%}")

    # --- Noise: whatever no structured subspace captures ---
    # Use the RBF fraction as the best "total structured" estimate since
    # RBF captures ALL smooth structure (pairs + triples + higher).
    # Whatever RBF can't capture after df-adjustment is likely noise.
    best_total = max(frac_rbf, frac_2nd, frac_3rd, frac_1st)
    frac_noise = max(0.0, 1.0 - best_total)
    if verbose:
        print(f"  Noise / unexplained: {frac_noise:.1%}")

    # --- Recommendation ---
    parts = []
    if frac_1st > 0.05:
        parts.append("increase K1 or check variable coverage")
    if frac_2nd > 0.05:
        top_p = top_pairs[0] if top_pairs else None
        parts.append(f"add second-order pairs (top: {top_p})" if top_p
                     else "add second-order pairs")
    if frac_3rd > 0.05:
        top_t = top_triples[0] if top_triples else None
        parts.append(f"add third-order (top: {top_t})" if top_t
                     else "add third-order triples")
    if frac_rbf > 0.05 and frac_rbf > frac_2nd and frac_rbf > frac_3rd:
        parts.append("add RBF/RFF residual for smooth higher-order effects")
    if not parts:
        parts.append("model appears adequate — residual is mostly noise")
    recommendation = "; ".join(parts)

    if verbose:
        print(f"  >> {recommendation}")

    return SieveResult(
        total_residual_variance=residual_var,
        first_order_fraction=frac_1st,
        second_order_fraction=frac_2nd,
        third_order_fraction=frac_3rd,
        residual_rbf_fraction=frac_rbf,
        noise_fraction=frac_noise,
        top_pairs=top_pairs,
        top_triples=top_triples,
        recommendation=recommendation,
    )


# =============================================================================
# Auto-decide stages: replace auto_threshold with sieve-based decisions
# =============================================================================

def auto_decide_stages(
    model,
    x_data: jnp.ndarray,
    y_data: jnp.ndarray,
    flag_threshold=_ALIAS_UNSET,
    max_order: int = 3,
    allow_residual: bool = True,
    K3: int = 1,
    verbose: bool = True,
    *,
    significance_threshold=_ALIAS_UNSET,
) -> Dict:
    """Use the heuristic residual sieve to decide which stages/orders to add.

    This data-driven ranking is model selection, not a calibrated test or the
    manuscript's FDR-controlled Theorem-2 procedure.

    Logic:
      1. Run unified sieve on current model
      2. If first-order fraction > threshold → increase K1 (or model is incomplete)
      3. If second-order fraction > threshold → add Stage B (pairs)
      4. If third-order fraction > threshold → add K3 (triples)
      5. If RBF fraction > threshold and > pair/triple → add Stage C (residual)
      6. Otherwise → stop (residual is noise)

    Args:
        model: fitted HiFiANOVA at current stage
        x_data: (N, D) training data
        y_data: (N,) targets
        flag_threshold: minimum fraction to trigger adding a stage
        significance_threshold: deprecated alias for ``flag_threshold``
        max_order: highest interaction order to consider (2 or 3)
        allow_residual: whether to recommend RBF residual
        K3: harmonic for triple scanning
        verbose: print decisions

    Returns:
        Dict with:
          sieve: SieveResult
          stages_to_add: list of stage recommendations
          recommended_config: dict of config keys to set
    """
    flag_threshold = _resolve_flag_threshold(
        flag_threshold, significance_threshold, 0.05)

    sieve = unified_residual_sieve(
        model, x_data, y_data,
        scan_triples=(max_order >= 3),
        scan_rbf=allow_residual,
        K3=K3,
        verbose=verbose,
    )

    stages = []
    config = {}
    thr = flag_threshold

    if sieve.first_order_fraction > thr:
        stages.append('increase_K1')
        config['K1_note'] = 'first-order re-projection captures residual — try higher K1'

    if sieve.second_order_fraction > thr:
        stages.append('B')
        config['K2'] = model.K2 if model.K2 > 0 else 3
        if sieve.top_pairs:
            config['suggested_pairs'] = [p for p, s in sieve.top_pairs if s > thr]

    if max_order >= 3 and sieve.third_order_fraction > thr:
        stages.append('B3')
        config['K3'] = K3
        if sieve.top_triples:
            config['suggested_triples'] = [t for t, s in sieve.top_triples if s > thr]

    if allow_residual and sieve.residual_rbf_fraction > thr:
        # Only recommend RBF if it captures more than structured alternatives
        if (sieve.residual_rbf_fraction > sieve.second_order_fraction and
                sieve.residual_rbf_fraction > sieve.third_order_fraction):
            stages.append('C_residual')
            config['residual'] = {'type': 'rbf', 'n_centers': 300, 'sigma': 0.2}

    if not stages:
        stages.append('stop')

    if verbose:
        print(f"\n  Decision: {stages}")

    return {
        'sieve': sieve,
        'stages_to_add': stages,
        'recommended_config': config,
    }
