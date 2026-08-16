"""Manages the combinatorial structure of variable pairs.

Supports multiple pair selection strategies to control the
combinatorial explosion of second-order terms:

  None / 'all'    — All C(D,2) pairs. Full expressiveness, may be large.
  'both'          — Pairs where BOTH variables have S_i > threshold.
                    Most aggressive reduction. Misses interactions where
                    one variable has no main effect.
  'either'        — Pairs where AT LEAST ONE variable has S_i > threshold.
                    Moderate reduction. Catches interactions involving
                    at least one known-active variable.
  'auto'          — Equivalent to 'both' (default conservative choice).
  list[int]       — Explicit list of active variable indices.

Scaling examples (D=50, K2=5, block=121 features per pair):
  All pairs:      C(50,2) = 1225 pairs → 148,225 features
  'both' (5 act): C(5,2)  =   10 pairs →   1,210 features
  'either'(5 act): 5*45    =  225 pairs →  27,225 features
"""

from ..array_backend import xp as jnp  # switchable array backend (numpy exact core)
import numpy as np
from itertools import combinations
from typing import Optional, List


class PairManager:
    """Manages variable pair enumeration and index mapping.

    Attributes:
        D: number of variables
        pair_indices: jnp.ndarray (P, 2) — selected (i,j) pairs with i<j
        P: number of pairs
        active_variables: list of active variable indices (None = all)
        selection_mode: how pairs were selected
    """

    def __init__(self, D: int, active_variables: Optional[List[int]] = None,
                 selection_mode: str = 'all'):
        """Create a PairManager.

        Args:
            D: total number of variables
            active_variables: if provided, forms pairs based on selection_mode.
                             None = all pairs (backward compatible).
            selection_mode: 'all', 'both', or 'either' (see module docstring)
        """
        self.D = D
        self.active_variables = active_variables
        self.selection_mode = selection_mode

        if active_variables is None or selection_mode == 'all':
            pairs = list(combinations(range(D), 2))
        elif selection_mode == 'both':
            # Only pairs where both variables are active
            active_set = set(active_variables)
            pairs = [(i, j) for i, j in combinations(range(D), 2)
                     if i in active_set and j in active_set]
        elif selection_mode == 'either':
            # Pairs where at least one variable is active
            active_set = set(active_variables)
            pairs = [(i, j) for i, j in combinations(range(D), 2)
                     if i in active_set or j in active_set]
        else:
            raise ValueError(f"Unknown selection_mode: {selection_mode}")

        self.P = len(pairs)
        if pairs:
            self.pair_indices = jnp.array(np.array(pairs, dtype=np.int32))
        else:
            self.pair_indices = jnp.zeros((0, 2), dtype=jnp.int32)

    def variable_slice(self, i: int, K: int, include_linear: bool = True,
                       basis_name: str = 'fourier') -> slice:
        """Slice into w1 for variable i."""
        from .features import basis_size
        block = basis_size(K, include_linear, basis_name)
        return slice(i * block, (i + 1) * block)

    def pair_slice(self, p: int, K: int, include_linear: bool = True, basis_name: str = 'fourier') -> slice:
        """Slice into w2 for pair index p."""
        from .features import basis_size
        block = basis_size(K, include_linear, basis_name) ** 2
        return slice(p * block, (p + 1) * block)

    def pair_to_variables(self, p: int) -> tuple:
        """Return (i, j) variable indices for pair index p."""
        return (int(self.pair_indices[p, 0]), int(self.pair_indices[p, 1]))

    def find_pair_index(self, i: int, j: int) -> int:
        """Find the pair index for variables (i, j). Returns -1 if not found."""
        if i > j:
            i, j = j, i
        # Use cached lookup dict for O(1) access
        if not hasattr(self, '_pair_lookup') or self._pair_lookup is None:
            self._pair_lookup = {}
            for p in range(self.P):
                key = (int(self.pair_indices[p, 0]), int(self.pair_indices[p, 1]))
                self._pair_lookup[key] = p
        return self._pair_lookup.get((i, j), -1)


def pair_manager_from_pairs(D: int, pairs) -> PairManager:
    """Build a :class:`PairManager` holding an EXPLICIT (i, j) pair list.

    Unlike the constructor (which enumerates pairs from an active-variable
    set), this pins the pair set verbatim — used by the user-defined
    term-structure paths (per-pair K2 mapping, variable_orders pair filters)
    where the pairs are specified, not selected. Pairs must already be
    canonical (i < j); order is preserved.
    """
    mgr = PairManager(D, active_variables=[], selection_mode='both')
    mgr.P = len(pairs)
    if mgr.P:
        mgr.pair_indices = jnp.array(np.array(list(pairs), dtype=np.int32))
    else:
        mgr.pair_indices = jnp.zeros((0, 2), dtype=jnp.int32)
    mgr.active_variables = None
    mgr.selection_mode = 'explicit_pairs'
    return mgr


def select_active_variables(first_order_sobol: dict, D: int,
                            threshold: float = 0.01,
                            max_variables: Optional[int] = None) -> List[int]:
    """Select active variables based on first-order Sobol indices.

    Args:
        first_order_sobol: dict {i: S_i} from first-order fit
        D: total number of variables
        threshold: minimum Sobol index to be considered active
        max_variables: hard cap on number of active variables

    Returns:
        Sorted list of active variable indices
    """
    ranked = sorted(first_order_sobol.items(), key=lambda x: -x[1])
    active = [i for i, s in ranked if s > threshold]

    if max_variables is not None and len(active) > max_variables:
        active = [i for i, _ in ranked[:max_variables]]

    if len(active) < 2:
        active = [i for i, _ in ranked[:2]]

    return sorted(active)


class TripleManager:
    """Manages variable triple enumeration and index mapping for third-order terms.

    Parallel to PairManager but for 3-combinations.

    Selection modes:
      None / 'all'     — All C(D,3) triples. Full expressiveness.
      'all_active'     — Triples where ALL THREE variables are active.
                         Most aggressive reduction (recommended default).
      'two_active'     — Triples where AT LEAST TWO variables are active.
      'one_active'     — Triples where AT LEAST ONE variable is active.
      list[int]        — Explicit list of active variable indices,
                         defaults to 'all_active' selection.

    Scaling examples (D=10, K3=1, block=27 features per triple):
      All triples:      C(10,3) = 120 triples →  3,240 features
      'all_active'(5):  C(5,3)  =  10 triples →    270 features
      'two_active'(5):  C(5,2)*5=  50 triples →  1,350 features
    """

    def __init__(self, D: int, active_variables: Optional[List[int]] = None,
                 selection_mode: str = 'all'):
        """Create a TripleManager.

        Args:
            D: total number of variables
            active_variables: if provided, forms triples based on selection_mode
            selection_mode: 'all', 'all_active', 'two_active', or 'one_active'
        """
        self.D = D
        self.active_variables = active_variables
        self.selection_mode = selection_mode

        if active_variables is None or selection_mode == 'all':
            triples = list(combinations(range(D), 3))
        elif selection_mode == 'all_active':
            active_set = set(active_variables)
            triples = [(i, j, k) for i, j, k in combinations(range(D), 3)
                       if i in active_set and j in active_set and k in active_set]
        elif selection_mode == 'two_active':
            active_set = set(active_variables)
            triples = [(i, j, k) for i, j, k in combinations(range(D), 3)
                       if sum(v in active_set for v in (i, j, k)) >= 2]
        elif selection_mode == 'one_active':
            active_set = set(active_variables)
            triples = [(i, j, k) for i, j, k in combinations(range(D), 3)
                       if i in active_set or j in active_set or k in active_set]
        else:
            raise ValueError(f"Unknown selection_mode: {selection_mode}")

        self.T = len(triples)
        if triples:
            self.triple_indices = jnp.array(np.array(triples, dtype=np.int32))
        else:
            self.triple_indices = jnp.zeros((0, 3), dtype=jnp.int32)

    def triple_slice(self, t: int, K: int, include_linear: bool = True, basis_name: str = 'fourier') -> slice:
        """Slice into w3 for triple index t."""
        from .features import basis_size
        block = basis_size(K, include_linear, basis_name) ** 3
        return slice(t * block, (t + 1) * block)

    def triple_to_variables(self, t: int) -> tuple:
        """Return (i, j, k) variable indices for triple index t."""
        return (int(self.triple_indices[t, 0]),
                int(self.triple_indices[t, 1]),
                int(self.triple_indices[t, 2]))

    def find_triple_index(self, i: int, j: int, k: int) -> int:
        """Find the triple index for variables (i, j, k). Returns -1 if not found."""
        indices = tuple(sorted([i, j, k]))
        # Use cached lookup dict for O(1) access
        if not hasattr(self, '_triple_lookup') or self._triple_lookup is None:
            self._triple_lookup = {}
            for t in range(self.T):
                key = (int(self.triple_indices[t, 0]),
                       int(self.triple_indices[t, 1]),
                       int(self.triple_indices[t, 2]))
                self._triple_lookup[key] = t
        return self._triple_lookup.get(indices, -1)
