"""Selectable SPD inverse for the ridge/analytics solves (opt-in Cholesky).

The analytics form ``A^{-1}`` for ``A = Phi^T W Phi + diag(reg)`` (symmetric
positive-definite when reg > 0). Two backends:

* ``'inv'`` (default) — ``numpy.linalg.inv`` (general LU). This is the historical
  behaviour; the golden master is captured against it, so it stays the default and
  the default path is byte-identical.
* ``'cholesky'`` — ``scipy.linalg`` Cholesky (``potrf`` + ``potri``), which exploits
  symmetry: more stable and ~2x cheaper. On the tested (well-conditioned) paths the
  two agree to ~1e-13 — the only measurable difference is a ~1e-8 shift in one
  near-noiseless overfit scenario's tiny ``sigma_hat`` (DEC-035), which is why
  Cholesky is opt-in rather than the default. It may help on genuinely
  ill-conditioned rich-basis fits, especially combined with ``precision="float64"``.

Selection precedence: explicit ``method=`` arg > ``set_linalg_method`` override >
``HIFI_LINALG`` env var > default (``'inv'``).
"""

import os

import numpy as np

_DEFAULT = "inv"
_VALID = ("inv", "cholesky")

# Process-global override set by set_linalg_method(); None => fall through to env.
_override = None


def resolve_linalg_method(method=None):
    """Resolve the SPD-inverse backend: ``'inv'`` (default) or ``'cholesky'``."""
    if method is not None:
        m = str(method).lower()
    elif _override is not None:
        m = _override
    else:
        m = (os.environ.get("HIFI_LINALG", "").strip().lower() or _DEFAULT)
    if m not in _VALID:
        raise ValueError(
            f"linalg method must be 'inv' or 'cholesky'; got {method!r}")
    return m


def spd_inverse(A: np.ndarray, method=None) -> np.ndarray:
    """Inverse of a symmetric positive-definite matrix.

    ``method='inv'`` (default) returns exactly ``numpy.linalg.inv(A)`` (so the
    default path is byte-identical). ``method='cholesky'`` uses a Cholesky
    factorisation and falls back to ``numpy.linalg.inv`` if ``A`` is not
    numerically PD (e.g. an unregularised rank-deficient design).
    """
    if resolve_linalg_method(method) == "cholesky":
        from scipy.linalg import cho_factor, cho_solve
        from numpy.linalg import LinAlgError
        try:
            c = cho_factor(A, lower=True, check_finite=False)
            return cho_solve(c, np.eye(A.shape[0]), check_finite=False)
        except LinAlgError:
            return np.linalg.inv(A)
    return np.linalg.inv(A)


def set_linalg_method(method):
    """Set a process-global SPD-inverse backend override (``None`` clears it)."""
    global _override
    _override = None if method is None else resolve_linalg_method(method)
    return _override
