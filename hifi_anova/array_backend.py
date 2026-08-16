"""Switchable array backend for the fit path (the NumPy "exact core").

Rationale (see ``backend_numpy_exact_core_design.md`` + the parity prototype
``prototypes/numpy_core_prototype.py``): the entire GUI-reachable fit path is
closed-form linear algebra plus a hand-derived convex Newton — no autodiff,
and not a single ``jax.jit`` (the fits run *eager* JAX, paying per-op,
per-shape XLA compilation with none of the fusion benefit). At interactive
problem sizes a float64 NumPy/LAPACK execution of the SAME code is hundreds
of times faster than the compile a structural change triggers, and — because
it IS the same code — statistically identical by construction.

This module provides the indirection that makes the fit path run on either
array library without duplicating any logic:

- ``xp`` — a proxy that resolves ``xp.<name>`` to ``numpy.<name>`` or
  ``jax.numpy.<name>`` depending on the ACTIVE backend at call time. Swept
  modules import it under their historical local name::

      from ..array_backend import xp as jnp   # switchable array backend

  so their bodies are unchanged (and byte-identical on the default backend).
- ``solve_pos(A, b)`` — the positive-definite linear solve
  (``jax.scipy.linalg.solve(..., assume_a='pos')`` / SciPy equivalent).
- ``get_array_backend`` / ``set_array_backend`` / ``use_array_backend`` —
  selection. The DEFAULT is ``'jax'``: with no opt-in, behavior (and the
  golden baselines) is exactly as before this module existed. Selection is a
  thread-local stack over a process-global default, so the one-call API can
  scope a backend to a single fit (``backend=`` argument) without races
  against other threads.

JAX remains an import-time dependency either way — the NumPy backend removes
per-shape XLA work from the hot path, not the JAX install. Paths that are
genuinely JAX-native (the NN residual / NN modes / ``hyperopt_jax``) are NOT
routed through this module; the one-call API refuses ``backend='numpy'`` for
configs that need them. The linear residual families (Stage C rbf/rff/nystrom
+ ``variance_residual``) ARE swept since BR-10/DEC-057.

Semantics notes for swept code:

- Dtypes: the one-call API enables ``jax_enable_x64`` unconditionally, so
  dtype-omitted ``jnp`` array creation defaults to float64 — the same as
  NumPy. Explicit dtypes (``self._dtype`` etc.) behave identically.
- No JAX-only idioms exist in the swept modules (no ``lax``, ``vmap``,
  ``.at[]``, ``jax.random`` — audited 2026-08-14). Since BR-10 the two
  exceptions live in ``model/linear_residual.py``, both guarded off the
  numpy path: backend-native RNG draws (``jax.random`` only on the jax
  backend; a seeded numpy rng otherwise) and the ``jax.vmap`` fallback in
  ``predict_residual_batch`` for JAX-native (NN) residuals. ``numpy``
  output arrays are valid pytree leaves for the equinox model containers.
"""
from __future__ import annotations

import threading
from contextlib import contextmanager

import numpy as _np

VALID_BACKENDS = ("jax", "numpy")

_global_default = "jax"
_tls = threading.local()


def _check(name: str) -> str:
    if name not in VALID_BACKENDS:
        raise ValueError(
            f"unknown array backend {name!r}; valid: {VALID_BACKENDS}")
    return name


def get_array_backend() -> str:
    """The backend active on THIS thread ('jax' or 'numpy')."""
    stack = getattr(_tls, "stack", None)
    return stack[-1] if stack else _global_default


def set_array_backend(name: str) -> None:
    """Set the process-global default backend (thread-local scopes win)."""
    global _global_default
    _global_default = _check(name)


@contextmanager
def use_array_backend(name: str):
    """Scope a backend to the current thread for the duration of the block."""
    _check(name)
    stack = getattr(_tls, "stack", None)
    if stack is None:
        stack = _tls.stack = []
    stack.append(name)
    try:
        yield
    finally:
        stack.pop()


class _ArrayModuleProxy:
    """Attribute proxy resolving to numpy or jax.numpy at ACCESS time."""

    __slots__ = ()

    def __getattr__(self, name):
        if get_array_backend() == "numpy":
            return getattr(_np, name)
        import jax.numpy as jnp
        return getattr(jnp, name)

    def __repr__(self):  # pragma: no cover - debugging nicety
        return f"<array-backend proxy: {get_array_backend()}>"


xp = _ArrayModuleProxy()


def solve_pos(A, b):
    """Positive-definite linear solve on the active backend.

    Backend twins of ``jax.scipy.linalg.solve(A, b, assume_a='pos')`` /
    ``scipy.linalg.solve(A, b, assume_a='pos')`` — both Cholesky-based.
    """
    if get_array_backend() == "numpy":
        import scipy.linalg as _sla
        return _sla.solve(_np.asarray(A, dtype=_np.float64),
                          _np.asarray(b, dtype=_np.float64),
                          assume_a="pos")
    import jax
    return jax.scipy.linalg.solve(A, b, assume_a="pos")
