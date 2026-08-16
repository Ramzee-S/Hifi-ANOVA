"""Goodness-of-fit metrics — the two coefficient-of-determination conventions.

Two R² conventions circulate, and they diverge whenever the residual carries a
nonzero empirical mean (a biased / regularized / no-intercept fit); they
coincide *exactly* iff ``mean(y - ŷ) = 0``:

``'explained_variance'`` — the framework-native quantity (library default)
    ``R²_EV = 1 − Var(y − ŷ) / Var(y)``
    Identical to scikit-learn's ``explained_variance_score``. It ignores any
    constant bias in the residual — it measures how much of the *variance* the
    prediction tracks. This is the convention the HiFi-ANOVA manuscript reports,
    because the whole accounting there is variance-based: the Sobol shares are
    ``S_u = Var(f_u)/Var(f)`` and the structural fidelity is
    ``𝔉 = ΣVar(f̂_u) / (ΣVar(f̂_u) + Var(ĝ))`` (Manuscript_Theoryv06 Eq. fidelity).
    Keeping it as the default is what keeps the library's reported numbers
    consistent with the manuscript.

``'classical'`` — the textbook coefficient of determination (aliases
``'r2_score'``, ``'sse_tss'``)
    ``R²_cls = 1 − Σ(y − ŷ)² / Σ(y − ȳ)²``
    Identical to scikit-learn's ``r2_score``. It *penalizes* a constant bias in
    the fit and can go negative. Report this when you want the standard,
    cross-library-comparable number.

Both are legitimate; the difference is only ever the residual mean (the fit's
bias). Use :func:`r_squared` for one convention or :func:`r_squared_report` for
both at once.
"""

import numpy as np

# Canonical convention names (aliases accepted by ``r_squared``).
R2_DEFINITIONS = ("explained_variance", "classical")

_EV_ALIASES = {"explained_variance", "ev", "explained"}
_CLASSICAL_ALIASES = {
    "classical", "r2_score", "sse_tss", "coefficient_of_determination",
}


def r_squared(y_true, y_pred, definition: str = "explained_variance") -> float:
    """Coefficient of determination under the requested convention.

    Args:
        y_true: observed targets, shape (N,).
        y_pred: predicted mean, shape (N,).
        definition: ``'explained_variance'`` (default) or ``'classical'``
            (aliases ``'r2_score'`` / ``'sse_tss'``). See the module docstring
            for the exact formulas and when they differ.

    Returns:
        The R² value as a Python float. Degenerate (``Var(y)=0`` /
        ``Σ(y−ȳ)²=0``) returns ``0.0`` — matching the library's historical
        guard — rather than raising.
    """
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    resid = yt - yp
    key = definition.lower()
    if key in _EV_ALIASES:
        var_y = float(np.var(yt))
        return 1.0 - float(np.var(resid)) / var_y if var_y > 0 else 0.0
    if key in _CLASSICAL_ALIASES:
        ss_tot = float(np.sum((yt - yt.mean()) ** 2))
        return 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else 0.0
    raise ValueError(
        f"Unknown R² definition {definition!r}; choose from {R2_DEFINITIONS} "
        f"(aliases: 'r2_score'/'sse_tss'/'ev')."
    )


def r_squared_report(y_true, y_pred) -> dict:
    """Both conventions at once: ``{'explained_variance', 'classical'}``."""
    return {
        "explained_variance": r_squared(y_true, y_pred, "explained_variance"),
        "classical": r_squared(y_true, y_pred, "classical"),
    }
