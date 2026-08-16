"""Fit-precision control: this knob governs the **JAX backend only** (DEC-035).

The precision *choice* — float32 (default, GPU-speed) vs float64 — applies to a
**JAX** fit. The **NumPy exact core is always float64** (DEC-056): it is a
float64 engine with no float32 mode, so this setting does not apply to it. Since
the default backend is ``auto`` (⇒ NumPy for a non-residual config), the default
*fit* a plain ``hifi_anova(X, y)`` call produces is **float64**; an explicit
``precision='float32'`` selects the JAX float32 path (``auto`` routes there —
float32 is a JAX-native speed mode; see ``api._resolve_array_backend``).

On the JAX path the fit precision resolves, in precedence order:

1. an explicit ``precision=`` argument (e.g. ``hifi_anova(X, y, precision="float64")``
   or ``config['precision']``),
2. ``set_fit_precision(...)`` (process-global; mainly tests / advanced use),
3. the ``HIFI_ANOVA_X64`` environment variable (``1``/``true``/``on`` → float64),
4. otherwise the default, ``"float32"`` (the JAX default; DEC-035).

The post-fit analytics (Sobol, CIs, prediction) always run in float64. Choosing
float64 additionally enables ``jax_enable_x64`` (a float64 array is silently
truncated to float32 by JAX unless x64 is on). It never *disables* x64.
"""

import os
import warnings

import jax
import jax.numpy as jnp

_DEFAULT = "float32"
_VALID = ("float32", "float64")

# Process-global override set by set_fit_precision(); None => fall through to env.
_override = None

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off", ""}


def _from_env():
    """Read HIFI_ANOVA_X64; return 'float64' (opt-in), else None (fall through).

    Recognized true values opt into float64; recognized false/empty values fall
    through to the next precedence level. An *unrecognized* value is not silently
    reinterpreted: it warns and is ignored (falls through), so a typo like
    ``HIFI_ANOVA_X64=ture`` does not quietly leave the fit at float32 while the
    caller believes float64 was requested. An explicit ``precision=`` argument or
    ``set_fit_precision`` is the unambiguous control.
    """
    raw = os.environ.get("HIFI_ANOVA_X64", "")
    v = raw.strip().lower()
    if v in _TRUE:
        return "float64"
    if v in _FALSE:
        return None
    warnings.warn(
        f"HIFI_ANOVA_X64={raw!r} is not a recognized boolean "
        f"(true: {sorted(_TRUE)}; false: {sorted(_FALSE - {''})}); ignoring it. "
        "The fit precision falls through to set_fit_precision / the float32 "
        "default. Pass precision='float64' for an unambiguous opt-in.",
        RuntimeWarning, stacklevel=2)
    return None


def resolve_precision(precision=None):
    """Resolve the effective fit precision string ('float32' | 'float64').

    Precedence: explicit ``precision`` arg > ``set_fit_precision`` override >
    ``HIFI_ANOVA_X64`` env > default ('float32').
    """
    if precision is not None:
        p = str(precision).lower()
    elif _override is not None:
        p = _override
    else:
        p = _from_env() or _DEFAULT
    if p not in _VALID:
        raise ValueError(
            f"precision must be 'float32' or 'float64'; got {precision!r}")
    return p


def wants_float32_fit(precision=None):
    """True iff an EXPLICIT source selects float32 — used to route ``auto`` to JAX.

    float32 is a JAX-native speed mode; the NumPy core is float64-only. So an
    explicit float32 request (a ``precision='float32'`` argument or a
    ``set_fit_precision('float32')`` override) makes ``auto`` pick JAX. The bare
    built-in default does NOT count — an unspecified precision is treated as "no
    preference" and lets ``auto`` stay on the float64 NumPy core. (The env var
    only ever opts *up* to float64, so it is not a float32 source.)
    """
    if precision is not None:
        return str(precision).lower() == "float32"
    return _override == "float32"


def fit_dtype(precision=None):
    """Return the JAX dtype for the fit (``jnp.float32`` | ``jnp.float64``).

    For ``float64`` this also enables ``jax_enable_x64`` so 64-bit arrays survive.
    """
    p = resolve_precision(precision)
    if p == "float64":
        jax.config.update("jax_enable_x64", True)
        return jnp.float64
    return jnp.float32


def set_fit_precision(precision):
    """Set a process-global fit-precision override (or ``None`` to clear it)."""
    global _override
    _override = None if precision is None else resolve_precision(precision)
    return _override
