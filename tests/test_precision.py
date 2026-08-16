"""The opt-in float64 fit button (DEC-035).

float32 stays the default; ``precision='float64'`` (or ``HIFI_ANOVA_X64`` /
``set_fit_precision``) opts into a genuinely float64 fit — the stored model
weights and the predictions come back float64. The default path is unchanged
(byte-identical; the golden master is the separate guard for that).
"""

import os

import numpy as np
import jax.numpy as jnp
import pytest

from hifi_anova import precision as P
from hifi_anova.api import hifi_anova


@pytest.fixture(autouse=True)
def _clean_precision_state():
    """Isolate each test from process-global override / env var."""
    P.set_fit_precision(None)
    saved = os.environ.pop("HIFI_ANOVA_X64", None)
    yield
    P.set_fit_precision(None)
    if saved is not None:
        os.environ["HIFI_ANOVA_X64"] = saved
    else:
        os.environ.pop("HIFI_ANOVA_X64", None)


# --- resolver precedence -------------------------------------------------

def test_default_is_float32():
    assert P.resolve_precision() == "float32"
    assert P.fit_dtype() == jnp.float32


def test_explicit_float64():
    assert P.resolve_precision("float64") == "float64"
    assert P.fit_dtype("float64") == jnp.float64


def test_explicit_arg_wins_over_override_and_env():
    P.set_fit_precision("float64")
    os.environ["HIFI_ANOVA_X64"] = "0"
    assert P.resolve_precision("float32") == "float32"


def test_override_wins_over_env():
    os.environ["HIFI_ANOVA_X64"] = "0"
    P.set_fit_precision("float64")
    assert P.resolve_precision() == "float64"


def test_env_toggles_both_ways():
    os.environ["HIFI_ANOVA_X64"] = "1"
    assert P.resolve_precision() == "float64"
    os.environ["HIFI_ANOVA_X64"] = "off"
    assert P.resolve_precision() == "float32"


def test_invalid_precision_raises():
    with pytest.raises(ValueError):
        P.resolve_precision("float16")


def test_invalid_env_warns_and_is_ignored():
    # An unrecognized HIFI_ANOVA_X64 is not silently reinterpreted: it warns and
    # falls through to the float32 default (DEC-044).
    os.environ["HIFI_ANOVA_X64"] = "ture"  # typo
    with pytest.warns(RuntimeWarning, match="HIFI_ANOVA_X64"):
        assert P.resolve_precision() == "float32"


# --- end-to-end fit dtype ------------------------------------------------

@pytest.mark.parametrize("prec,expect", [("float32", jnp.float32),
                                         ("float64", jnp.float64)])
def test_fit_and_predict_dtype_follows_precision(prec, expect):
    rng = np.random.RandomState(0)
    X = rng.rand(300, 3)
    y = np.sin(2 * X[:, 0]) + X[:, 1] ** 2 + 0.1 * rng.randn(300)
    res = hifi_anova(X, y, K1=4, K2=2, verbose=False, precision=prec)
    assert res.model.mean_model.w1.dtype == expect
    assert np.asarray(res.predict(X[:4])).dtype == expect


def test_default_call_is_float64_via_numpy_core():
    """DEC-056: the DEFAULT fit is float64 — a plain call resolves backend='auto'
    ⇒ the NumPy exact core, a float64 engine. (Pre-DEC-056 this was float32.)"""
    rng = np.random.RandomState(0)
    X = rng.rand(300, 3)
    y = np.sin(2 * X[:, 0]) + 0.1 * rng.randn(300)
    res = hifi_anova(X, y, K1=4, K2=2, verbose=False)  # no precision, no backend
    assert res.model.mean_model.w1.dtype == jnp.float64
    assert res.config["array_backend"] == "numpy"
    assert res.config["precision"] == "float64"


def test_jax_backend_default_is_float32():
    """DEC-035 preserved for the JAX path: backend='jax' with no precision arg
    still fits float32 (the historical GPU-speed default)."""
    rng = np.random.RandomState(0)
    X = rng.rand(300, 3)
    y = np.sin(2 * X[:, 0]) + 0.1 * rng.randn(300)
    res = hifi_anova(X, y, K1=4, K2=2, verbose=False, backend="jax")
    assert res.model.mean_model.w1.dtype == jnp.float32
    assert res.config["precision"] == "float32"


def test_explicit_numpy_rejects_float32():
    """The NumPy core is float64-only: backend='numpy' + precision='float32'
    raises rather than silently returning float64 (DEC-056)."""
    rng = np.random.RandomState(0)
    X = rng.rand(120, 3)
    y = np.sin(2 * X[:, 0]) + 0.1 * rng.randn(120)
    with pytest.raises(ValueError, match="float64-only"):
        hifi_anova(X, y, K1=4, K2=2, verbose=False,
                   backend="numpy", precision="float32")


# --- one-call precedence: env / global override reach an ordinary fit --------
# Regression for DEC-044: the old signature hard-coded precision='float32' and
# passed it explicitly, so HIFI_ANOVA_X64 / set_fit_precision never affected a
# normal hifi_anova() call. The resolver was correct; the boundary bypassed it.

def _small_problem():
    rng = np.random.RandomState(0)
    X = rng.rand(300, 3)
    y = np.sin(2 * X[:, 0]) + X[:, 1] ** 2 + 0.1 * rng.randn(300)
    return X, y


def test_onecall_env_true_implicit_is_float64():
    X, y = _small_problem()
    os.environ["HIFI_ANOVA_X64"] = "1"
    res = hifi_anova(X, y, K1=4, K2=2, verbose=False)  # no precision arg
    assert res.model.mean_model.w1.dtype == jnp.float64
    assert res.config["precision"] == "float64"  # effective value recorded


def test_onecall_override_implicit_is_float64():
    X, y = _small_problem()
    P.set_fit_precision("float64")
    res = hifi_anova(X, y, K1=4, K2=2, verbose=False)  # no precision arg
    assert res.model.mean_model.w1.dtype == jnp.float64
    assert res.config["precision"] == "float64"


def test_onecall_override_beats_env():
    X, y = _small_problem()
    os.environ["HIFI_ANOVA_X64"] = "0"
    P.set_fit_precision("float64")
    res = hifi_anova(X, y, K1=4, K2=2, verbose=False)
    assert res.model.mean_model.w1.dtype == jnp.float64


def test_onecall_explicit_float32_beats_override_and_env():
    X, y = _small_problem()
    P.set_fit_precision("float64")
    os.environ["HIFI_ANOVA_X64"] = "1"
    res = hifi_anova(X, y, K1=4, K2=2, verbose=False, precision="float32")
    assert res.model.mean_model.w1.dtype == jnp.float32
    assert res.config["precision"] == "float32"


def test_onecall_explicit_float64_beats_off_env():
    X, y = _small_problem()
    os.environ["HIFI_ANOVA_X64"] = "off"
    res = hifi_anova(X, y, K1=4, K2=2, verbose=False, precision="float64")
    assert res.model.mean_model.w1.dtype == jnp.float64


def test_onecall_env_true_save_load_roundtrips_float64(tmp_path):
    from hifi_anova.model.io import load_model
    X, y = _small_problem()
    os.environ["HIFI_ANOVA_X64"] = "1"
    res = hifi_anova(X, y, K1=4, K2=2, verbose=False)  # implicit → float64
    res.save(str(tmp_path))
    loaded = load_model(str(tmp_path))
    assert loaded["model"].mean_model.w1.dtype == jnp.float64
    assert loaded["config"].get("precision") == "float64"


@pytest.mark.parametrize("prec,expect", [("float32", jnp.float32),
                                         ("float64", jnp.float64)])
def test_save_load_roundtrips_dtype(tmp_path, prec, expect):
    # The deserialization template must rebuild with the saved fit dtype, else
    # Equinox rejects float64 leaves against a float32 skeleton (DEC-035 io fix).
    from hifi_anova.model.io import load_model
    rng = np.random.RandomState(0)
    X = rng.rand(300, 3)
    y = np.sin(2 * X[:, 0]) + 0.1 * rng.randn(300)
    res = hifi_anova(X, y, K1=4, K2=2, verbose=False, precision=prec)
    pred0 = np.asarray(res.predict(X[:5]))
    res.save(str(tmp_path))
    loaded = load_model(str(tmp_path))
    assert loaded['model'].mean_model.w1.dtype == expect
    x = jnp.array(np.clip(loaded['transformer'].transform(X[:5]), 0, 1),
                  dtype=loaded['model'].mean_model.f0.dtype)
    pred1 = np.asarray(loaded['model'].predict_mean_only(x))
    # The reload must reproduce the predictions. float32 is bit-exact; float64
    # via the NumPy core (DEC-056) differs by at most one ULP because res.predict
    # and model.predict_mean_only take marginally different float64 reduction
    # orders — a rounding artifact, not a round-trip loss (the weights reload
    # bit-exact; the dtype assertion above is the primary check).
    if expect is jnp.float32:
        np.testing.assert_array_equal(pred0, pred1)
    else:
        np.testing.assert_allclose(pred0, pred1, rtol=0, atol=1e-12)
