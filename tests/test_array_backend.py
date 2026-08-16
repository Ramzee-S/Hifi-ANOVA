"""NumPy exact-core backend: selection semantics + cross-backend parity.

The numpy backend runs the SAME fit-path code as jax through the
``hifi_anova.array_backend`` proxy, so parity here is a regression guard on
the indirection (import sweep, solver routing, result-method scoping) — the
statistics are shared by construction. Tiny fits only (CPU-friendly).
"""
import numpy as np
import pytest

import hifi_anova as ha
from hifi_anova.array_backend import (
    get_array_backend, set_array_backend, use_array_backend)


def _data(n=200, d=4, seed=0, hetero=False):
    rng = np.random.default_rng(seed)
    X = rng.random((n, d))
    sigma = 0.05 + (0.25 * X[:, 1] if hetero else 0.0)
    y = (np.sin(2 * np.pi * X[:, 0]) + X[:, 1]
         + 0.4 * X[:, d - 2] * X[:, d - 1]
         + sigma * rng.standard_normal(n))
    return X, y


def test_default_backend_is_jax_and_selection_api():
    assert get_array_backend() == "jax"
    with use_array_backend("numpy"):
        assert get_array_backend() == "numpy"
        with use_array_backend("jax"):
            assert get_array_backend() == "jax"
        assert get_array_backend() == "numpy"
    assert get_array_backend() == "jax"
    with pytest.raises(ValueError):
        set_array_backend("torch")


def test_backend_argument_validation_and_auto():
    X, y = _data(60, 3)
    with pytest.raises(ValueError, match="backend must be one of"):
        ha.hifi_anova(X, y, backend="torch", verbose=False)
    # numpy refuses the JAX-native (NN residual) surface with guidance;
    # the linear residual families are core-legal since BR-10/DEC-057.
    with pytest.raises(ValueError, match="JAX-native"):
        ha.hifi_anova(X, y, backend="numpy", verbose=False,
                      residual_nn={"enabled": True})
    # auto: numpy inside the supported surface, jax outside it
    r = ha.hifi_anova(X, y, K1=2, mode="first", backend="auto",
                      verbose=False, seed=42)
    assert r.config["array_backend"] == "numpy"
    from hifi_anova.api import _resolve_array_backend
    # Linear residual families route to the core on 'auto' (DEC-057)…
    assert (_resolve_array_backend(
        "auto", mode="second", residual="rff", config_kwargs={}) == "numpy")
    assert (_resolve_array_backend(
        "auto", mode="second", residual=None,
        config_kwargs={"residual": {"type": "nystrom"}}) == "numpy")
    # …while NN residuals and the stage-laddering modes stay JAX-native.
    assert (_resolve_array_backend(
        "auto", mode="second", residual=None,
        config_kwargs={"residual": {"type": "nn", "enabled": True}}) == "jax")
    assert (_resolve_array_backend(
        "auto", mode="full", residual=None, config_kwargs={}) == "jax")


def test_parity_second_order_float64():
    """Homoscedastic pairs fit: float64 on both backends ⇒ near machine-eps."""
    X, y = _data(250, 4, seed=1)
    kw = dict(K1=3, K2=2, mode="second", basis_name="legendre",
              pair_selection=[2, 3], verbose=False, seed=42,
              precision="float64")
    rn = ha.hifi_anova(X, y, backend="numpy", **kw)
    rj = ha.hifi_anova(X, y, backend="jax", **kw)
    assert rn.config["array_backend"] == "numpy"
    assert rj.config["array_backend"] == "jax"
    assert np.max(np.abs(rn.predict(X) - rj.predict(X))) < 1e-10
    assert abs(float(rn.loo_cv) - float(rj.loo_cv)) < 1e-12
    s_n, s_j = (r.sobol["mean_sobol"] for r in (rn, rj))
    for k in s_j["first_order"]:
        assert abs(float(s_n["first_order"][k])
                   - float(s_j["first_order"][k])) < 1e-12
    for k in (s_j.get("second_order") or {}):
        assert abs(float(s_n["second_order"][k])
                   - float(s_j["second_order"][k])) < 1e-12
    # result compute methods are scoped to the recorded backend
    assert isinstance(np.asarray(rn.predict(X[:3])), np.ndarray)


def test_parity_stage_d_term_structure():
    """Stage-D + K2 mapping + K2h + var pair + variance_variables: the
    alternating loop, Newton solve and guards agree across backends."""
    X, y = _data(300, 4, seed=2, hetero=True)
    kw = dict(K1=3, mode="second", basis_name="legendre",
              heteroscedastic=True, K2={(2, 3): 2, (0, 1): 1}, K2h=2,
              var_pair_selection=[(0, 1)], variance_variables=[0, 1, 2],
              verbose=False, seed=42, precision="float64")
    rn = ha.hifi_anova(X, y, backend="numpy", **kw)
    rj = ha.hifi_anova(X, y, backend="jax", **kw)
    assert np.max(np.abs(rn.predict(X) - rj.predict(X))) < 1e-8
    assert np.max(np.abs(np.asarray(rn.sigma_x2(X))
                         - np.asarray(rj.sigma_x2(X)))) < 1e-8
    assert rn.loo_tier == rj.loo_tier
    assert abs(float(rn.loo_nll) - float(rj.loo_nll)) < 1e-8
    v_n = rn.sobol.get("log_variance_sobol") or {}
    v_j = rj.sobol.get("log_variance_sobol") or {}
    for k in (v_j.get("first_order") or {}):
        assert abs(float(v_n["first_order"][k])
                   - float(v_j["first_order"][k])) < 1e-8
    assert (rn.loo_variance_floor_active == rj.loo_variance_floor_active)


def test_parity_mixed_k_and_numpy_default_float64():
    X, y = _data(200, 3, seed=3)
    bpv = {0: {"basis": "legendre", "K": 3},
           1: {"basis": "legendre", "K": 2},
           2: {"basis": "legendre", "K": 1}}
    kw = dict(K1=3, mode="first", basis_name="legendre",
              basis_per_variable=bpv, variable_selection=None,
              verbose=False, seed=42, precision="float64")
    rn = ha.hifi_anova(X, y, backend="numpy", **kw)
    rj = ha.hifi_anova(X, y, backend="jax", **kw)
    assert np.max(np.abs(rn.predict(X) - rj.predict(X))) < 1e-10
    # DEC-056: the NumPy core is float64-only — backend='numpy' with no
    # precision arg is float64 (not the JAX float32 default). It agrees with a
    # JAX float32 fit at f32 noise (float64 core vs float32 jax).
    r64 = ha.hifi_anova(X, y, K1=2, mode="first", basis_name="legendre",
                        verbose=False, seed=42, backend="numpy")
    j32 = ha.hifi_anova(X, y, K1=2, mode="first", basis_name="legendre",
                        verbose=False, seed=42, backend="jax")
    assert str(r64.model.mean_model.w1.dtype) == "float64"
    assert str(j32.model.mean_model.w1.dtype) == "float32"
    assert np.max(np.abs(r64.predict(X) - j32.predict(X))) < 1e-5


def test_parity_third_order_triples():
    """K3/triples run on the shared builders too — cross-backend parity."""
    rng = np.random.default_rng(4)
    X = rng.random((250, 3))
    y = (X[:, 0] + X[:, 1] * X[:, 2]
         + 0.5 * X[:, 0] * X[:, 1] * X[:, 2]
         + 0.05 * rng.standard_normal(250))
    kw = dict(K1=2, K2=1, K3=1, triple_selection="all_active", mode="second",
              basis_name="legendre", pair_selection=[0, 1, 2],
              verbose=False, seed=42, precision="float64")
    rn = ha.hifi_anova(X, y, backend="numpy", **kw)
    rj = ha.hifi_anova(X, y, backend="jax", **kw)
    assert np.max(np.abs(rn.predict(X) - rj.predict(X))) < 1e-10
    t_n = rn.sobol["mean_sobol"].get("third_order") or {}
    t_j = rj.sobol["mean_sobol"].get("third_order") or {}
    assert t_j and set(t_n) == set(t_j)
    for k in t_j:
        assert abs(float(t_n[k]) - float(t_j[k])) < 1e-12


def test_save_load_roundtrip_preserves_backend(tmp_path):
    """A numpy-core model round-trips through save/load AS numpy: the loader
    rebuilds the template under the backend recorded in the saved config, so
    the reloaded weights are numpy arrays and predictions match exactly."""
    X, y = _data(200, 3, seed=5)
    r = ha.hifi_anova(X, y, K1=3, mode="first", basis_name="legendre",
                      verbose=False, seed=42, precision="float64",
                      backend="numpy")
    p = str(tmp_path / "m")
    r.save(p)
    from hifi_anova.model.io import load_model
    loaded = load_model(p)
    assert loaded["config"].get("array_backend") == "numpy"
    w1 = loaded["model"].mean_model.w1
    assert type(w1).__name__ == "ndarray"      # numpy, not a jax Array
    yh0 = np.asarray(r.predict(X[:20]))
    xt = loaded["transformer"].transform(X[:20])
    yh1 = np.asarray(loaded["model"].predict_mean_only(
        np.asarray(xt, w1.dtype)))
    assert np.max(np.abs(yh0 - yh1)) < 1e-8
