"""BR-10: linear Stage-C residual families on the NumPy exact core.

Cross-backend float64 parity for the deterministic families (rbf/nystrom —
seeded k-means centers), the batched-forward ≡ vmap guarantee, the
ANOVA-coefficients-unchanged guarantee on both backends, save/load with a
residual attached, and the surviving JAX-native refusals. Tiny fits only.

RFF is deliberately absent from the cross-backend parity set: its random
frequencies are drawn backend-natively (jax.random vs np.random.default_rng),
so RFF fits are deterministic per backend but never comparable across
backends (DEC-057).
"""
import numpy as np
import pytest

import hifi_anova as ha
from hifi_anova.array_backend import use_array_backend


def _data(n=250, d=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.random((n, d))
    # K1=3/K2=2 truncation leaves real structure for the residual stage.
    y = (np.sin(6.0 * X[:, 0]) + X[:, 1]
         + 0.6 * np.sin(4.0 * X[:, 1] * X[:, 2])
         + 0.05 * rng.standard_normal(n))
    return X, y


def _fit(X, y, backend, family, **extra):
    kw = dict(K1=3, K2=2, mode="second", residual=family, verbose=False,
              seed=42, precision="float64", backend=backend)
    kw.update(extra)
    return ha.hifi_anova(X, y, **kw)


@pytest.mark.parametrize("family,size_kw", [
    ("rbf", {"n_centers": 15}),
    ("nystrom", {"n_inducing": 15}),
])
def test_parity_residual_float64(family, size_kw):
    """rbf/nystrom residual fits agree across backends at float64.

    Coefficients, predictions, and the fidelity block are shared-code
    results; centers come from seeded k-means, so both backends see the
    identical residual basis.
    """
    X, y = _data()
    rn = _fit(X, y, "numpy", family, **size_kw)
    rj = _fit(X, y, "jax", family, **size_kw)
    assert rn.config["array_backend"] == "numpy"
    assert rj.config["array_backend"] == "jax"

    # Residual weights: float64 on both (the float32 cast is gone — BR-10).
    an = rn.model.residual_net.weights
    aj = rj.model.residual_net.weights
    assert str(an.dtype) == "float64" and str(aj.dtype) == "float64"
    assert type(an).__module__ == "numpy"
    assert np.max(np.abs(np.asarray(an) - np.asarray(aj))) < 1e-8

    # Combined-model predictions.
    assert np.max(np.abs(np.asarray(rn.predict(X))
                         - np.asarray(rj.predict(X)))) < 1e-8

    # Fidelity bridge (only populated when a residual stage ran).
    fn = rn.sobol.get("fidelity") or {}
    fj = rj.sobol.get("fidelity") or {}
    assert fj.get("var_residual", 0.0) > 0.0
    assert abs(float(fn["value"]) - float(fj["value"])) < 1e-8


def test_anova_coefficients_unchanged_by_residual_both_backends():
    """The docstring guarantee: engaging Stage C leaves the fitted Fourier
    coefficients — hence the CORE-normalized Sobol attribution — exactly
    unchanged. (The headline ``mean_sobol`` is the 𝓕-scaled total bridge
    when a residual ran (M3/DEC-032), so it shifts by 𝓕 by design; the
    invariant set is ``mean_sobol_core``.)"""
    X, y = _data(seed=1)
    for backend in ("numpy", "jax"):
        r0 = ha.hifi_anova(X, y, K1=3, K2=2, mode="second", verbose=False,
                           seed=42, precision="float64", backend=backend)
        r1 = _fit(X, y, backend, "rbf", n_centers=15)
        w0 = np.asarray(r0.model.mean_model.w1)
        w1 = np.asarray(r1.model.mean_model.w1)
        assert np.array_equal(w0, w1), backend
        s0 = r0.sobol["mean_sobol_core"]["first_order"]
        s1 = r1.sobol["mean_sobol_core"]["first_order"]
        for k in s0:
            assert abs(float(s0[k]) - float(s1[k])) < 1e-12, backend


def test_orthogonality_after_projection_both_backends():
    """Φᵀ Z_proj ≈ 0 (in-sample) on both backends, straight from the
    shipped projection helper on a residual feature build."""
    from hifi_anova.core.projection import (
        project_features_orthogonal, verify_orthogonality)
    from hifi_anova.model.linear_residual import RBFResidual
    rng = np.random.default_rng(2)
    X = rng.random((150, 3))
    Phi = rng.random((150, 10))
    for backend in ("numpy", "jax"):
        with use_array_backend(backend):
            res = RBFResidual.create(X, n_centers=12, sigma=0.2)
            Z = res.build_features(X)
            Z_proj, _ = project_features_orthogonal(Z, Phi)
            check = verify_orthogonality(Z_proj, Phi, atol=1e-6)
            assert check["is_orthogonal"], (backend, check)


def test_predict_batch_matches_per_sample_and_vmap():
    """predict_batch is the batched twin of the per-sample __call__ (and of
    jax.vmap of it on the jax backend)."""
    X, y = _data(seed=3)
    # numpy core: predict_batch vs an explicit per-row __call__ loop.
    rn = _fit(X, y, "numpy", "rbf", n_centers=15)
    res_n = rn.model.residual_net
    xs = X[:20]
    batched = np.asarray(res_n.predict_batch(xs))
    looped = np.array([float(res_n(xs[i])) for i in range(len(xs))])
    assert np.max(np.abs(batched - looped)) < 1e-12

    # jax backend: predict_batch vs jax.vmap at float64.
    import jax
    rj = _fit(X, y, "jax", "rbf", n_centers=15)
    res_j = rj.model.residual_net
    with use_array_backend("jax"):
        vm = np.asarray(jax.vmap(res_j)(xs))
        bt = np.asarray(res_j.predict_batch(xs))
    assert np.max(np.abs(vm - bt)) < 1e-12


def test_rff_backend_native_draws_deterministic_per_backend():
    """RFF is core-legal but backend-native: same-seed numpy fits are
    identical; the numpy and jax frequency draws differ by construction."""
    X, y = _data(seed=4)
    r1 = _fit(X, y, "numpy", "rff", n_features=30)
    r2 = _fit(X, y, "numpy", "rff", n_features=30)
    assert np.array_equal(np.asarray(r1.model.residual_net.omega),
                          np.asarray(r2.model.residual_net.omega))
    assert np.array_equal(np.asarray(r1.predict(X[:10])),
                          np.asarray(r2.predict(X[:10])))
    rj = _fit(X, y, "jax", "rff", n_features=30)
    assert not np.allclose(np.asarray(r1.model.residual_net.omega),
                           np.asarray(rj.model.residual_net.omega))


def test_save_load_roundtrip_numpy_residual(tmp_path):
    """A numpy-core fit with a residual attached round-trips through the
    full-model pickle fallback; reloaded predictions match."""
    X, y = _data(seed=5)
    r = _fit(X, y, "numpy", "rbf", n_centers=15)
    p = str(tmp_path / "m")
    r.save(p)
    from hifi_anova.model.io import load_model
    loaded = load_model(p)
    assert loaded["config"].get("array_backend") == "numpy"
    model = loaded["model"]
    assert model.residual_net is not None
    assert type(model.residual_net.weights).__module__ == "numpy"
    xt = loaded["transformer"].transform(X[:20])
    with use_array_backend("numpy"):
        yh1 = np.asarray(model.predict(np.asarray(xt, dtype=np.float64))[0])
    yh0 = np.asarray(r.predict(X[:20]))
    assert np.max(np.abs(yh0 - yh1)) < 1e-8


def test_numpy_refusals_nn_and_modes():
    """What stays JAX-native: the NN residual and the stage-laddering modes."""
    X, y = _data(120, seed=6)
    with pytest.raises(ValueError, match="JAX-native"):
        ha.hifi_anova(X, y, backend="numpy", verbose=False,
                      residual_nn={"enabled": True})
    with pytest.raises(ValueError, match="JAX-native"):
        ha.hifi_anova(X, y, backend="numpy", verbose=False, mode="full")
    from hifi_anova.api import _resolve_array_backend
    with pytest.raises(ValueError, match="JAX-native"):
        _resolve_array_backend(
            "numpy", mode="second", residual=None,
            config_kwargs={"residual": {"type": "nn", "enabled": True}})


def test_analyze_residuals_includes_attached_residual():
    """analyze_residuals subtracts the attached residual model (and phi3):
    the analyzed leftover shrinks once Stage C has captured structure.
    (X is already in the unit cube, so the models apply directly.)"""
    from hifi_anova.analysis import analyze_residuals
    X, y = _data(seed=7)
    r0 = ha.hifi_anova(X, y, K1=3, K2=2, mode="second", verbose=False,
                       seed=42, precision="float64", backend="numpy")
    r1 = _fit(X, y, "numpy", "rbf", n_centers=15)
    with use_array_backend("numpy"):
        d0 = analyze_residuals(r0.model, X, y)
        d1 = analyze_residuals(r1.model, X, y)
    assert d1.residual_variance < d0.residual_variance


# ======== BR-11/BR-12 (Ses06): solved-layout projection + intercept-only ====

def _manual_complement(res, X, family_cfg=None):
    """Attach a complement the engine way: create → project vs the SOLVED
    design → ridge → _create_fitted_residual. Returns (fitted, Phi_solved)."""
    from hifi_anova.training.analytic_residual import (
        create_residual, _create_fitted_residual)
    from hifi_anova.core.projection import project_features_orthogonal
    from hifi_anova.training.ridge import weighted_ridge_solve
    m = res.model
    cfg = {"seed": 42, "n_inducing": 25}
    cfg.update(family_cfg or {})
    obj = create_residual("nystrom", cfg, X, X.shape[1])
    Z = obj.build_features(X)
    Phi = m.build_phi_all_fit(X)
    Zp, C = project_features_orthogonal(Z, Phi)
    # the layout tests only need SOME fitted residual — zero targets give
    # alpha=0, which exercises the same rebuild/projection machinery
    alpha = weighted_ridge_solve(np.asarray(Zp, float),
                                 np.zeros(Zp.shape[0]),
                                 np.full(Zp.shape[1], 1.0))
    return _create_fitted_residual(obj, alpha, C, m), np.asarray(Phi, float)


@pytest.mark.parametrize("kw,expect", [
    # mixed per-variable K: var_specs carried into the residual
    (dict(K1=4, K2=0, mode="first", variable_selection=None,
          basis_per_variable={0: {"basis": "fourier", "K": 4},
                              1: {"basis": "fourier", "K": 2},
                              2: {"basis": "fourier", "K": 3}}), "var_specs"),
    # per-pair K2 pinning: ragged pair layout carried
    (dict(K1=3, K2={(0, 1): 2, (1, 2): 1}, mode="second",
          variable_selection=None), "pair_k2"),
    # order-selective first-order subset
    (dict(K1=3, K2=0, mode="first", variable_selection=None,
          heteroscedastic=True, variable_orders={2: []}), "fo_included"),
])
def test_solved_layout_prediction_rebuild(kw, expect):
    """BR-11: a fitted residual rebuilds the SOLVED design layout at
    prediction time — byte-equal to ``model.build_phi_all_fit`` on NEW data —
    for every non-uniform layout the trainer produces."""
    X, y = _data()
    with use_array_backend("numpy"):
        res = ha.hifi_anova(X, y, verbose=False, seed=42,
                            precision="float64", backend="numpy", **kw)
        m = res.model
        assert getattr(m, expect) is not None
        fitted, Phi = _manual_complement(res, X)
        # the layout fields were copied from the model
        assert getattr(fitted, expect) == getattr(m, expect)
        Xn = np.random.default_rng(7).random((60, X.shape[1]))
        rebuilt = np.asarray(fitted._fourier_batch(Xn), float)
        want = np.asarray(m.build_phi_all_fit(Xn), float)
        assert rebuilt.shape == want.shape
        assert np.max(np.abs(rebuilt - want)) == 0.0
        # in-sample orthogonality of the prediction-path projection
        zin = np.asarray(fitted.build_features(X), float) \
            - np.asarray(fitted._fourier_batch(X), float) \
            @ np.asarray(fitted.proj_coeffs, float)
        assert np.max(np.abs(Phi.T @ zin)) < 1e-6 * X.shape[0]


def test_one_call_residual_on_order_selective_fit():
    """BR-11 end to end through the one-call API: a residual attached to a
    variable_orders fit projects against the SOLVED design, so the excluded
    variable's first-order structure is captured by the residual (before
    Ses06 the full-layout projection would have removed it)."""
    X, y = _data()
    with use_array_backend("numpy"):
        res = ha.hifi_anova(X, y, K1=3, K2=0, mode="first",
                            variable_selection=None, heteroscedastic=True,
                            variable_orders={0: []}, residual="rbf",
                            n_centers=40, verbose=False, seed=42,
                            precision="float64", backend="numpy")
        m = res.model
        assert m.fo_included == (1, 2)
        rn = m.residual_net
        assert rn is not None and rn.fo_included == (1, 2)
        from hifi_anova.model.linear_residual import predict_residual_batch
        g = np.asarray(predict_residual_batch(rn, X), float)
        # x0 carries sin(6 x0) — the base cannot fit it; the residual must:
        # its output correlates strongly with the excluded marginal
        marg = np.sin(6.0 * X[:, 0])
        cc = np.corrcoef(g, marg - marg.mean())[0, 1]
        assert abs(cc) > 0.6


def test_intercept_only_fit_and_complement():
    """BR-12: variable_orders all-[] on a CONSTANT fit ⇒ an intercept-only
    mean (fo_included=(), empty solved design, predictions == f0, disclosed
    by warning) whose complement captures everything above f0 (projection
    no-op)."""
    X, y = _data()
    with use_array_backend("numpy"):
        with pytest.warns(UserWarning, match="INTERCEPT-ONLY"):
            res = ha.hifi_anova(X, y, K1=3, K2=0, mode="first",
                                variable_selection=None,
                                variable_orders={0: [], 1: [], 2: []},
                                verbose=False, seed=42,
                                precision="float64", backend="numpy")
        m = res.model
        assert m.fo_included == ()
        rec = res._fitted_design
        assert np.asarray(rec.Phi).shape[1] == 0
        f0 = float(np.asarray(m.mean_model.f0))
        pred, _ = m.predict(X)
        assert np.allclose(np.asarray(pred), f0)
        # complement: projection is a no-op, the residual fits y - f0
        from hifi_anova.training.analytic_residual import (
            create_residual, _create_fitted_residual)
        from hifi_anova.core.projection import project_features_orthogonal
        from hifi_anova.training.ridge import weighted_ridge_solve
        from hifi_anova.model.linear_residual import predict_residual_batch
        obj = create_residual("nystrom",
                              {"seed": 42, "n_inducing": 60,
                               "lengthscale": 0.25}, X, 3)
        Z = obj.build_features(X)
        Zp, C = project_features_orthogonal(Z, m.build_phi_all_fit(X))
        assert np.asarray(C).shape == (0, 60)
        assert np.array_equal(np.asarray(Zp), np.asarray(Z, float))
        alpha = weighted_ridge_solve(np.asarray(Zp, float), y - f0,
                                     np.full(60, 1e-3))
        fitted = _create_fitted_residual(obj, alpha, C, m)
        assert fitted.fo_included == ()
        yhat = f0 + np.asarray(predict_residual_batch(fitted, X), float)
        r2 = 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)
        assert r2 > 0.9


def test_stage_a_vo_fit_carries_fo_included():
    """The Stage-A-only path (first-order, CONSTANT noise) stamps
    ``fo_included`` on the model — pre-Ses06 it silently claimed the FULL
    layout while the record held the subset."""
    X, y = _data()
    with use_array_backend("numpy"):
        with pytest.warns(UserWarning, match="neither the mean nor"):
            res = ha.hifi_anova(X, y, K1=3, K2=0, mode="first",
                                variable_selection=None,
                                variable_orders={1: []}, verbose=False,
                                seed=42, precision="float64",
                                backend="numpy")
        m = res.model
        assert m.fo_included == (0, 2)
        assert (np.asarray(res._fitted_design.Phi).shape[1]
                == np.asarray(m.build_phi_all_fit(X)).shape[1])
