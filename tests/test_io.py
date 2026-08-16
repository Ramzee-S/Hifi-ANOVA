"""Tests for model save/load (hifi_anova.model.io).

Smoke: the deserialization-template guards (clear errors for the structures the
auto-template cannot rebuild). Integration: exact save/load round-trip for
first- and second-order homoscedastic models — the K2>0 case regressed when the
index arrays became dynamic pytree leaves and is the reason for the template fix.
"""

import os
import json

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from hifi_anova.model.io import (
    save_model, load_model, _build_template_model, _NeedsFullModel)


# ── template guards (smoke, hermetic) ────────────────────────────────────────

@pytest.mark.smoke
def test_template_signals_full_model_for_unsupported_structures():
    """The metadata template can rebuild a plain mean model, but signals
    _NeedsFullModel for a variance model, constant variance, residual net, or
    mixed basis — load_model then restores those from the full-model pickle."""
    base = {'D': 3, 'K1': 3, 'K2': 0, 'K3': 0, 'basis_name': 'fourier'}
    # a plain first-order model builds fine
    _build_template_model({**base, 'is_mixed': False})
    for bad in ('has_variance_model', 'has_constant_log_var',
                'has_residual', 'is_mixed'):
        with pytest.raises(_NeedsFullModel):
            _build_template_model({**base, bad: True})


@pytest.mark.smoke
def test_template_second_order_shapes():
    """The K2>0 template reconstructs the second-order leaf shapes from P."""
    from hifi_anova.core.features import basis_size
    meta = {'D': 4, 'K1': 5, 'K2': 3, 'K3': 0, 'basis_name': 'fourier',
            'P': 6, 'T': 0, 'is_mixed': False}
    tmpl = _build_template_model(meta)
    B2 = basis_size(3, True, 'fourier')
    assert len(tmpl.mean_model.w2) == 6 * B2 * B2
    assert tmpl.pair_indices is not None and tmpl.pair_indices.shape == (6, 2)


# ── round-trip (integration, hermetic fit) ───────────────────────────────────

def _fit(cfg):
    from hifi_anova.training.trainer import HiFiANOVATrainer
    rng = np.random.RandomState(0)
    x = rng.rand(400, 4).astype(np.float32)
    y = (2 * x[:, 0] + 3 * x[:, 1] + x[:, 0] * x[:, 1]).astype(np.float32)
    xt, yt = jnp.array(x[:300]), jnp.array(y[:300])
    xv, yv = jnp.array(x[300:]), jnp.array(y[300:])
    model, _ = HiFiANOVATrainer(cfg).fit(xt, yt, xv, yv)
    return model, xv


@pytest.mark.integration
def test_roundtrip_first_order(tmp_path):
    model, xv = _fit({'K1': 5, 'K2': 0, 'stages': ['A']})
    pred_before, _ = model.predict(xv)
    save_model(model, str(tmp_path / 'm'), overwrite=True)
    m2 = load_model(str(tmp_path / 'm'))['model']
    pred_after, _ = m2.predict(xv)
    assert float(jnp.max(jnp.abs(pred_before - pred_after))) == 0.0
    assert m2.pair_indices is None


@pytest.mark.integration
def test_result_keys_persist_canonically_and_legacy_artifact_loads(tmp_path):
    """New JSON stores canonical names; old JSON is migrated with read aliases."""
    model, _ = _fit({'K1': 3, 'K2': 0, 'stages': ['A']})
    results = {
        'sobol': {'variance_sobol': {'first_order': {0: 1.0}}},
        'analytics': {'tr_HHt': 2.5},
    }
    out = tmp_path / 'm'
    save_model(model, str(out), overwrite=True, results=results)

    persisted = json.load(open(out / 'results.json'))
    assert 'log_variance_sobol' in persisted['sobol']
    assert 'variance_sobol' not in persisted['sobol']
    assert persisted['analytics']['tr_H2'] == 2.5
    assert 'tr_HHt' not in persisted['analytics']

    # Simulate an adjacent legacy artifact and verify exact-value reads.
    json.dump(results, open(out / 'results.json', 'w'))
    loaded = load_model(str(out))['results']
    with pytest.warns(DeprecationWarning, match='variance_sobol'):
        assert loaded['sobol']['variance_sobol'] is loaded['sobol']['log_variance_sobol']
    with pytest.warns(DeprecationWarning, match='tr_HHt'):
        assert loaded['analytics']['tr_HHt'] == loaded['analytics']['tr_H2']


@pytest.mark.integration
def test_roundtrip_second_order(tmp_path):
    """K2>0: pair_indices (dynamic integer leaf) and w2 must round-trip exactly."""
    model, xv = _fit({'K1': 5, 'K2': 3, 'stages': ['A', 'B']})
    assert model.pair_indices is not None and model.pair_indices.shape[0] > 0
    pred_before, _ = model.predict(xv)
    save_model(model, str(tmp_path / 'm'), overwrite=True)
    m2 = load_model(str(tmp_path / 'm'))['model']
    pred_after, _ = m2.predict(xv)
    assert float(jnp.max(jnp.abs(pred_before - pred_after))) == 0.0
    assert np.array_equal(np.asarray(model.pair_indices),
                          np.asarray(m2.pair_indices))
    assert len(model.mean_model.w2) == len(m2.mean_model.w2)


# ── round-trip for heteroscedastic / constant-variance / residual models ─────
# These structures can't be rebuilt from metadata; load_model restores them from
# the full-model pickle. Verify mean AND variance predictions round-trip exactly.

def _fit_hetero(cfg, hetero_data=True, seed=1):
    from hifi_anova.training.trainer import HiFiANOVATrainer
    from hifi_anova.data.synthetic import generate_ishigami
    from hifi_anova.data.preprocessing import preprocess_data
    if hetero_data:
        X, y, _ = generate_ishigami(1500, heteroscedastic=True,
                                    variance_variable=2, seed=seed)
    else:  # near-noiseless -> Stage D reverts to a constant variance
        X, _, _ = generate_ishigami(1500, seed=seed)
        y = np.asarray(np.sin(X[:, 0]))
    d = preprocess_data(X, y, seed=0)
    model, _ = HiFiANOVATrainer(cfg).fit(
        d['x_train'], d['y_train'], d['x_val'], d['y_val'])
    return model, d['x_test']


@pytest.mark.integration
def test_roundtrip_heteroscedastic(tmp_path):
    """A fitted variance model round-trips (mean + input-dependent variance)."""
    model, xv = _fit_hetero(
        {'K1': 6, 'K2': 3, 'Kh': 3, 'stages': ['A', 'B', 'D'],
         'strategy': 'curvature', 'max_outer_iter': 8})
    assert model.variance_model is not None            # kept (real hetero)
    m_before, v_before = model.predict(xv)
    save_model(model, str(tmp_path / 'm'), overwrite=True)
    m2 = load_model(str(tmp_path / 'm'))['model']
    m_after, v_after = m2.predict(xv)
    assert m2.variance_model is not None
    assert float(jnp.max(jnp.abs(m_before - m_after))) == 0.0
    assert float(jnp.max(jnp.abs(v_before - v_after))) == 0.0
    assert float(jnp.std(v_after)) > 0                 # variance is input-dependent


@pytest.mark.integration
def test_roundtrip_constant_variance_fallback(tmp_path):
    """The guard's constant-variance fallback (variance_model=None +
    constant_log_var) round-trips, including its constant variance."""
    model, xv = _fit_hetero(
        {'K1': 6, 'K2': 3, 'Kh': 3, 'stages': ['A', 'B', 'D'],
         'strategy': 'curvature', 'max_outer_iter': 8}, hetero_data=False)
    assert model.variance_model is None                # reverted
    assert model.constant_log_var is not None
    m_before, v_before = model.predict(xv)
    save_model(model, str(tmp_path / 'm'), overwrite=True)
    m2 = load_model(str(tmp_path / 'm'))['model']
    m_after, v_after = m2.predict(xv)
    assert m2.variance_model is None and m2.constant_log_var is not None
    assert float(jnp.max(jnp.abs(m_before - m_after))) == 0.0
    assert float(jnp.max(jnp.abs(v_before - v_after))) == 0.0
    assert float(jnp.std(v_after)) < 1e-8              # constant variance


@pytest.mark.integration
def test_roundtrip_linear_residual(tmp_path):
    """A model with a linear (RBF) residual net round-trips exactly."""
    from hifi_anova.training.trainer import HiFiANOVATrainer
    from hifi_anova.data.synthetic import generate_ishigami
    from hifi_anova.data.preprocessing import preprocess_data
    X, y, _ = generate_ishigami(1200, noise_std=0.3, seed=2)
    d = preprocess_data(X, y, seed=0)
    model, _ = HiFiANOVATrainer(
        {'K1': 6, 'K2': 3, 'stages': ['A', 'B', 'C'],
         'residual': {'type': 'rbf', 'n_centers': 80, 'sigma': 0.2}}).fit(
        d['x_train'], d['y_train'], d['x_val'], d['y_val'])
    assert model.residual_net is not None
    m_before = np.asarray(model.predict_mean_only(d['x_test']))
    save_model(model, str(tmp_path / 'm'), overwrite=True)
    m2 = load_model(str(tmp_path / 'm'))['model']
    m_after = np.asarray(m2.predict_mean_only(d['x_test']))
    assert m2.residual_net is not None
    assert float(np.max(np.abs(m_before - m_after))) == 0.0


@pytest.mark.smoke
def test_load_errors_clearly_without_pickle(tmp_path):
    """An old save (variance model, no model.pkl) gives a clear, actionable error."""
    model, _ = _fit_hetero(
        {'K1': 5, 'K2': 3, 'Kh': 3, 'stages': ['A', 'B', 'D'],
         'strategy': 'curvature', 'max_outer_iter': 8})
    save_model(model, str(tmp_path / 'm'), overwrite=True)
    os.remove(str(tmp_path / 'm' / 'model.pkl'))       # simulate a pre-pickle save
    with pytest.raises(NotImplementedError, match="model.pkl"):
        load_model(str(tmp_path / 'm'))


# ── Stage-D mean-estimator convention metadata (DEC-039 provenance) ──────────

def _fit_with_results(cfg, seed=1):
    """Like ``_fit_hetero`` but also returns the training-results dict."""
    from hifi_anova.training.trainer import HiFiANOVATrainer
    from hifi_anova.data.synthetic import generate_ishigami
    from hifi_anova.data.preprocessing import preprocess_data
    X, y, _ = generate_ishigami(1500, heteroscedastic=True,
                                variance_variable=2, seed=seed)
    d = preprocess_data(X, y, seed=0)
    model, results = HiFiANOVATrainer(cfg).fit(
        d['x_train'], d['y_train'], d['x_val'], d['y_val'])
    return model, results


@pytest.mark.integration
def test_mean_intercept_mode_persisted_heteroscedastic(tmp_path):
    """A Stage-D save carries the effective mean-estimator convention, and load
    returns it verbatim (DEC-039)."""
    from hifi_anova.training.fitted_design import (
        MEAN_INTERCEPT_PROFILED_JOINT_GLS, MEAN_INTERCEPT_LEGACY_UNKNOWN)
    model, results = _fit_with_results(
        {'K1': 6, 'K2': 3, 'Kh': 3, 'stages': ['A', 'B', 'D'],
         'strategy': 'curvature', 'max_outer_iter': 8})
    assert results['stage_D']['selected'] == 'heteroscedastic'
    save_model(model, str(tmp_path / 'm'), overwrite=True, results=results)

    meta = json.load(open(str(tmp_path / 'm' / 'meta.json')))
    assert meta['mean_intercept_mode'] == MEAN_INTERCEPT_PROFILED_JOINT_GLS
    loaded = load_model(str(tmp_path / 'm'))
    assert loaded['meta']['mean_intercept_mode'] == MEAN_INTERCEPT_PROFILED_JOINT_GLS

    # Legacy artifact: an older heteroscedastic save predating the field. Strip it
    # and confirm load supplies the defined 'legacy_unknown' interpretation (the
    # Stage-D mean vintage cannot be recovered from metadata).
    del meta['mean_intercept_mode']
    json.dump(meta, open(str(tmp_path / 'm' / 'meta.json'), 'w'))
    reloaded = load_model(str(tmp_path / 'm'))
    assert reloaded['meta']['has_variance_model'] is True
    assert reloaded['meta']['mean_intercept_mode'] == MEAN_INTERCEPT_LEGACY_UNKNOWN


@pytest.mark.integration
def test_mean_intercept_mode_homoscedastic_absent_then_inferred(tmp_path):
    """A homoscedastic save omits the Stage-D convention; load safely infers the
    ordinary unit-weight centered mean (DEC-039)."""
    from hifi_anova.training.fitted_design import MEAN_INTERCEPT_UNWEIGHTED
    model, xv = _fit({'K1': 5, 'K2': 0, 'stages': ['A']})
    save_model(model, str(tmp_path / 'm'), overwrite=True)   # no results/Stage D
    meta = json.load(open(str(tmp_path / 'm' / 'meta.json')))
    assert 'mean_intercept_mode' not in meta                 # not written
    loaded = load_model(str(tmp_path / 'm'))
    assert loaded['meta']['has_variance_model'] is False
    assert loaded['meta']['mean_intercept_mode'] == MEAN_INTERCEPT_UNWEIGHTED


@pytest.mark.integration
def test_mean_intercept_mode_retained_on_bare_save(tmp_path):
    """A bare ``save_model(model, path)`` — no results dict — still records the
    Stage-D provenance, because the effective convention is carried ON the model
    (DEC-047). Guards the low-level API the way ``HiFiResult.save`` is guarded."""
    from hifi_anova.training.fitted_design import MEAN_INTERCEPT_PROFILED_JOINT_GLS
    model, results = _fit_with_results(
        {'K1': 6, 'K2': 3, 'Kh': 3, 'stages': ['A', 'B', 'D'],
         'strategy': 'curvature', 'max_outer_iter': 8})
    assert results['stage_D']['selected'] == 'heteroscedastic'
    assert model.mean_intercept_mode == MEAN_INTERCEPT_PROFILED_JOINT_GLS

    # NOTE: no results= passed — the reviewer's failing sequence.
    save_model(model, str(tmp_path / 'm'), overwrite=True)
    meta = json.load(open(str(tmp_path / 'm' / 'meta.json')))
    assert meta['mean_intercept_mode'] == MEAN_INTERCEPT_PROFILED_JOINT_GLS
    loaded = load_model(str(tmp_path / 'm'))
    assert loaded['meta']['mean_intercept_mode'] == MEAN_INTERCEPT_PROFILED_JOINT_GLS


def test_model_convention_default_matches_canonical_constant():
    """The model field's literal default must stay in sync with the canonical
    constant (they are kept separate only to avoid a model↔training import cycle)."""
    import dataclasses
    from hifi_anova.model.hifi_anova import HiFiANOVA
    from hifi_anova.training.fitted_design import MEAN_INTERCEPT_UNWEIGHTED
    default = [f.default for f in dataclasses.fields(HiFiANOVA)
              if f.name == 'mean_intercept_mode'][0]
    assert default == MEAN_INTERCEPT_UNWEIGHTED
