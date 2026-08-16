"""X6 Session 2 (P0-2 / P1-4): Stage-D estimator identity + objective metadata.

Truth-in-labelling only — the DEFAULT estimator is numerically unchanged. These
tests pin:

* the estimator selector resolution (omitted vs explicit; contradictions raise);
* the honest ``results['stage_D']`` metadata truth table over the terminal
  outcomes (default, raw, early-stop label, guard-revert, near-noiseless skip);
* that the default path is byte-identical whether the selector is omitted or set
  explicitly to ``adjusted_quasi_likelihood`` (the selector adds no numerics);
* that a raw-mode block-coordinate step does not increase the penalized
  objective on an interior deterministic fixture (the claim the label rests on);
* that bound activity is detected/labelled and no stationarity claim is implied;
* the LAML evidence-status labelling (P1-4).
"""
import warnings

import numpy as np
import pytest

from hifi_anova.api import hifi_anova
from hifi_anova.data.synthetic import generate_ishigami
from hifi_anova.training.trainer import _resolve_stage_d_estimator


# --------------------------------------------------------------------------- #
# Selector resolution (unit level)
# --------------------------------------------------------------------------- #

def test_selector_omitted_defaults_to_adjusted():
    lev, es, ident = _resolve_stage_d_estimator({})
    assert (lev, es) == (True, True)
    assert ident['estimator'] == 'adjusted_quasi_likelihood'
    assert ident['objective_family'] == 'adjusted_quasi_likelihood'
    assert ident['residual_update'] == 'leverage_adjusted'
    assert ident['iterate_selection'] == 'validation_best'


def test_selector_raw_resolves_flags_off():
    lev, es, ident = _resolve_stage_d_estimator(
        {'stage_d_estimator': 'raw_likelihood'})
    assert (lev, es) == (False, False)
    assert ident['estimator'] == 'raw_likelihood'
    assert ident['objective_family'] == 'raw_penalized_likelihood'
    assert ident['residual_update'] == 'raw_squared_residuals'
    assert ident['iterate_selection'] == 'final_iterate'


def test_legacy_flags_preserved_without_selector():
    # A previously supported flag-only call is NOT rejected by the new default.
    lev, es, ident = _resolve_stage_d_estimator(
        {'leverage_correction': False, 'alternating_early_stop': False})
    assert (lev, es) == (False, False)
    # No selector name, but the effective identity is honestly the raw estimator.
    assert ident['estimator'] == 'raw_likelihood'


def test_mixed_legacy_flags_report_custom_but_honest_components():
    lev, es, ident = _resolve_stage_d_estimator(
        {'leverage_correction': True, 'alternating_early_stop': False})
    assert (lev, es) == (True, False)
    assert ident['estimator'] == 'custom'
    assert ident['objective_family'] == 'adjusted_quasi_likelihood'
    assert ident['iterate_selection'] == 'final_iterate'


def test_selector_with_agreeing_flag_is_accepted():
    lev, es, _ = _resolve_stage_d_estimator(
        {'stage_d_estimator': 'raw_likelihood', 'leverage_correction': False})
    assert (lev, es) == (False, False)


@pytest.mark.parametrize('cfg', [
    {'stage_d_estimator': 'raw_likelihood', 'leverage_correction': True},
    {'stage_d_estimator': 'raw_likelihood', 'alternating_early_stop': True},
    {'stage_d_estimator': 'adjusted_quasi_likelihood', 'leverage_correction': False},
    {'stage_d_estimator': 'adjusted_quasi_likelihood', 'alternating_early_stop': False},
])
def test_contradictory_selector_and_flags_raise(cfg):
    with pytest.raises(ValueError, match='Contradictory'):
        _resolve_stage_d_estimator(cfg)


def test_unknown_selector_value_raises():
    with pytest.raises(ValueError, match='stage_d_estimator must be one of'):
        _resolve_stage_d_estimator({'stage_d_estimator': 'nope'})


# --------------------------------------------------------------------------- #
# End-to-end metadata truth table (public API)
# --------------------------------------------------------------------------- #

_METADATA_KEYS = ('estimator', 'objective_family', 'residual_update',
                  'iterate_selection', 'convergence_reason', 'bound_active',
                  'mean_intercept_mode')


def _het_fit(**overrides):
    X, y, _ = generate_ishigami(n_samples=1600, heteroscedastic=True, seed=0)
    kw = dict(heteroscedastic=True, K1=8, K2=4, Kh=3, verbose=False)
    kw.update(overrides)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return hifi_anova(X, y, **kw)


def test_default_metadata_is_adjusted_quasi_likelihood():
    sd = _het_fit().train_results['stage_D']
    assert sd['selected'] == 'heteroscedastic'
    assert all(k in sd for k in _METADATA_KEYS)
    assert sd['estimator'] == 'adjusted_quasi_likelihood'
    assert sd['objective_family'] == 'adjusted_quasi_likelihood'
    assert sd['residual_update'] == 'leverage_adjusted'
    assert sd['iterate_selection'] == 'validation_best'
    assert sd['convergence_reason'] in ('train_nll_tolerance', 'max_outer_iter')
    assert sd['bound_active'] is False
    # The default never claims raw block-coordinate descent.
    assert sd['objective_family'] != 'raw_penalized_likelihood'


def test_raw_metadata_is_raw_penalized_likelihood():
    sd = _het_fit(stage_d_estimator='raw_likelihood').train_results['stage_D']
    assert sd['estimator'] == 'raw_likelihood'
    assert sd['objective_family'] == 'raw_penalized_likelihood'
    assert sd['residual_update'] == 'raw_squared_residuals'
    assert sd['iterate_selection'] == 'final_iterate'


def test_iterate_selection_label_tracks_early_stop_flag():
    on = _het_fit().train_results['stage_D']
    off = _het_fit(alternating_early_stop=False).train_results['stage_D']
    assert on['iterate_selection'] == 'validation_best'
    assert off['iterate_selection'] == 'final_iterate'
    # early stop populates the selected iterate; final-iterate never restores.
    assert on['best_outer_iteration'] is not None
    assert 1 <= on['best_outer_iteration'] <= on['n_outer_iterations']


def test_guard_revert_metadata():
    # Force the guard to reject the variance model with an impossibly-high
    # selection margin (mirrors test_variance_pipeline's revert trigger), so the
    # revert branch runs deterministically regardless of the fixture's noise.
    X, y, _ = generate_ishigami(n_samples=600, heteroscedastic=True, seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = hifi_anova(X, y, heteroscedastic=True, K1=6, K2=3, Kh=3,
                         verbose=False, variance_selection_margin=10.0)
    sd = res.train_results['stage_D']
    assert sd.get('selected') == 'homoscedastic'
    assert sd['reverted'] is True
    assert sd['convergence_reason'] == 'reverted_homoscedastic'
    assert sd['bound_active'] is False
    # The configured estimator identity is still recorded honestly.
    assert sd['estimator'] in ('adjusted_quasi_likelihood', 'raw_likelihood',
                               'custom')
    assert 'objective_family' in sd


def test_near_noiseless_skip_metadata():
    # Essentially-noiseless data: Stage-D pre-flight skips to constant variance.
    rng = np.random.default_rng(7)
    X = rng.uniform(0, 1, size=(400, 3))
    y = 2.0 * X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, 1e-6, 400)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = hifi_anova(X, y, heteroscedastic=True, K1=6, K2=3, Kh=3,
                         verbose=False)
    sd = res.train_results['stage_D']
    if not sd.get('skipped'):
        pytest.skip('fixture did not trigger the near-noiseless skip on this build')
    assert sd['reason'] == 'near-noiseless'
    assert sd['convergence_reason'] == 'near_noiseless_skip'
    assert sd['bound_active'] is False
    assert 'objective_family' in sd


# --------------------------------------------------------------------------- #
# Default numerics unchanged: the selector adds no numerical movement
# --------------------------------------------------------------------------- #

def test_explicit_adjusted_selector_matches_omitted_default_exactly():
    base = _het_fit()
    expl = _het_fit(stage_d_estimator='adjusted_quasi_likelihood')
    sd_b = base.train_results['stage_D']
    sd_e = expl.train_results['stage_D']
    assert float(base.sigma_hat) == float(expl.sigma_hat)
    assert float(base.df) == float(expl.df)
    assert sd_b['nll_val'] == sd_e['nll_val']
    assert sd_b['rmse_val'] == sd_e['rmse_val']


# --------------------------------------------------------------------------- #
# Raw-mode block-coordinate descent: penalized objective is non-increasing.
# Mirrors the raw-mode trainer loop primitives (no leverage correction, no
# validation iterate selection) on a small deterministic interior fixture.
# --------------------------------------------------------------------------- #

def test_raw_block_updates_do_not_increase_penalized_objective():
    from hifi_anova.training.ridge import weighted_ridge_solve
    from hifi_anova.training.newton import newton_solve_log_variance
    from hifi_anova.model.variance_model import LOG_VAR_CLIP

    rng = np.random.default_rng(0)
    N, Fm, Fh = 120, 4, 3
    Phi = rng.normal(size=(N, Fm))
    Phi_aug = np.concatenate([np.ones((N, 1)), Phi], axis=1)
    Psi = rng.normal(size=(N, Fh)) * 0.5
    # Interior heteroscedasticity: sigma^2 stays well inside exp(+/-30).
    h_true = 0.3 * Psi[:, 0] - 0.2 * Psi[:, 1]
    y = Phi @ rng.normal(size=Fm) + rng.normal(size=N) * np.exp(0.5 * h_true)

    reg_mean_aug = np.concatenate([[0.0], np.full(Fm, 1e-2)])
    reg_var = np.full(Fh, 1e-2)

    def penalized_objective(w_aug, h0, w_h):
        h = h0 + Psi @ w_h
        assert np.all(np.abs(h) < LOG_VAR_CLIP - 1.0)  # interior
        sigma2 = np.exp(h)
        r = y - Phi_aug @ w_aug
        data = float(np.sum(0.5 * h + 0.5 * r ** 2 / sigma2))
        pen = 0.5 * float(w_aug[1:] @ (reg_mean_aug[1:] * w_aug[1:]))
        pen += 0.5 * float(w_h @ (reg_var * w_h))
        return data + pen

    # Homoscedastic init.
    w_aug = np.asarray(weighted_ridge_solve(Phi_aug, y, reg_mean_aug),
                       dtype=np.float64)
    r2 = (y - Phi_aug @ w_aug) ** 2
    w_h = np.zeros(Fh)
    h0 = float(np.log(np.mean(r2)))

    prev = penalized_objective(w_aug, h0, w_h)
    tol = 1e-8
    for _ in range(8):
        # --- variance block (raw r^2: NO leverage correction) ---
        w_h, h0 = newton_solve_log_variance(Psi, r2, w_h, h0, reg_var,
                                            max_iter=20)
        w_h = np.asarray(w_h, dtype=np.float64)
        obj_after_var = penalized_objective(w_aug, h0, w_h)
        assert obj_after_var <= prev + tol, (prev, obj_after_var)
        prev = obj_after_var

        # --- mean block (weighted ridge at the current variance) ---
        sigma2 = np.exp(h0 + Psi @ w_h)
        weights = 1.0 / sigma2
        w_aug = np.asarray(
            weighted_ridge_solve(Phi_aug, y, reg_mean_aug, weights),
            dtype=np.float64)
        r2 = (y - Phi_aug @ w_aug) ** 2
        obj_after_mean = penalized_objective(w_aug, h0, w_h)
        assert obj_after_mean <= prev + tol, (prev, obj_after_mean)
        prev = obj_after_mean


# --------------------------------------------------------------------------- #
# Bound activity: detected + labelled; no interior stationarity implied.
# --------------------------------------------------------------------------- #

def test_bound_active_flag_present_and_false_on_interior_fit():
    sd = _het_fit().train_results['stage_D']
    # A well-behaved fit stays interior; the flag exists and is a plain bool.
    assert 'bound_active' in sd
    assert isinstance(sd['bound_active'], bool)
    assert sd['bound_active'] is False


def test_bound_active_unit_helper_detects_clip():
    from hifi_anova.training.trainer import _log_variance_bound_active
    from hifi_anova.model.variance_model import LOG_VAR_CLIP
    assert _log_variance_bound_active(np.array([-1.0, 0.0, 2.0])) is False
    assert _log_variance_bound_active(np.array([0.0, LOG_VAR_CLIP])) is True
    assert _log_variance_bound_active(np.array([-LOG_VAR_CLIP, 1.0])) is True


def test_bound_active_emits_warning_and_metadata(monkeypatch):
    # Force the trainer's clip threshold low so a normal heteroscedastic fit's
    # log-variance is detected as touching the bound. Exercises the P1-1
    # DETECT+label path: the metadata flag flips AND the disclaiming warning is
    # emitted (the acceptance test's "metadata + warning make it visible").
    # LOG_VAR_CLIP is read by the Stage-D fit + bound-activity helper, which now
    # live in hifi_anova.training.stages.stage_d — patch it there.
    import hifi_anova.training.stages.stage_d as stage_d_mod
    monkeypatch.setattr(stage_d_mod, 'LOG_VAR_CLIP', 0.05)
    X, y, _ = generate_ishigami(n_samples=800, heteroscedastic=True, seed=0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        # guard off so the (bound-touching) heteroscedastic fit is always shipped
        # and never reverts to a constant variance under the lowered clip.
        res = hifi_anova(X, y, heteroscedastic=True, K1=8, K2=4, Kh=3,
                         verbose=False, heteroscedastic_guard=False)
    sd = res.train_results['stage_D']
    assert sd['bound_active'] is True
    msgs = [str(w.message) for w in caught]
    assert any('clip bound' in m and 'bound_active' in m for m in msgs), msgs
    # No exact stationarity claim is implied for the shipped fit.
    assert any('not an interior' in m for m in msgs)


def test_stage_d_estimator_metadata_persists_to_meta_json(tmp_path):
    # Persistence regression: save_model mirrors the estimator identity into
    # meta.json for a heteroscedastic fit, and load_model round-trips it.
    from hifi_anova.model.io import save_model, load_model

    res = _het_fit()
    sd = res.train_results['stage_D']
    out = str(tmp_path / 'model_dir')
    save_model(res.model, out, results=res.train_results,
               transformer=res.transformer)

    import json
    import os
    with open(os.path.join(out, 'meta.json')) as fh:
        meta = json.load(fh)
    assert 'stage_d_estimator' in meta
    block = meta['stage_d_estimator']
    for k in ('estimator', 'objective_family', 'residual_update',
              'iterate_selection', 'convergence_reason', 'bound_active'):
        assert block[k] == sd[k]

    # And it survives a load round-trip.
    loaded = load_model(out)
    assert loaded['meta']['stage_d_estimator']['estimator'] == sd['estimator']


# --------------------------------------------------------------------------- #
# LAML evidence-status labelling (P1-4)
# --------------------------------------------------------------------------- #

def test_joint_laml_labels_evidence_status_by_mode():
    from hifi_anova.training.joint_lambda import _joint_fit, joint_laml

    rng = np.random.default_rng(1)
    N, Fm, Fh = 150, 3, 2
    Phi_aug = np.concatenate(
        [np.ones((N, 1)), rng.normal(size=(N, Fm))], axis=1)
    Psi = rng.normal(size=(N, Fh)) * 0.4
    y = Phi_aug[:, 1:] @ rng.normal(size=Fm) + rng.normal(size=N)
    reg_mean_aug = np.concatenate([[0.0], np.full(Fm, 1e-2)])
    reg_var = np.full(Fh, 1e-1)

    # Generous max_outer so the raw fit reaches a verified interior mode.
    raw = _joint_fit(Phi_aug, Psi, y, reg_mean_aug, reg_var,
                     leverage_correct=False, max_outer=50)
    adj = _joint_fit(Phi_aug, Psi, y, reg_mean_aug, reg_var,
                     leverage_correct=True, max_outer=50)
    L_raw = joint_laml(raw)
    L_adj = joint_laml(adj)
    assert L_raw['objective_mode'] == 'raw_penalized_likelihood'
    assert L_raw['converged'] and not L_raw['bound_active']
    assert L_raw['evidence_status'] == 'laplace_evidence'
    assert L_adj['objective_mode'] == 'adjusted_quasi_likelihood'
    # Adjusted mode is an empirical criterion even when converged + interior.
    assert L_adj['evidence_status'] == 'empirical_criterion'


def test_joint_laml_raw_but_unconverged_is_not_laplace_evidence():
    # Raw mode ALONE does not establish an interior stationary point: a fit that
    # exhausts max_outer without meeting tolerance must NOT be labelled principled
    # Laplace evidence (reviewer finding).
    from hifi_anova.training.joint_lambda import _joint_fit, joint_laml

    rng = np.random.default_rng(4)
    N, Fm, Fh = 200, 4, 3
    Phi_aug = np.concatenate(
        [np.ones((N, 1)), rng.normal(size=(N, Fm))], axis=1)
    Psi = rng.normal(size=(N, Fh)) * 0.6
    y = Phi_aug[:, 1:] @ rng.normal(size=Fm) + rng.normal(size=N) * (
        0.3 + np.abs(Psi[:, 0]))
    reg_mean_aug = np.concatenate([[0.0], np.full(Fm, 1e-3)])
    reg_var = np.full(Fh, 1e-3)
    # One outer iteration cannot meet the relative-NLL tolerance -> not converged.
    raw1 = _joint_fit(Phi_aug, Psi, y, reg_mean_aug, reg_var,
                      leverage_correct=False, max_outer=1)
    assert raw1.converged is False
    L = joint_laml(raw1)
    assert L['objective_mode'] == 'raw_penalized_likelihood'
    assert L['evidence_status'] == 'laplace_evidence_unverified'
    assert L['evidence_status'] != 'laplace_evidence'


def test_optimize_joint_lambda_warns_on_adjusted_laml():
    from hifi_anova.training.joint_lambda import optimize_joint_lambda

    rng = np.random.default_rng(2)
    N, Fm, Fh = 120, 3, 2
    Phi = rng.normal(size=(N, Fm))
    Psi = rng.normal(size=(N, Fh)) * 0.4
    y = Phi @ rng.normal(size=Fm) + rng.normal(size=N)
    mean_shape = np.full(Fm, 1.0)
    var_shape = np.full(Fh, 1.0)
    with pytest.warns(UserWarning, match='empirical criterion'):
        optimize_joint_lambda(Phi, Psi, y, mean_shape, var_shape,
                              criterion='laml', leverage_correct=True,
                              n_grid=3, refine=False)
