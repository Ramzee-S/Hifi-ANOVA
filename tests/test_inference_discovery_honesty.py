"""X6 Session-03 regressions: conditional inference and sieve vocabulary."""

import json
import pickle
from dataclasses import asdict

import jax.numpy as jnp
import numpy as np
import pytest

from hifi_anova.analysis.automl import sobol_confidence_intervals
from hifi_anova.analysis.interaction_discovery import (
    MissingPairResult,
    scan_missing_pairs,
    scan_missing_triples,
)
from hifi_anova.api import (
    _fixed_configuration_inference_metadata,
    hifi_anova,
)
from hifi_anova.model.io import load_model


class _ScanModel:
    D = 2
    K2 = 1
    pair_indices = None
    triple_indices = None
    basis_name = 'fourier'
    include_linear_3 = True

    def predict(self, x):
        return jnp.zeros(x.shape[0]), jnp.ones(x.shape[0])


def test_discovery_canonical_names_and_deprecated_aliases(capsys):
    x = jnp.asarray(np.linspace(0.05, 0.95, 24).reshape(12, 2))
    y = jnp.linspace(-1.0, 1.0, 12)
    result = scan_missing_pairs(
        _ScanModel(), x, y, flag_threshold=0.02, verbose=True)

    assert result.flag_threshold == 0.02
    assert result.n_flagged >= 0
    assert set(asdict(result)) >= {'n_flagged', 'flag_threshold'}
    assert 'n_significant' not in asdict(result)
    out = capsys.readouterr().out
    assert 'flagged' in out
    assert 'not an FDR-controlled test' in out
    assert ' significant' not in out

    with pytest.warns(DeprecationWarning, match='n_significant'):
        assert result.n_significant == result.n_flagged
    with pytest.warns(DeprecationWarning, match='significance_threshold'):
        assert result.significance_threshold == result.flag_threshold
    with pytest.warns(DeprecationWarning, match='significance_threshold'):
        legacy = scan_missing_pairs(
            _ScanModel(), x, y, significance_threshold=0.02, verbose=False)
    assert legacy.n_flagged == result.n_flagged
    assert legacy.flag_threshold == result.flag_threshold


def test_triple_mapping_aliases_warn_and_json_is_canonical():
    x = jnp.asarray(np.linspace(0.05, 0.95, 24).reshape(12, 2))
    y = jnp.linspace(-1.0, 1.0, 12)
    result = scan_missing_triples(
        _ScanModel(), x, y, flag_threshold=0.03, verbose=False)

    assert result['n_flagged'] == 0
    assert result['flag_threshold'] == 0.03
    with pytest.warns(DeprecationWarning, match='n_significant'):
        assert result['n_significant'] == result['n_flagged']
    with pytest.warns(DeprecationWarning, match='n_significant'):
        assert 'n_significant' in result
    with pytest.warns(DeprecationWarning, match='significance_threshold'):
        assert result.get('significance_threshold') == result['flag_threshold']
    saved = json.loads(json.dumps(result))
    assert saved['n_flagged'] == 0
    assert saved['flag_threshold'] == 0.03
    assert 'n_significant' not in saved
    assert 'significance_threshold' not in saved


def test_missing_pair_constructor_and_legacy_pickle_compatibility():
    details = {(0, 1): {'fraction_of_residual': 0.2}}
    # Historical seven-position order ends in pair_details; the new threshold
    # field must not silently consume that seventh argument.
    positional = MissingPairResult(
        {(0, 1): 0.2}, [((0, 1), 0.2)], 1, 1, 2.0, 0.2, details)
    assert positional.n_flagged == 1
    assert positional.pair_details is details
    assert positional.flag_threshold == 0.001

    with pytest.warns(DeprecationWarning, match='n_significant'):
        keyword = MissingPairResult(
            pair_scores={}, ranked_pairs=[], n_scanned=0,
            n_significant=2, total_residual_variance=1.0,
            total_captured_by_missing=0.0)
    assert keyword.n_flagged == 2

    # Simulate the state dict stored by the pre-DEC-050 dataclass. Unpickling
    # bypasses __init__ and exercises the explicit state migration.
    legacy = MissingPairResult.__new__(MissingPairResult)
    legacy.__dict__.update({
        'pair_scores': {}, 'ranked_pairs': [], 'n_scanned': 0,
        'n_significant': 3, 'total_residual_variance': 1.0,
        'total_captured_by_missing': 0.0, 'pair_details': details,
    })
    payload = pickle.dumps(legacy)
    with pytest.warns(DeprecationWarning, match='migrated'):
        restored = pickle.loads(payload)
    assert restored.n_flagged == 3
    assert restored.flag_threshold == 0.001
    assert restored.pair_details == details


def test_zero_component_is_nonregular_without_changing_legacy_tuple():
    # Exactly orthogonal columns make the second fitted block exactly zero while
    # the first remains active. The legacy tuple stays available, but its status
    # prevents it from being presented as an ordinary 95% delta interval.
    Phi = np.asarray([
        [1.0, 1.0],
        [1.0, -1.0],
        [-1.0, 1.0],
        [-1.0, -1.0],
    ])
    y = Phi[:, 0].copy()
    groups = [
        (1, 0, slice(0, 1), np.eye(1)),
        (1, 1, slice(1, 2), np.eye(1)),
    ]
    ci = sobol_confidence_intervals(
        Phi, y, np.zeros(2), D=2, groups=groups)

    assert ci['first_order'][0] == (1.0, 1.0, 1.0)
    assert ci['first_order'][1] == (0.0, 0.0, 0.0)
    assert ci['component_status']['first_order'][0] == 'nonregular_boundary'
    assert ci['component_status']['first_order'][1] == 'nonregular_null'
    assert ci['interval_method'] == 'HC3_delta_t'


def test_one_component_complete_share_is_nonregular_boundary():
    Phi = np.asarray([[1.0], [-1.0], [1.0], [-1.0]])
    ci = sobol_confidence_intervals(
        Phi, Phi[:, 0], np.zeros(1), D=1, K1=0, G1=np.eye(1))
    assert ci['first_order'][0] == (1.0, 1.0, 1.0)
    assert ci['component_status']['first_order'][0] == 'nonregular_boundary'


@pytest.mark.parametrize(
    ('config', 'results', 'expected'),
    [
        ({'variable_selection': None, 'triple_selection': 'all'},
         {'stage_B': {'n_triples': 2}}, False),
        ({'variable_selection': 'bic'}, {}, True),
        ({'pair_selection': [0, 2]}, {}, False),
        ({'K3': 1, 'triple_selection': 'all_active'},
         {'stage_B': {'n_triples': 1}}, True),
        ({'basis_per_variable': 'auto'}, {'mixed_basis': True}, True),
        ({'basis_per_variable': {0: {'basis': 'fourier', 'K': 2}}},
         {'mixed_basis': True}, False),
        ({'heteroscedastic_guard': True},
         {'stage_D': {'nll_homoscedastic': 1.0,
                      'nll_heteroscedastic': 0.9}}, True),
        ({'heteroscedastic_guard': True},
         {'stage_D': {'skipped': True, 'reason': 'near-noiseless'}}, True),
        ({'heteroscedastic_guard': False},
         {'stage_D': {'selected': 'heteroscedastic'}}, False),
        ({}, {'selection_applied': True}, True),
    ],
)
def test_same_data_structure_selection_truth_table(config, results, expected):
    metadata = _fixed_configuration_inference_metadata(config, results)
    assert metadata['structure_selected_on_same_data'] is expected


@pytest.fixture(scope='module')
def inference_result():
    rng = np.random.default_rng(50)
    x = rng.uniform(size=(180, 3))
    y = (2.0 * np.sin(2 * np.pi * x[:, 0])
         + 0.7 * (x[:, 1] - 0.5) + rng.normal(0.0, 0.15, len(x)))
    return hifi_anova(x, y, K1=2, K2=1, verbose=False)


def test_same_data_selection_metadata_and_conditional_summary(
        inference_result, capsys):
    result = inference_result
    assert result.inference_metadata == {
        'inference_scope': 'fixed_configuration',
        'structure_selected_on_same_data': True,
        'post_selection_coverage': 'not_claimed',
        'conditioned_on': [
            'transform', 'basis', 'admitted_structure', 'penalties', 'weights'],
        'interval_method': 'HC3_delta_t',
    }
    result.summary()
    out = capsys.readouterr().out
    assert 'post-selection coverage is not claimed' in out
    assert 'Sobol Indices (conditional intervals):' in out
    assert 'Sobol Indices (95% CI):' not in out


def test_inference_metadata_round_trips_json_safe(inference_result, tmp_path):
    path = tmp_path / 'fit'
    inference_result.save(path)
    loaded = load_model(path)

    assert loaded['results']['inference_metadata'] == (
        inference_result.inference_metadata)
    assert loaded['meta']['inference_metadata'] == inference_result.inference_metadata
    assert loaded['results']['sobol_ci_status'] == (
        inference_result.sobol_ci_status)
    json.dumps(inference_result.inference_metadata)


def test_nonregular_component_not_printed_as_ordinary_ci(
        inference_result, capsys):
    result = inference_result
    name = next(iter(result.sobol_ci))
    old_ci = result.sobol_ci[name]
    old_status = result.sobol_ci_status.get(name)
    try:
        result.sobol_ci[name] = (0.0, 0.0, 0.0)
        result.sobol_ci_status[name] = 'nonregular_null'
        result.summary()
        out = capsys.readouterr().out
        assert f"{name:15s}: 0.0000 [0.0000, 0.0000]" not in out
        assert 'Ordinary delta intervals are suppressed' in out
    finally:
        result.sobol_ci[name] = old_ci
        if old_status is None:
            result.sobol_ci_status.pop(name, None)
        else:
            result.sobol_ci_status[name] = old_status


def test_nonregular_complete_share_not_printed_as_ordinary_ci(
        inference_result, capsys):
    result = inference_result
    name = next(iter(result.sobol_ci))
    old_ci = result.sobol_ci[name]
    old_status = result.sobol_ci_status.get(name)
    try:
        result.sobol_ci[name] = (1.0, 1.0, 1.0)
        result.sobol_ci_status[name] = 'nonregular_boundary'
        result.summary(observed=True)
        out = capsys.readouterr().out
        assert f"{name:15s}: 1.0000 [1.0000, 1.0000]" not in out
        assert 'Nonregular complete-share boundary' in out
        assert 'zero-width interval is suppressed' in out
    finally:
        result.sobol_ci[name] = old_ci
        if old_status is None:
            result.sobol_ci_status.pop(name, None)
        else:
            result.sobol_ci_status[name] = old_status
