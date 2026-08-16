"""Session-04 public naming aliases: canonical storage, identical legacy reads."""

import json

import matplotlib
import numpy as np
import pytest

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from hifi_anova._result_aliases import canonical_result_mapping
from hifi_anova.analysis.automl import ridge_analytics


def test_tr_h2_is_canonical_and_legacy_alias_is_identical():
    rng = np.random.RandomState(4)
    phi = rng.normal(size=(30, 4))
    y = rng.normal(size=30)
    analytics = ridge_analytics(phi, y, np.full(4, 0.2))

    assert 'tr_H2' in dict(analytics)
    assert 'tr_HHt' not in dict(analytics)
    with pytest.warns(DeprecationWarning, match='tr_HHt'):
        legacy = analytics['tr_HHt']
    assert legacy == analytics['tr_H2']
    assert analytics['df_residual'] == max(
        30 - 2 * analytics['df'] + analytics['tr_H2'], 1.0)


def test_log_variance_key_alias_and_json_are_canonical():
    block = {'first_order': {0: 0.75, 1: 0.25}}
    result = canonical_result_mapping({'variance_sobol': block})

    assert list(result) == ['log_variance_sobol']
    with pytest.warns(DeprecationWarning, match='variance_sobol'):
        legacy = result['variance_sobol']
    assert legacy is result['log_variance_sobol']
    payload = json.dumps(result)
    assert 'log_variance_sobol' in payload
    assert '"variance_sobol"' not in payload


def test_legacy_persisted_names_migrate_recursively_without_value_change():
    legacy_payload = {
        'sobol': {'variance_sobol': {'first_order': {'0': 1.0}}},
        'analytics': {'tr_HHt': 2.5},
    }
    loaded = canonical_result_mapping(json.loads(json.dumps(legacy_payload)))

    assert loaded['sobol']['log_variance_sobol']['first_order']['0'] == 1.0
    assert loaded['analytics']['tr_H2'] == 2.5
    with pytest.warns(DeprecationWarning, match='variance_sobol'):
        assert loaded['sobol']['variance_sobol'] is loaded['sobol']['log_variance_sobol']
    with pytest.warns(DeprecationWarning, match='tr_HHt'):
        assert loaded['analytics']['tr_HHt'] == loaded['analytics']['tr_H2']


def test_alias_aware_copy_and_union_preserve_legacy_reads():
    block = {'first_order': {0: 1.0}}
    original = canonical_result_mapping({'log_variance_sobol': block})

    for derived in (original.copy(), original | {'extra': 1},
                    {'extra': 1} | original):
        with pytest.warns(DeprecationWarning, match='variance_sobol'):
            assert derived['variance_sobol'] is derived['log_variance_sobol']

    replacement = {'first_order': {0: 0.5}}
    merged = original | {'variance_sobol': replacement}
    assert merged['log_variance_sobol'] is replacement


def _legacy_plot_mapping():
    return {
        'mean_sobol': {
            'first_order': {0: 0.6, 1: 0.4},
            'total_order': {0: 0.6, 1: 0.4},
        },
        'variance_sobol': {
            'first_order': {0: 0.25, 1: 0.75},
            'total_order': {0: 0.25, 1: 0.75},
        },
    }


def test_plain_legacy_mapping_reaches_both_dual_bar_plot_implementations():
    from hifi_anova.analysis.plots import plot_dual_sobol as publication_plot
    from hifi_anova.analysis.visualization import plot_dual_sobol as simple_plot

    with pytest.warns(DeprecationWarning, match='variance_sobol'):
        simple_fig = simple_plot(_legacy_plot_mapping())
    simple_heights = [patch.get_height() for patch in simple_fig.axes[0].patches]
    assert simple_heights[-2:] == [0.25, 0.75]

    with pytest.warns(DeprecationWarning, match='variance_sobol'):
        publication_fig, publication_ax = publication_plot(_legacy_plot_mapping())
    publication_heights = [patch.get_height()
                           for patch in publication_ax.patches]
    assert sorted(publication_heights[-2:]) == [0.25, 0.75]
    assert 'Log-variance' in publication_ax.get_legend_handles_labels()[1][1]
    plt.close(simple_fig)
    plt.close(publication_fig)


def test_plain_legacy_mapping_reaches_both_ellipse_plot_implementations():
    from hifi_anova.analysis.plots import (
        plot_sensitivity_ellipses as publication_plot)
    from hifi_anova.analysis.visualization import (
        plot_sensitivity_ellipses as simple_plot)

    with pytest.warns(DeprecationWarning, match='variance_sobol'):
        simple_fig = simple_plot(_legacy_plot_mapping(), mode='plane')
    simple_y = sorted(patch.center[1] for patch in simple_fig.axes[0].patches)
    assert simple_y == [0.25, 0.75]
    assert 'Log-variance' in simple_fig.axes[0].get_ylabel()

    with pytest.warns(DeprecationWarning, match='variance_sobol'):
        publication_fig, publication_ax = publication_plot(
            _legacy_plot_mapping(), mode='plane')
    publication_y = sorted(patch.center[1] for patch in publication_ax.patches)
    assert publication_y == [0.25, 0.75]
    assert 'Log-variance' in publication_ax.get_ylabel()
    plt.close(simple_fig)
    plt.close(publication_fig)
