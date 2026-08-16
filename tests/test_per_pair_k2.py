"""Per-pair second-order harmonic order (X11C-S02 / GUI3 request BR-04).

``K2`` may be a mapping ``{(i, j): K2_ij}`` that pins the exact retained pairs
AND gives each its own harmonic order. These tests cover:

* the ragged pair-block layout end-to-end (features → penalty → model →
  Sobol → CI record → prediction), homoscedastic and Stage-D;
* exactness against a scalar-K2 fit when every pair shares one order — the
  ragged path must reproduce the uniform path's model on the same pair set;
* the guards (canonical pair keys, no selection/pruning heuristics, no K3/
  mixed/auto composition) and persistence (pickle fallback + field survival).

The scalar-K2 default path is untouched by construction (all new branches key
on ``pair_k2 is not None``); the golden characterization pins that separately.
"""

import numpy as np
import pytest

from hifi_anova.api import hifi_anova
from hifi_anova.core.features import (
    basis_size, build_second_order_features,
    build_second_order_features_per_pair,
)

pytestmark = pytest.mark.integration


def _data(N=400, D=4, seed=0, noise=0.1):
    rng = np.random.default_rng(seed)
    X = rng.uniform(size=(N, D))
    y = (np.sin(2 * np.pi * X[:, 0]) + X[:, 1] ** 2
         + 1.5 * np.sin(2 * np.pi * X[:, 0]) * np.cos(2 * np.pi * X[:, 1])
         + 0.8 * X[:, 2] * X[:, 3]
         + noise * rng.standard_normal(N))
    return X, y


# ---------------------------------------------------------------- features

def test_per_pair_builder_matches_uniform_when_orders_equal():
    X, _ = _data(N=50)
    pairs = np.array([[0, 1], [2, 3]], dtype=np.int32)
    phi_u = np.asarray(build_second_order_features(X, 3, pairs))
    phi_r, info = build_second_order_features_per_pair(X, [3, 3], pairs)
    np.testing.assert_allclose(np.asarray(phi_r), phi_u, rtol=0, atol=0)
    b = basis_size(3, True, 'fourier')
    assert info == ((0, 1, b, b, b * b, 0), (2, 3, b, b, b * b, b * b))


def test_per_pair_builder_ragged_layout():
    X, _ = _data(N=30)
    pairs = np.array([[0, 1], [1, 2]], dtype=np.int32)
    phi, info = build_second_order_features_per_pair(X, [4, 2], pairs)
    b4 = basis_size(4, True, 'fourier')
    b2 = basis_size(2, True, 'fourier')
    assert phi.shape[1] == b4 * b4 + b2 * b2
    # each block equals the uniform builder restricted to that pair
    phi_p0 = np.asarray(build_second_order_features(
        X, 4, np.array([[0, 1]], dtype=np.int32)))
    phi_p1 = np.asarray(build_second_order_features(
        X, 2, np.array([[1, 2]], dtype=np.int32)))
    np.testing.assert_allclose(np.asarray(phi)[:, :b4 * b4], phi_p0)
    np.testing.assert_allclose(np.asarray(phi)[:, b4 * b4:], phi_p1)


# ---------------------------------------------------------------- one-call fit

def test_per_pair_k2_fit_end_to_end():
    X, y = _data()
    res = hifi_anova(X, y, K1=4, K2={(0, 1): 4, (2, 3): 2}, verbose=False)
    m = res.model
    b41 = basis_size(4, True, 'fourier') ** 2
    b22 = basis_size(2, True, 'fourier') ** 2

    assert m.pair_k2 == (4, 2)
    assert np.asarray(m.pair_indices).tolist() == [[0, 1], [2, 3]]
    assert len(m.mean_model.w2) == b41 + b22
    assert m.mean_model.get_coefficients_for_pair(0).shape == (b41,)
    assert m.mean_model.get_coefficients_for_pair(1).shape == (b22,)
    assert m.mean_model.get_pair_gram(0).shape == (b41, b41)
    assert m.mean_model.get_pair_gram(1).shape == (b22, b22)
    assert m.G2 is None  # no single shared pair Gram in per-pair mode

    # Sobol spectrum keyed by the user's pairs; the strong (0,1) term found
    so = res.sobol['mean_sobol']['second_order']
    assert set(so) == {(0, 1), (2, 3)}
    assert so[(0, 1)] > 0.1

    # record uses the explicit per-group layout; diagnostics/CI finite
    rec = res._fitted_design
    assert rec.sobol_groups is not None
    orders = [g[0] for g in rec.sobol_groups]
    assert orders.count(1) == 4 and orders.count(2) == 2
    assert np.isfinite(res.df) and np.isfinite(res.loo_cv)
    for name, (S, lo, hi) in res.sobol_ci.items():
        assert lo <= S <= hi

    # prediction paths (mean, intervals) run on the ragged layout
    pred = res.predict(X[:16])
    assert np.all(np.isfinite(pred))
    lo, hi = res.predict_intervals(X[:5])
    assert np.all(np.asarray(hi) >= np.asarray(lo))


def test_uniform_orders_reproduce_uniform_pair_blocks():
    """A mapping with one shared order builds the same design blocks the
    uniform builder would on that pair set, and the model's build_phi2 matches
    the training-side ragged builder exactly."""
    X, y = _data()
    res = hifi_anova(X, y, K1=3, K2={(0, 1): 3, (2, 3): 3}, verbose=False)
    m = res.model
    b = basis_size(3, True, 'fourier') ** 2
    xg = np.asarray(res._data['x_test'])[:32]
    phi2_model = np.asarray(m.build_phi2(xg))
    assert phi2_model.shape[1] == 2 * b
    phi2_uniform = np.asarray(build_second_order_features(
        xg, 3, np.asarray(m.pair_indices)))
    np.testing.assert_allclose(phi2_model, phi2_uniform, rtol=0, atol=0)
    assert np.all(np.isfinite(m.predict_mean_only(xg)))


def test_per_pair_k2_heteroscedastic_stage_d():
    rng = np.random.default_rng(1)
    N, D = 500, 3
    X = rng.uniform(size=(N, D))
    sigma = 0.05 + 0.4 * X[:, 2]
    y = (np.sin(2 * np.pi * X[:, 0])
         + 2.0 * np.sin(2 * np.pi * X[:, 0]) * np.cos(2 * np.pi * X[:, 1])
         + sigma * rng.standard_normal(N))
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = hifi_anova(X, y, K1=4, K2={(0, 1): 4}, heteroscedastic=True,
                         mode='heteroscedastic', verbose=False)
    assert res.model.pair_k2 == (4,)
    assert (0, 1) in res.sobol['mean_sobol']['second_order']
    assert np.all(np.asarray(res.sigma_x2(X[:10])) > 0)
    assert np.all(np.isfinite(res.predict(X[:10])))
    rec = res._fitted_design
    assert rec.sobol_groups is not None
    # weighted + interpretive companions both carry the ragged layout
    if rec.interpretive is not None:
        assert rec.interpretive.sobol_groups is not None


# ---------------------------------------------------------------- guards

def test_k2_mapping_guards():
    X, y = _data(N=120)
    with pytest.raises(ValueError, match="canonical order"):
        hifi_anova(X, y, K2={(1, 0): 3}, verbose=False)
    with pytest.raises(ValueError, match="out of range"):
        hifi_anova(X, y, K2={(0, 9): 3}, verbose=False)
    with pytest.raises(ValueError, match="pair_pruning"):
        hifi_anova(X, y, K2={(0, 1): 3}, pair_pruning='bic', verbose=False)
    with pytest.raises(ValueError, match="pair_selection"):
        hifi_anova(X, y, K2={(0, 1): 3}, pair_selection=[0, 1], verbose=False)
    # explicit variable_selection with a K2 mapping: the one-call API also
    # sets pair_candidates, so the trainer rejects the combination on
    # whichever incompatible key it sees first.
    with pytest.raises(ValueError, match="per-pair mapping"):
        hifi_anova(X, y, K2={(0, 1): 3}, variable_selection='bic',
                   verbose=False)
    with pytest.raises(ValueError, match="third-order"):
        hifi_anova(X, y, K2={(0, 1): 3}, K3=1, verbose=False)
    with pytest.raises(ValueError, match="at least one pair"):
        hifi_anova(X, y, K2={}, verbose=False)


def test_k2_mapping_records_term_structure_provenance():
    X, y = _data(N=200)
    res = hifi_anova(X, y, K1=3, K2={(0, 1): 3}, verbose=False)
    ts = res.train_results.get('term_structure')
    assert ts is not None
    assert ts['pair_k2'] == {'0,1': 3}


# ---------------------------------------------------------------- persistence

def test_per_pair_k2_save_load_roundtrip(tmp_path):
    from hifi_anova.model.io import save_model, load_model
    X, y = _data(N=250)
    res = hifi_anova(X, y, K1=3, K2={(0, 1): 2}, verbose=False)
    path = str(tmp_path / 'm')
    save_model(res.model, path, config=res.config)
    loaded = load_model(path)['model']
    assert getattr(loaded, 'pair_k2', None) == (2,)
    xg = np.asarray(res._data['x_test'])[:16]
    np.testing.assert_allclose(
        np.asarray(res.model.predict_mean_only(xg)),
        np.asarray(loaded.predict_mean_only(xg)))


def test_k2_mapping_config_roundtrips_json_safe(tmp_path):
    """A per-pair ``K2`` mapping is persisted in ``meta.json['config']`` in the
    JSON-safe ``"i,j"`` form (matching results['term_structure']['pair_k2']),
    not stringified whole, so the retained-pair config round-trips faithfully
    and the original tuple-keyed mapping is reconstructable (P1-8)."""
    from hifi_anova.model.io import save_model, load_model
    X, y = _data(N=250)
    res = hifi_anova(X, y, K1=3, K2={(0, 1): 4, (2, 3): 2}, verbose=False)
    path = str(tmp_path / 'm')
    save_model(res.model, path, config=res.config)

    cfg_k2 = load_model(path)['config']['K2']
    # faithful, JSON-native mapping (not an opaque "{(0, 1): 4, ...}" string)
    assert cfg_k2 == {'0,1': 4, '2,3': 2}
    # the exact tuple-keyed mapping is recoverable from the stored form
    recovered = {tuple(int(i) for i in k.split(',')): int(v)
                 for k, v in cfg_k2.items()}
    assert recovered == {(0, 1): 4, (2, 3): 2}


# ---------------------------------------------------------------- plotting

def test_interaction_plots_ragged_pair_blocks():
    """Interaction heatmap / grid must size each pair's block from ITS order,
    not the uniform ``model.K2`` (which holds ``max(pair_k2)``). A pair with a
    smaller order has fewer coefficients than the max block, so the uniform
    ``wp.reshape(B, B)`` used to raise on the ragged layout (heatmap pair 1:
    reshape (25,) into (9, 9); grid: same)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from hifi_anova.analysis.visualization import plot_interaction_heatmap
    from hifi_anova.analysis.plots import plot_interaction_grid

    X, y = _data()
    res = hifi_anova(X, y, K1=4, K2={(0, 1): 4, (2, 3): 2}, verbose=False)
    m = res.model
    assert m.pair_k2 == (4, 2)  # ragged: orders differ
    # every pair, including the smaller-order one, renders without a reshape error
    for p in range(len(m.pair_k2)):
        fig = plot_interaction_heatmap(m, p)
        assert fig is not None
        plt.close(fig)
    fig, axes = plot_interaction_grid(m, sobol_results=res.sobol, top_k=2)
    assert fig is not None and axes.size >= 2
    plt.close(fig)


def test_sobol_dict_plots_smoke_on_term_structure():
    """Sobol-dict plot gallery smoke on a per-pair-K2 fit — these consume the
    Sobol dict (pairs keyed by the user's retained set), not the ragged
    coefficient layout, but pin that they render end-to-end."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from hifi_anova.analysis.plots import (
        plot_sobol_waterfall, plot_order_decomposition,
        plot_sensitivity_ellipses, plot_variance_treemap,
        plot_variance_sunburst)

    X, y = _data()
    res = hifi_anova(X, y, K1=4, K2={(0, 1): 4, (2, 3): 2}, verbose=False)
    sob = res.sobol
    for fn in (plot_sobol_waterfall, plot_order_decomposition,
               plot_sensitivity_ellipses, plot_variance_treemap,
               plot_variance_sunburst):
        out = fn(sob)
        fig = out[0] if isinstance(out, tuple) else out
        assert fig is not None
        plt.close('all')
