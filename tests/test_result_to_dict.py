"""HiFiResult.to_dict(): JSON-serializable reporting payload (GUI/back-end)."""

import json
import numpy as np

from hifi_anova import hifi_anova


def _toy(n=400, d=3, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, (n, d))
    y = np.sin(2 * np.pi * X[:, 0]) + 0.5 * (X[:, 1] - 0.5) + 0.1 * rng.standard_normal(n)
    return X, y


def test_to_dict_is_json_serializable():
    X, y = _toy()
    res = hifi_anova(X, y, K1=4, K2=2, mode='second', verbose=False)
    d = res.to_dict()
    # Round-trips through json without raising.
    s = json.dumps(d)
    back = json.loads(s)
    assert isinstance(back, dict)


def test_to_dict_excludes_handles_includes_reporting():
    X, y = _toy()
    res = hifi_anova(X, y, K1=4, K2=2, mode='second', verbose=False)
    d = res.to_dict()
    # Non-serializable handles and internals are excluded.
    for k in ('model', 'transformer', '_Phi_train', '_reg_diag', '_data',
              '_fitted_design'):
        assert k not in d
    # Reporting surface is present.
    for k in ('sobol_ci', 'sigma_hat', 'r_squared', 'loo_cv', 'df',
              'feature_names', 'config', 'inference_metadata'):
        assert k in d
    # fitted_design record is dropped from the nested train_results.
    assert 'fitted_design' not in d.get('train_results', {})


def test_to_dict_stringifies_tuple_sobol_keys():
    X, y = _toy()
    # K2>0 so second-order interaction blocks (tuple keys) exist in the spectrum.
    res = hifi_anova(X, y, K1=4, K2=3, mode='second', verbose=False)
    d = res.to_dict()
    second = d['sobol']['mean_sobol'].get('second_order', {})
    # All keys are strings (json requires it); tuple (i,j) -> "i,j".
    assert all(isinstance(k, str) for k in second)


def test_to_dict_values_are_plain_python():
    X, y = _toy()
    res = hifi_anova(X, y, K1=4, K2=2, mode='second', verbose=False)
    d = res.to_dict()
    # A CI tuple becomes a list of plain floats.
    fo = d['sobol_ci']['first_order'] if 'first_order' in d['sobol_ci'] else d['sobol_ci']
    some = next(iter(fo.values()))
    assert isinstance(some, list)
    assert all(isinstance(v, float) for v in some)
