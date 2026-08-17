"""Regression coverage for public-API and trainer integration boundaries."""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from hifi_anova.model.hifi_anova import HiFiANOVA
from hifi_anova.model.mean_model import MeanModel
from hifi_anova.model.predict import predict_intervals
from hifi_anova.training.trainer import HiFiANOVATrainer


pytestmark = pytest.mark.smoke


def _mean_only_model(constant_log_var=None):
    """Minimal fitted-shape model for interval tests."""
    mean = MeanModel(
        f0=jnp.array(0.0),
        w1=jnp.zeros(3),                 # D=1, Fourier K1=1 -> 3 columns
        w2=jnp.array([]),
        K1=1, K2=0, D=1,
    )
    return HiFiANOVA(
        mean_model=mean,
        K1=1, K2=0, Kh=0, D=1,
        constant_log_var=constant_log_var,
    )


def test_homoscedastic_intervals_use_fitted_noise_variance():
    """A plain mean model must use sigma2_hat, not its neutral variance=1."""
    model = _mean_only_model()
    result = predict_intervals(
        model, jnp.array([[0.2], [0.8]]),
        sigma2_hat=0.04, include_epistemic=False,
    )
    assert np.allclose(result['var_aleatoric'], 0.04)
    assert np.allclose(result['var_total'], 0.04)


def test_constant_variance_fallback_takes_precedence_over_sigma_hat():
    """A Stage-D constant fallback is already fitted and must be preserved."""
    model = _mean_only_model(constant_log_var=jnp.log(0.25))
    result = predict_intervals(
        model, jnp.array([[0.2], [0.8]]),
        sigma2_hat=0.04, include_epistemic=False,
    )
    assert np.allclose(result['var_aleatoric'], 0.25)


def test_one_call_second_mode_with_k2_zero_disables_stage_b():
    """K2=0 is the documented switch for disabling pair interactions."""
    from hifi_anova.api import hifi_anova

    rng = np.random.default_rng(0)
    X = rng.uniform(0.0, 1.0, size=(180, 3))
    y = 2.0 * X[:, 0] - X[:, 1] + rng.normal(0.0, 0.05, size=len(X))
    result = hifi_anova(
        X, y, K1=2, K2=0, mode='second',
        variable_selection=None, verbose=False,
    )

    assert result.model.K2 == 0
    assert result.model.pair_indices is None
    assert 'stage_B' not in result.train_results
    assert np.all(np.isfinite(result.predict(X[:5])))


def test_one_call_rejects_unknown_mode():
    """A misspelled mode must fail instead of silently becoming 'second'."""
    from hifi_anova.api import hifi_anova

    X = np.zeros((20, 2))
    y = np.zeros(20)
    with pytest.raises(ValueError, match="Unknown mode 'secodn'"):
        hifi_anova(X, y, mode='secodn', verbose=False)


def test_one_call_supports_mixed_basis():
    """basis_per_variable now drives the one-call analytics (DEC-030).

    A mixed per-variable basis fits in the trainer AND the post-fit diagnostics
    are record-driven: the fitted-design record carries per-group column slices
    and Grams, so the Sobol CIs are block-correct rather than crashing on the old
    uniform rebuild. This asserts the one-call returns a well-formed result (the
    heavier block-correctness check lives in test_mixed_basis_onecall.py).
    """
    from hifi_anova.api import hifi_anova

    rng = np.random.default_rng(2)
    X = rng.uniform(0.0, 1.0, size=(120, 2))
    y = X[:, 0] + rng.normal(0.0, 0.05, size=len(X))
    res = hifi_anova(
        X, y, mode='second', verbose=False,
        basis_per_variable={0: {'basis': 'fourier', 'K': 3},
                            1: {'basis': 'legendre', 'K': 3}},
    )
    # Block-correct first-order CIs for both variables, each bracketing its point.
    assert len(res.sobol_ci) == 2
    for _name, (S, lo, hi) in res.sobol_ci.items():
        assert 0.0 <= lo <= S <= hi <= 1.0
    # Homoscedastic ⇒ no efficient set / gap.
    assert res.sobol_ci_efficient is None
    assert res.sobol_gap is None


def test_verbose_false_silences_trainer(capsys):
    """verbose=False must suppress the trainer's stage prints, not just summary."""
    from hifi_anova.api import hifi_anova

    rng = np.random.default_rng(3)
    X = rng.uniform(0.0, 1.0, size=(120, 2))
    y = X[:, 0] + rng.normal(0.0, 0.05, size=len(X))

    hifi_anova(X, y, K1=2, K2=1, mode='second',
               variable_selection=None, verbose=False)
    out = capsys.readouterr().out
    assert out == "", f"verbose=False still printed:\n{out}"

    hifi_anova(X, y, K1=2, K2=1, mode='second',
               variable_selection=None, verbose=True)
    out = capsys.readouterr().out
    assert "Stage A" in out


@pytest.mark.parametrize('bad', ['length', 'inf', 'ndim', 'constant'])
def test_preprocess_rejects_invalid_inputs(bad):
    """Input validation must fail early with a clear ValueError."""
    from hifi_anova.data.preprocessing import preprocess_data

    rng = np.random.default_rng(4)
    X = rng.uniform(0.0, 1.0, size=(50, 2))
    y = X[:, 0] + rng.normal(0.0, 0.05, size=len(X))

    if bad == 'length':
        with pytest.raises(ValueError, match="length mismatch"):
            preprocess_data(X, y[:-1])
    elif bad == 'inf':
        Xb = X.copy(); Xb[0, 0] = np.inf
        with pytest.raises(ValueError, match="non-finite"):
            preprocess_data(Xb, y)
    elif bad == 'ndim':
        with pytest.raises(ValueError, match="2-D"):
            preprocess_data(X[:, 0], y)
    elif bad == 'constant':
        with pytest.raises(ValueError, match="constant"):
            preprocess_data(X, np.ones(len(X)))


def test_package_version_is_single_sourced():
    """pyproject must not carry a version literal that can drift from __init__."""
    try:
        import tomllib  # Python 3.11+ stdlib
    except ModuleNotFoundError:  # Python 3.10 (a supported version) has no tomllib
        tomllib = pytest.importorskip(
            "tomli", reason="need tomllib (py311+) or tomli to parse pyproject")
    from pathlib import Path
    import hifi_anova

    pyproject = Path(hifi_anova.__file__).resolve().parent.parent / 'pyproject.toml'
    if not pyproject.exists():
        pytest.skip("pyproject.toml not alongside the package (installed dist)")
    meta = tomllib.loads(pyproject.read_text())
    project = meta['project']
    assert 'version' not in project, (
        "pyproject [project].version is hardcoded; it can drift from "
        "hifi_anova.__version__. Use dynamic = ['version'].")
    assert 'version' in project.get('dynamic', [])
    assert meta['tool']['setuptools']['dynamic']['version']['attr'] == \
        'hifi_anova.__version__'


@pytest.mark.parametrize(('basis_name', 'K1'), [
    ('legendre', 2),
    ('haar', 2),
])
def test_trainer_selection_uses_actual_non_fourier_block_layout(
        basis_name, K1):
    """BIC must evaluate the third variable's real Legendre/Haar block.

    The former trainer call omitted basis metadata, so selection assumed a
    Fourier block of ``2*K1+1`` columns. For these designs that made the third
    group an empty slice with delta-BIC zero even though x3 carries the signal.
    """
    rng = np.random.default_rng(1)
    x = rng.uniform(0.0, 1.0, size=(240, 3))
    y = 5.0 * (x[:, 2] - 0.5) + rng.normal(0.0, 0.03, size=len(x))
    split = 180

    trainer = HiFiANOVATrainer({
        'K1': K1,
        'K2': 1,
        'basis_name': basis_name,
        'strategy': 'variance',
        'stages': ['A', 'B'],
        'variable_selection': 'bic',
        'pair_candidates': 'either',
    })
    _, results = trainer.fit(
        jnp.asarray(x[:split]), jnp.asarray(y[:split]),
        jnp.asarray(x[split:]), jnp.asarray(y[split:]),
    )

    x3 = results['variable_selection']['per_group'][2]
    assert x3['selected']
    assert x3['delta_bic'] > 10.0

