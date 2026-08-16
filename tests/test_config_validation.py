"""Config footgun guards on HiFiANOVATrainer (DEC-036).

Two guarantees:
  * an unrecognized TOP-LEVEL config key raises a ``ValueError`` (typos no
    longer silently no-op), with an ``allow_unknown_keys`` escape hatch;
  * the caller's config dict — nested dicts included — is never mutated by
    construction (the trainer deep-copies before ``resolve_mode`` / stage logic).
"""

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest

from hifi_anova.training.trainer import HiFiANOVATrainer, KNOWN_CONFIG_KEYS


pytestmark = pytest.mark.smoke


def test_unknown_top_level_key_raises_with_suggestion():
    """A typo'd key must fail loudly and point at the nearest known key."""
    with pytest.raises(ValueError) as exc:
        HiFiANOVATrainer({'stages': ['A', 'B'], 'stategy': 'variance'})
    msg = str(exc.value)
    assert 'stategy' in msg
    assert 'strategy' in msg          # difflib near-match suggestion


def test_stray_selection_method_key_is_caught():
    """The historical footgun: 'selection_method' (real key: variable_selection)."""
    with pytest.raises(ValueError) as exc:
        HiFiANOVATrainer({'stages': ['A', 'B'], 'selection_method': 'bic'})
    assert 'selection_method' in str(exc.value)


def test_known_config_constructs_cleanly():
    """A config using only recognized keys must construct without error."""
    trainer = HiFiANOVATrainer({
        'K1': 5, 'K2': 3, 'stages': ['A', 'B'],
        'strategy': 'variance',
        'lambda_order1': 0.001, 'lambda_order2': 0.01,
        'variable_selection': 'bic', 'pair_candidates': 'either',
        'precision': 'float32', 'verbose': False,
    })
    assert trainer.config['stages'] == ['A', 'B']


def test_allow_unknown_keys_bypasses_validation():
    """The escape hatch lets a stray key through untouched."""
    trainer = HiFiANOVATrainer({
        'stages': ['A'], 'allow_unknown_keys': True, 'future_experimental': 42,
    })
    assert trainer.config['future_experimental'] == 42


def test_internal_auto_keys_are_allowed():
    """resolve_mode injects _auto_mode/_auto_threshold; these must not trip."""
    trainer = HiFiANOVATrainer({'mode': 'auto', 'auto_threshold': 0.02})
    assert trainer.config['_auto_mode'] is True


def test_caller_config_dict_not_mutated():
    """Construction must not mutate the caller's dict or its nested dicts.

    mode='full' drives resolve_mode to set residual_nn['enabled']=True; without
    the defensive deepcopy that write would leak into the caller's nested dict.
    """
    original = {
        'K1': 5, 'K2': 3, 'mode': 'full',
        'residual_nn': {'enabled': False, 'hidden_dims': [32]},
    }
    import copy as _copy
    snapshot = _copy.deepcopy(original)
    HiFiANOVATrainer(original)
    assert original == snapshot          # top-level 'mode' + nested dict intact
    assert original['residual_nn']['enabled'] is False


def test_known_keys_covers_documented_api_surface():
    """Guard against the allowlist silently drifting from the escape hatch."""
    assert 'allow_unknown_keys' in KNOWN_CONFIG_KEYS
    assert {'_auto_mode', '_auto_threshold'} <= KNOWN_CONFIG_KEYS


def test_one_call_api_surfaces_bogus_kwarg():
    """hifi_anova(X, y, bogus_kw=1) funnels the typo into config → must raise."""
    from hifi_anova.api import hifi_anova

    rng = np.random.default_rng(0)
    X = rng.uniform(0.0, 1.0, size=(120, 3))
    y = np.sin(2 * np.pi * X[:, 0]) + 0.1 * rng.standard_normal(120)
    with pytest.raises(ValueError) as exc:
        hifi_anova(X, y, verbose=False, definitely_not_a_key=1)
    assert 'definitely_not_a_key' in str(exc.value)
