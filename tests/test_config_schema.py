"""Declarative config schema + typed HiFiConfig container."""

import json
import numpy as np
import pytest

from hifi_anova import config_schema, HiFiConfig, CONFIG_SCHEMA, hifi_anova
from hifi_anova.config_schema import FieldSpec
from hifi_anova.training.trainer import KNOWN_CONFIG_KEYS

# Internal / escape-hatch keys the schema deliberately does not describe.
# 'array_backend' is fit PROVENANCE, not a user knob: the user passes the
# public `backend=` arg (jax|numpy|auto), which the api wrapper resolves and
# records here — so it is deliberately absent from the user-facing schema.
_INTERNAL = {'_auto_mode', '_auto_threshold', '_mixed_selection_neutralized',
             'allow_unknown_keys', 'array_backend'}


def test_schema_matches_known_config_keys():
    """Every user-facing trainer key is described, and no schema key is bogus.

    This is the drift guard: adding a config key to the trainer without a schema
    entry (or vice versa) fails here.
    """
    schema_keys = set(CONFIG_SCHEMA)
    user_facing = KNOWN_CONFIG_KEYS - _INTERNAL
    missing = user_facing - schema_keys
    extra = schema_keys - KNOWN_CONFIG_KEYS
    assert not missing, f"config keys missing a schema entry: {sorted(missing)}"
    assert not extra, f"schema keys not in KNOWN_CONFIG_KEYS: {sorted(extra)}"


def test_schema_entries_well_formed():
    for key, spec in CONFIG_SCHEMA.items():
        assert isinstance(spec, FieldSpec)
        assert spec.key == key
        assert spec.type in ('int', 'float', 'bool', 'str', 'enum', 'list', 'dict')
        assert spec.help, f"{key} has no help text"
        if spec.type == 'enum':
            assert spec.choices, f"enum {key} has no choices"


def test_schema_as_dict_is_json_serializable():
    d = config_schema(as_dict=True)
    json.dumps(d)   # must not raise
    assert d['K1']['type'] == 'int'
    assert d['basis_name']['choices'] == list(CONFIG_SCHEMA['basis_name'].choices)


def test_hificonfig_rejects_unknown_key():
    with pytest.raises(KeyError):
        HiFiConfig(K1=5, stategy='variance')   # typo


def test_hificonfig_rejects_bad_enum():
    with pytest.raises(ValueError):
        HiFiConfig(basis_name='wavelet')       # not a known family


def test_hificonfig_to_dict_roundtrip_and_defaults():
    cfg = HiFiConfig(K1=6, heteroscedastic_guard=False)
    assert cfg.to_dict() == {'K1': 6, 'heteroscedastic_guard': False}
    # Attribute access: set value returned; unset falls back to schema default.
    assert cfg.K1 == 6
    assert cfg.K2 == CONFIG_SCHEMA['K2'].default    # 3
    assert cfg.get('verbose') is True               # schema default
    assert HiFiConfig.from_dict({'K1': 6}).to_dict() == {'K1': 6}


def test_hificonfig_splats_into_one_call():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (300, 3))
    y = np.sin(2 * np.pi * X[:, 0]) + 0.1 * rng.standard_normal(300)
    cfg = HiFiConfig(K1=4, K2=2, mode='second')
    res = hifi_anova(X, y, verbose=False, **cfg.to_dict())
    assert res.model is not None
