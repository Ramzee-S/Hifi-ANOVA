"""Warning-aware, one-release aliases for public result mappings."""

from __future__ import annotations

import warnings
from collections.abc import Mapping


PUBLIC_RESULT_ALIASES = {
    'variance_sobol': 'log_variance_sobol',
    'tr_HHt': 'tr_H2',
}


def _canonical_shallow(mapping):
    """Canonicalize one mapping level without copying its values."""
    canonical = ResultAliasDict()
    for key, value in mapping.items():
        target = PUBLIC_RESULT_ALIASES.get(key, key)
        if target in canonical and key in PUBLIC_RESULT_ALIASES:
            continue
        canonical[target] = value
    return canonical


class ResultAliasDict(dict):
    """A canonical-key dict with deprecated read aliases.

    Iteration and JSON serialization expose only the physically stored canonical
    keys. Legacy key lookup remains available for one release and returns the
    exact same value object.
    """

    def _canonical_key(self, key):
        canonical = PUBLIC_RESULT_ALIASES.get(key)
        if canonical is not None:
            warnings.warn(
                f"{key} is deprecated; use {canonical}. The renamed surface "
                "does not change the underlying value.",
                DeprecationWarning,
                stacklevel=3,
            )
            return canonical
        return key

    def __getitem__(self, key):
        return super().__getitem__(self._canonical_key(key))

    def get(self, key, default=None):
        return super().get(self._canonical_key(key), default)

    def __contains__(self, key):
        return super().__contains__(self._canonical_key(key))

    def copy(self):
        """Return a shallow alias-aware copy, matching ``dict.copy`` semantics."""
        return type(self)(self)

    def __or__(self, other):
        if not isinstance(other, Mapping):
            return NotImplemented
        left = _canonical_shallow(self)
        right = _canonical_shallow(other)
        return ResultAliasDict(dict(left) | dict(right))

    def __ror__(self, other):
        if not isinstance(other, Mapping):
            return NotImplemented
        left = _canonical_shallow(other)
        right = _canonical_shallow(self)
        return ResultAliasDict(dict(left) | dict(right))

    def __ior__(self, other):
        if not isinstance(other, Mapping):
            return NotImplemented
        canonical = self | other
        super().clear()
        super().update(canonical)
        return self


def canonical_result_mapping(mapping, *, warn_legacy=False):
    """Recursively canonicalize result keys and attach read aliases.

    Legacy-only artifacts are migrated in memory. If both spellings occur, the
    canonical spelling wins; this keeps current artifacts deterministic without
    inventing a persistence-schema version.
    """
    if isinstance(mapping, dict):
        canonical = ResultAliasDict()
        for key, value in mapping.items():
            target = PUBLIC_RESULT_ALIASES.get(key, key)
            if warn_legacy and target != key:
                warnings.warn(
                    f"{key} is deprecated; use {target}. The renamed surface "
                    "does not change the underlying value.",
                    DeprecationWarning,
                    stacklevel=3,
                )
            if target in canonical and key in PUBLIC_RESULT_ALIASES:
                continue
            canonical[target] = canonical_result_mapping(
                value, warn_legacy=warn_legacy)
        return canonical
    if isinstance(mapping, list):
        return [canonical_result_mapping(value, warn_legacy=warn_legacy)
                for value in mapping]
    return mapping
