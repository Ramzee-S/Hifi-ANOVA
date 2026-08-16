"""Declarative, machine-readable schema for the trainer/one-call config.

Purpose: give a GUI or web front-end a single place to enumerate the config
keys with their types, defaults, allowed values, ranges, and help text — so a
form can be generated automatically instead of hard-coding ~50 controls. The
enum choice sets are imported from :mod:`hifi_anova.validation` (the validation
authority), so the schema cannot drift from what the trainer actually accepts.

This module describes the *public* config surface. Runtime validation still
lives in :meth:`HiFiANOVATrainer._validate_config_keys` (key names) and
:func:`hifi_anova.validation.validate_config` (types/ranges/enums); the schema
does not replace them. A test pins ``CONFIG_SCHEMA`` keys to the trainer's
``KNOWN_CONFIG_KEYS`` so the two stay in lock-step.

:class:`HiFiConfig` is an optional typed, schema-validated container: build one
with keyword args, get IDE/GUI-friendly key checking, and splat it into the
one-call API:

    cfg = HiFiConfig(K1=6, heteroscedastic=... )   # invalid key -> error
    result = hifi_anova(X, y, **cfg.to_dict())
"""

from dataclasses import dataclass
from typing import Any, Optional, Tuple, Dict

from .validation import (
    STAGES, BASIS_FAMILIES, _BASIS_TYPES, _PRECISIONS, RESIDUAL_TYPES,
    _SELECTION_METHODS, _PRUNING_METHODS, _PAIR_CANDIDATE_MODES,
    _PAIR_SELECTION_MODES, _TRIPLE_SELECTION_MODES, _VAR_PAIR_SELECTION_MODES,
    _VAR_TRIPLE_SELECTION_MODES, _STAGE_D_ESTIMATORS, _STRATEGY_NAMES, suggest,
)

_MODES = ("first", "second", "full", "heteroscedastic", "auto")

# Sentinel for HiFiConfig.get(): distinguishes "no fallback given" (fall back to
# the schema default) from an explicit ``default=None``.
_MISSING = object()


@dataclass(frozen=True)
class FieldSpec:
    """One config key's schema entry (see :data:`CONFIG_SCHEMA`)."""
    key: str
    type: str                       # 'int'|'float'|'bool'|'str'|'enum'|'list'|'dict'
    default: Any = None
    choices: Optional[Tuple] = None  # for 'enum'/'list' element choices
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    nullable: bool = False           # None is a valid value (e.g. 'disable')
    group: str = "misc"
    help: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """JSON-friendly plain-dict form (for shipping the schema to a GUI)."""
        return {
            "key": self.key, "type": self.type, "default": self.default,
            "choices": list(self.choices) if self.choices else None,
            "min": self.minimum, "max": self.maximum,
            "nullable": self.nullable, "group": self.group, "help": self.help,
        }


def _f(*args, **kw) -> FieldSpec:
    return FieldSpec(*args, **kw)


# The schema. Defaults reflect the one-call ``hifi_anova(...)`` effective
# defaults where a key is a named argument, else the trainer's ``cfg.get(...)``
# default. A ``None`` default with an explanatory ``help`` marks a value the
# library resolves at fit time.
_SPECS = [
    # --- basis / order sizes ---
    _f("K1", "int", 5, minimum=1, group="basis",
       help="Max harmonic/degree for first-order terms."),
    _f("K2", "int", 3, minimum=0, group="basis",
       help="Max harmonic for second-order (pairwise) terms; 0 disables pairs. "
            "May also be a per-pair mapping {(i, j): K2_ij} that pins the "
            "exact retained pairs with each pair's own order (BR-04)."),
    _f("K3", "int", 0, minimum=0, group="basis",
       help="Max harmonic for third-order (triple) terms; 0 disables triples."),
    _f("Kh", "int", 3, minimum=0, group="variance",
       help="Max harmonic for the first-order log-variance model (Stage D)."),
    _f("K2h", "int", 0, minimum=0, group="variance",
       help="Max harmonic for second-order variance terms; 0 disables."),
    _f("K3h", "int", 0, minimum=0, group="variance",
       help="Max harmonic for third-order variance terms; 0 disables."),
    _f("basis_name", "enum", "fourier", choices=BASIS_FAMILIES, group="basis",
       help="Uniform basis family for every variable."),
    _f("basis_type", "enum", "full", choices=_BASIS_TYPES, group="basis",
       help="Linear-term inclusion policy across orders."),
    _f("basis_per_variable", "dict", None, nullable=True, group="basis",
       help="Mixed per-variable {idx:{'basis','K'}} spec, or 'auto'. Stage A+B only."),
    _f("variable_orders", "dict", None, nullable=True, group="basis",
       help="Order-selective variable membership {idx: [orders]}, orders a "
            "subset of {1, 2}: [2] admits a variable to pair terms while "
            "EXCLUDING its first-order block (non-hierarchical model — the "
            "pair share is conditional on the omitted marginal); [1] keeps "
            "the marginal but excludes the variable from every pair (BR-06)."),
    _f("variance_variables", "list", None, nullable=True, group="variance",
       help="First-order variance-model variable subset (Stage D). Excluding "
            "a variable ASSERTS the noise is homoscedastic along it — a "
            "modeling assumption, not a data-driven finding (BR-01)."),
    # --- staging / mode / strategy ---
    _f("stages", "list", ["A", "B"], choices=STAGES, group="staging",
       help="Explicit stage list; a 'mode' preset overrides this."),
    _f("mode", "enum", "second", choices=_MODES, group="staging",
       help="Named stage preset (first/second/full/heteroscedastic/auto)."),
    _f("strategy", "str", None, choices=_STRATEGY_NAMES, nullable=True,
       group="regularization",
       help="Regularization strategy; None resolves to 'curvature' when "
            "heteroscedastic else 'variance'. Also accepts sobolev*/spectral* "
            "families or a per-order dict."),
    # --- regularization strengths ---
    _f("lambda_order1", "float", 1e-3, minimum=0.0, group="regularization",
       help="Ridge penalty scale for first-order terms."),
    _f("lambda_order2", "float", 1e-2, minimum=0.0, group="regularization",
       help="Ridge penalty scale for second-order terms."),
    _f("lambda_order3", "float", 0.1, minimum=0.0, group="regularization",
       help="Ridge penalty scale for third-order terms."),
    _f("lambda_h", "float", 0.1, minimum=0.0, group="variance",
       help="Penalty scale for the first-order variance model (Stage D)."),
    _f("lambda_h2", "float", None, minimum=0.0, nullable=True, group="variance",
       help="Second-order variance penalty; defaults to 10*lambda_h."),
    _f("lambda_h3", "float", None, minimum=0.0, nullable=True, group="variance",
       help="Third-order variance penalty; defaults to 100*lambda_h."),
    _f("lambda_residual", "float", 1.0, minimum=0.0, group="residual",
       help="Penalty scale for the Stage-C residual model."),
    _f("lambda_h_residual", "float", None, minimum=0.0, nullable=True,
       group="variance", help="Variance-residual penalty; defaults to 10*lambda_h."),
    # --- variable / pair / triple selection & pruning ---
    _f("variable_selection", "enum", "bic", choices=_SELECTION_METHODS,
       nullable=True, group="selection",
       help="Active-variable selection method; None disables. Unsupported on "
            "mixed per-variable bases."),
    _f("first_order_pruning", "enum", "none", choices=_PRUNING_METHODS,
       group="selection", help="Post-fit first-order block pruning criterion."),
    _f("pair_candidates", "enum", None, choices=_PAIR_CANDIDATE_MODES,
       nullable=True, group="selection",
       help="Pair candidate-generation heuristic (pairs among active vars)."),
    _f("pair_selection", "str", None, choices=_PAIR_SELECTION_MODES,
       nullable=True, group="selection",
       help="Legacy one-shot pair selection, or an explicit index list."),
    _f("pair_pruning", "enum", "none", choices=_PRUNING_METHODS,
       group="selection", help="Post-fit pair pruning criterion."),
    _f("pair_threshold", "float", 0.01, minimum=0.0, maximum=1.0,
       group="selection", help="Sobol-share threshold for candidate selection."),
    _f("max_pair_variables", "int", None, minimum=1, nullable=True,
       group="selection", help="Cap on the number of active variables paired."),
    _f("triple_selection", "enum", "all_active", choices=_TRIPLE_SELECTION_MODES,
       group="selection", help="Third-order candidate selection mode."),
    _f("triple_pruning", "enum", "none", choices=_PRUNING_METHODS,
       group="selection", help="Post-fit triple pruning criterion."),
    _f("var_pair_selection", "enum", None, choices=_VAR_PAIR_SELECTION_MODES,
       nullable=True, group="variance",
       help="Second-order variance pair selection ('all'/'auto')."),
    _f("var_triple_selection", "enum", None, choices=_VAR_TRIPLE_SELECTION_MODES,
       nullable=True, group="variance",
       help="Third-order variance triple selection ('all')."),
    _f("variance_selection_margin", "float", 2e-3, minimum=0.0, group="variance",
       help="Relative held-out-NLL margin to keep the heteroscedastic model."),
    _f("variance_selection_mean_consistent", "bool", True, group="variance",
       help="Compare keep/revert using the same (weighted) mean on both sides."),
    _f("variance_selection_mean_fallback", "bool", False, group="variance",
       help="Keep the variance model with the unit-weight mean if the GLS mean "
            "degraded the package."),
    _f("variance_residual", "dict", None, nullable=True, group="variance",
       help="Optional variance-residual (rbf/rff/nystrom) spec dict."),
    # --- residual (Stage C) ---
    _f("residual", "dict", None, choices=RESIDUAL_TYPES, nullable=True,
       group="residual",
       help="Stage-C residual type/spec (nn/rbf/rff/nystrom); None disables."),
    _f("residual_nn", "dict", None, nullable=True, group="residual",
       help="Stage-C neural-residual spec dict."),
    # --- heteroscedastic (Stage D) alternation ---
    _f("heteroscedastic_guard", "bool", True, group="variance",
       help="Keep the variance model only if it beats a constant variance on "
            "held-out NLL."),
    _f("min_noise_ratio", "float", 1e-2, minimum=0.0, maximum=1.0,
       group="variance",
       help="Skip Stage D below this noise-to-signal ratio (near-noiseless)."),
    _f("leverage_correction", "bool", True, group="variance",
       help="Leverage-debias in-sample squared residuals before the variance "
            "solve (part of the default estimator)."),
    _f("alternating_early_stop", "bool", True, group="variance",
       help="Keep the held-out-best Stage-D outer iterate (validation selection)."),
    _f("alternating_tol", "float", 1e-4, minimum=0.0, group="variance",
       help="Relative train-NLL tolerance for the alternating loop (>0)."),
    _f("max_outer_iter", "int", 10, minimum=1, group="variance",
       help="Max Stage-D outer (mean/variance) alternations."),
    _f("newton_max_iter", "int", 10, minimum=1, group="variance",
       help="Max Newton iterations per variance solve."),
    _f("stage_d_joint_gls_mean", "bool", True, group="variance",
       help="Profile the intercept jointly (weighted-center y and Phi) in the "
            "Stage-D mean update."),
    _f("stage_d_estimator", "enum", None, choices=_STAGE_D_ESTIMATORS,
       nullable=True, group="variance",
       help="Stage-D estimator identity selector; resolves leverage/early-stop."),
    # --- linear-term toggles ---
    _f("include_linear_1", "bool", True, group="basis",
       help="Include the linear term in the first-order mean basis."),
    _f("include_linear_2", "bool", True, group="basis",
       help="Include linear terms in second-order mean features."),
    _f("include_linear_3", "bool", True, group="basis",
       help="Include linear terms in third-order mean features."),
    _f("include_linear_h1", "bool", True, group="variance",
       help="Include the linear term in the first-order variance basis."),
    _f("include_linear_h2", "bool", True, group="variance",
       help="Include linear terms in second-order variance features."),
    _f("include_linear_h3", "bool", True, group="variance",
       help="Include linear terms in third-order variance features."),
    # --- runtime ---
    _f("precision", "enum", None, choices=_PRECISIONS, nullable=True,
       group="runtime",
       help="Fit-weight precision ('float32'/'float64'); None resolves by "
            "precedence. Post-fit analytics always run in float64."),
    _f("verbose", "bool", True, group="runtime",
       help="Print stage-by-stage progress to stdout."),
    _f("auto_threshold", "float", 0.01, minimum=0.0, group="staging",
       help="Residual-fraction threshold for mode='auto' stage decisions."),
]

CONFIG_SCHEMA: Dict[str, FieldSpec] = {s.key: s for s in _SPECS}


def config_schema(as_dict: bool = False):
    """Return the config schema.

    ``as_dict=False`` (default) returns ``{key: FieldSpec}``; ``as_dict=True``
    returns ``{key: {...}}`` plain dicts, JSON-serializable for shipping the
    schema to a front-end.
    """
    if as_dict:
        return {k: s.to_dict() for k, s in CONFIG_SCHEMA.items()}
    return dict(CONFIG_SCHEMA)


class HiFiConfig:
    """Optional typed, schema-validated config container.

    Light wrapper over a dict: it checks key names (with typo suggestions) and
    enum membership against :data:`CONFIG_SCHEMA` at construction, then defers
    full type/range validation to the trainer at fit time. Only explicitly-set
    keys are stored, so ``to_dict()`` splats cleanly into ``hifi_anova(...)``
    and unset keys fall through to the library defaults.
    """

    def __init__(self, **values):
        object.__setattr__(self, "_values", {})
        for k, v in values.items():
            self[k] = v   # validates via __setitem__

    def __setitem__(self, key, value):
        if key not in CONFIG_SCHEMA:
            raise KeyError(
                f"Unknown config key {key!r}"
                f"{suggest(key, CONFIG_SCHEMA)}. See config_schema().")
        spec = CONFIG_SCHEMA[key]
        if (spec.choices and value is not None
                and spec.type in ("enum", "str") and value not in spec.choices):
            # Only reject when the value is a plain scalar outside the choice
            # set; dict/list-valued fields (e.g. per-order strategy) are passed
            # through for the trainer's richer validation.
            if not isinstance(value, (dict, list, tuple)):
                raise ValueError(
                    f"{key} must be one of {list(spec.choices)} (or None)"
                    f"{suggest(value, spec.choices)}; got {value!r}.")
        self._values[key] = value

    def __getitem__(self, key):
        return self._values[key]

    def __setattr__(self, key, value):
        self[key] = value

    def __getattr__(self, key):
        try:
            return self._values[key]
        except KeyError:
            if key in CONFIG_SCHEMA:
                return CONFIG_SCHEMA[key].default
            raise AttributeError(key)

    def get(self, key, default=_MISSING):
        """Value if set, else an explicit ``default``, else the schema default."""
        if key in self._values:
            return self._values[key]
        if default is not _MISSING:
            return default
        if key in CONFIG_SCHEMA:
            return CONFIG_SCHEMA[key].default
        return None

    def to_dict(self) -> Dict[str, Any]:
        """The explicitly-set keys, for ``hifi_anova(X, y, **cfg.to_dict())``."""
        return dict(self._values)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HiFiConfig":
        return cls(**d)

    def __repr__(self):
        inner = ", ".join(f"{k}={v!r}" for k, v in self._values.items())
        return f"HiFiConfig({inner})"
