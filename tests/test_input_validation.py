"""Public + nested input validation (DEC-046).

Invalid or contradictory public configuration must fail *early* with a specific,
actionable ``ValueError`` — naming the option, the received value/type, and the
valid range/alternatives — instead of being silently accepted (a footgun) or
crashing obscurely deep in the solve. Tests are table-driven for the invalid
matrix and pin the exception type + an actionable message fragment (never a bare
"some exception"); positive controls assert valid boundary values still pass.

Layers exercised:
  * pure helpers (``hifi_anova.validation``) — fast, no fit;
  * the trainer boundary (``HiFiANOVATrainer.__init__`` value validation) —
    construction only, no fit;
  * the one-call boundary (``hifi_anova``) — invalid inputs raise before any fit,
    so these stay fast.
"""

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest

from hifi_anova import validation as V
from hifi_anova.training.trainer import HiFiANOVATrainer
from hifi_anova.api import hifi_anova


pytestmark = pytest.mark.smoke


def _Xy(n=140, d=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, size=(n, d))
    y = np.sin(2 * np.pi * X[:, 0]) + 0.5 * X[:, 1] + 0.1 * rng.standard_normal(n)
    return X, y


# ---------------------------------------------------------------------------
# Numeric type/range helpers (pure, fast)
# ---------------------------------------------------------------------------
class TestNumericHelpers:
    def test_require_int_rejects_bool(self):
        with pytest.raises(ValueError, match="not a bool"):
            V.require_int("K1", True)

    def test_require_int_rejects_numpy_bool(self):
        with pytest.raises(ValueError, match="not a bool"):
            V.require_int("K1", np.bool_(True))

    def test_require_int_rejects_float(self):
        with pytest.raises(ValueError, match="must be an integer"):
            V.require_int("K1", 5.0)

    def test_require_int_accepts_numpy_integer(self):
        assert V.require_int("K1", np.int64(7)) == 7

    def test_require_int_min(self):
        with pytest.raises(ValueError, match=">= 1"):
            V.require_int("K1", 0, minimum=1)

    @pytest.mark.parametrize("bad,frag", [
        (float("nan"), "finite"),
        (float("inf"), "finite"),
        (True, "not a bool"),
        (-1.0, ">= 0"),
    ])
    def test_require_number_rejects(self, bad, frag):
        with pytest.raises(ValueError, match=frag):
            V.require_number("lambda_order1", bad, minimum=0.0)

    def test_require_number_strict_min(self):
        with pytest.raises(ValueError, match="> 0"):
            V.require_number("tol", 0.0, minimum=0.0, strict_min=True)

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
    def test_require_fraction_bounds(self, bad):
        with pytest.raises(ValueError):
            V.require_fraction("auto_threshold", bad)

    def test_require_fraction_ok(self):
        assert V.require_fraction("auto_threshold", 0.35) == 0.35

    def test_require_choice_suggests(self):
        with pytest.raises(ValueError, match="did you mean 'bic'"):
            V.require_choice("variable_selection", "bik",
                             ("bic", "group_lasso", "1se"))


# ---------------------------------------------------------------------------
# Public data boundary (X / y) — raises before any fit
# ---------------------------------------------------------------------------
class TestPublicDataBoundary:
    def test_X_one_dim(self):
        with pytest.raises(ValueError, match="X must be 2-D"):
            hifi_anova(np.linspace(0, 1, 100), np.zeros(100), verbose=False)

    def test_X_scalar(self):
        with pytest.raises(ValueError, match="scalar"):
            hifi_anova(np.float64(1.0), np.float64(1.0), verbose=False)

    def test_X_nonnumeric(self):
        with pytest.raises(ValueError, match="numeric feature matrix"):
            hifi_anova(np.array([["a", "b"]] * 10), np.zeros(10), verbose=False)

    def test_X_list_of_lists_is_accepted_shape(self):
        """A rectangular Python list is valid 2-D numeric data (no AttributeError)."""
        X = [[0.1, 0.2, 0.3]] * 5
        # Shape/dtype pass; the too-few-samples check then fires in preprocessing.
        with pytest.raises(ValueError, match="Not enough samples"):
            hifi_anova(X, list(range(5)), verbose=False)

    def test_y_wide_2d_rejected(self):
        X, y = _Xy()
        with pytest.raises(ValueError, match="scalar target"):
            hifi_anova(X, np.column_stack([y, y]), verbose=False)

    def test_y_nonnumeric(self):
        X, y = _Xy()
        with pytest.raises(ValueError, match="numeric target"):
            hifi_anova(X, np.array(["a"] * len(y)), verbose=False)


# ---------------------------------------------------------------------------
# feature_names
# ---------------------------------------------------------------------------
class TestFeatureNames:
    @pytest.mark.parametrize("names,frag", [
        (["a", "b"], "provide exactly 3"),
        (["a", "b", "c", "d"], "provide exactly 3"),
        ([0, 1, 2], "must all be strings"),
        (["a", "a", "b"], "duplicate"),
        ("abc", "must be a list"),
    ])
    def test_bad_feature_names(self, names, frag):
        X, y = _Xy(d=3)
        with pytest.raises(ValueError, match=frag):
            hifi_anova(X, y, feature_names=names, verbose=False)


# ---------------------------------------------------------------------------
# basis_per_variable nested schema
# ---------------------------------------------------------------------------
class TestBasisPerVariable:
    @pytest.mark.parametrize("bpv,frag", [
        ({0: {"basis": "spline", "K": 4}}, "must be one of"),
        ({9: {"basis": "legendre", "K": 4}}, "out of range"),
        ({"0": {"basis": "legendre", "K": 4}}, "integer variable indices"),
        ({True: {"basis": "legendre", "K": 4}}, "integer variable indices"),
        ({0: {"basi": "legendre", "K": 4}}, "Unknown key"),
        ({0: {"basis": "legendre", "K": 0}}, ">= 1"),
        ({0: {"basis": "legendre", "K": -2}}, ">= 1"),
        ({0: {"basis": "legendre", "K": 3.5}}, "must be an integer"),
        ([{"basis": "legendre", "K": 4}], "'auto' or a mapping"),
        ("bogus", "'auto' or a mapping"),
    ])
    def test_bad_basis_per_variable_helper(self, bpv, frag):
        with pytest.raises(ValueError, match=frag):
            V.validate_basis_per_variable(bpv, D=3)

    def test_auto_and_partial_mapping_ok(self):
        assert V.validate_basis_per_variable("auto", D=3) == "auto"
        assert V.validate_basis_per_variable(
            {0: {"basis": "legendre", "K": 4}}, D=3)

    def test_bad_basis_per_variable_integration(self):
        """A malformed spec is caught on the mixed fit path, not silently defaulted."""
        X, y = _Xy(d=3)
        with pytest.raises(ValueError, match="out of range"):
            hifi_anova(X, y, basis_per_variable={5: {"basis": "legendre", "K": 3}},
                       variable_selection=None, verbose=False)


# ---------------------------------------------------------------------------
# Residual family schemas
# ---------------------------------------------------------------------------
class TestResidualSchema:
    def test_unknown_type(self):
        with pytest.raises(ValueError, match="Unknown residual type"):
            V.validate_residual_spec("residual", {"type": "gaussian"})

    def test_nested_typo(self):
        with pytest.raises(ValueError, match="did you mean 'n_centers'"):
            V.validate_residual_spec("residual", {"type": "rbf", "n_centrs": 5})

    def test_variance_residual_rejects_nn(self):
        with pytest.raises(ValueError, match="Unknown residual type"):
            V.validate_residual_spec("variance_residual", {"type": "nn"},
                                     allow_nn=False)

    def test_valid_specs(self):
        # Returns the normalized spec dict (with 'type' present).
        assert V.validate_residual_spec(
            "residual", {"type": "rbf", "n_centers": 50, "sigma": 0.2}
        )["type"] == "rbf"
        assert V.validate_residual_spec(
            "residual", {"type": "nn", "enabled": False})["type"] == "nn"

    def test_one_call_unknown_residual(self):
        X, y = _Xy()
        with pytest.raises(ValueError, match="Unknown residual type"):
            hifi_anova(X, y, residual="rbg", verbose=False)


# ---------------------------------------------------------------------------
# Stage / mode dependency matrix
# ---------------------------------------------------------------------------
class TestStagesAndMode:
    @pytest.mark.parametrize("stages,frag", [
        (["A", "Z"], "unknown stage"),
        (["A", "A", "B"], "duplicate"),
        (["B", "A"], "canonical order"),
        (["B", "D"], "must include 'A'"),
        ([], "empty"),
        ("AB", "must be a list"),
    ])
    def test_bad_stages(self, stages, frag):
        with pytest.raises(ValueError, match=frag):
            HiFiANOVATrainer({"stages": stages, "K1": 5, "K2": 3,
                              "strategy": "variance"})

    @pytest.mark.parametrize("stages", [["A"], ["A", "B"], ["A", "D"],
                                        ["A", "B", "C"], ["A", "B", "C", "D"]])
    def test_valid_stage_subsets_construct(self, stages):
        HiFiANOVATrainer({"stages": stages, "K1": 5, "K2": 3,
                          "strategy": "variance"})

    def test_auto_plus_heteroscedastic_rejected(self):
        X, y = _Xy()
        with pytest.raises(ValueError, match="ambiguous"):
            hifi_anova(X, y, mode="auto", heteroscedastic=True, verbose=False)

    def test_unknown_mode(self):
        X, y = _Xy()
        with pytest.raises(ValueError, match="Unknown mode"):
            hifi_anova(X, y, mode="turbo", verbose=False)


# ---------------------------------------------------------------------------
# Numeric type/range matrix (via trainer construction — no fit)
# ---------------------------------------------------------------------------
class TestNumericConfigViaTrainer:
    @pytest.mark.parametrize("cfg,frag", [
        ({"K1": -1}, "K1 must be >= 1"),
        ({"K1": 5.0}, "K1 must be an integer"),
        ({"K1": True}, "not a bool"),
        ({"K2": -3}, "K2 must be >= 0"),
        ({"lambda_order1": -0.5}, "lambda_order1 must be >= 0"),
        ({"lambda_order1": float("nan")}, "must be finite"),
        ({"min_noise_ratio": -1.0}, "min_noise_ratio must be >= 0"),
        ({"max_outer_iter": 0}, "max_outer_iter must be >= 1"),
        ({"alternating_tol": 0.0}, "alternating_tol must be > 0"),
        ({"strategy": "curvatur"}, "strategy must be one of"),
        ({"variable_selection": "lasso"}, "variable_selection must be one of"),
        ({"first_order_pruning": "bogus"}, "first_order_pruning must be one of"),
    ])
    def test_bad_numeric_or_enum(self, cfg, frag):
        base = {"stages": ["A", "B"], "K1": 5, "K2": 3, "strategy": "variance"}
        base.update(cfg)
        with pytest.raises(ValueError, match=frag):
            HiFiANOVATrainer(base)

    @pytest.mark.parametrize("cfg", [
        {"K2": 0}, {"lambda_order1": 0.0}, {"strategy": "sobolev_1"},
        {"strategy": "spectral_2"}, {"variable_selection": None},
        {"first_order_pruning": "none"}, {"min_noise_ratio": 0.0},
    ])
    def test_valid_boundary_values_construct(self, cfg):
        base = {"stages": ["A", "B"], "K1": 5, "K2": 3, "strategy": "variance"}
        base.update(cfg)
        HiFiANOVATrainer(base)


# ---------------------------------------------------------------------------
# allow_unknown_keys does NOT bypass value/type/range safety
# ---------------------------------------------------------------------------
class TestAllowUnknownKeys:
    def test_experimental_key_allowed_but_values_still_checked(self):
        # The experimental key is tolerated ...
        HiFiANOVATrainer({"stages": ["A"], "allow_unknown_keys": True,
                          "future_experimental": 42})
        # ... but a bad *value* on a known key still fails.
        with pytest.raises(ValueError, match="K1 must be >= 1"):
            HiFiANOVATrainer({"stages": ["A"], "allow_unknown_keys": True,
                              "K1": -1})


# ---------------------------------------------------------------------------
# One-call and direct-trainer consistency
# ---------------------------------------------------------------------------
class TestConsistency:
    def test_bad_K1_same_error_both_paths(self):
        X, y = _Xy()
        with pytest.raises(ValueError, match="K1 must be >= 1"):
            hifi_anova(X, y, K1=-1, verbose=False)
        with pytest.raises(ValueError, match="K1 must be >= 1"):
            HiFiANOVATrainer({"stages": ["A", "B"], "K1": -1, "K2": 3,
                              "strategy": "variance"})


# ---------------------------------------------------------------------------
# Positive controls: valid uniform + mixed calls still fit
# ---------------------------------------------------------------------------
class TestValidControls:
    def test_uniform_call_fits(self):
        X, y = _Xy(n=160, d=3)
        res = hifi_anova(X, y, K1=4, K2=2, mode="second", verbose=False)
        assert res.model is not None
        assert len(res.feature_names) == 3

    def test_mixed_call_fits(self):
        X, y = _Xy(n=200, d=3)
        res = hifi_anova(
            X, y, K1=4, K2=2,
            basis_per_variable={0: {"basis": "legendre", "K": 3},
                                1: {"basis": "fourier", "K": 2}},
            variable_selection=None, verbose=False)
        assert res.model is not None


# ---------------------------------------------------------------------------
# Reviewer follow-ups (DEC-046 hardening)
# ---------------------------------------------------------------------------
class TestResidualShorthandNormalization:
    """A string shorthand must be normalized into the stored config, not left as
    a bare string that later crashes on `.get()` (residual + variance_residual)."""

    def test_variance_residual_string_normalized_in_config(self):
        # The trainer deep-copies the caller's dict (DEC-036), so the normalized
        # form lives in trainer.config — which is what Stage D reads via .get().
        t = HiFiANOVATrainer({"stages": ["A", "D"], "variance_residual": "rbf"})
        assert t.config["variance_residual"] == {"type": "rbf"}

    def test_residual_string_normalized_in_config(self):
        t = HiFiANOVATrainer({"stages": ["A", "C"], "residual": "rbf"})
        assert t.config["residual"]["type"] == "rbf"

    @pytest.mark.integration
    def test_variance_residual_string_fits_stage_d(self):
        """The former crash: variance_residual='rbf' → AttributeError in Stage D."""
        X, y = _Xy(n=240, d=3, seed=3)
        res = hifi_anova(X, y, K1=4, K2=2, heteroscedastic=True,
                         variance_residual="rbf", verbose=False)
        assert res.model is not None


class TestResidualValueValidation:
    @pytest.mark.parametrize("spec,frag", [
        ({"type": "rbf", "n_centers": 0}, ">= 1"),
        ({"type": "rbf", "sigma": -1.0}, "> 0"),
        ({"type": "rff", "n_features": 0}, ">= 1"),
        ({"type": "rff", "gamma": 0.0}, "> 0"),
        ({"type": "nystrom", "n_inducing": -5}, ">= 1"),
        ({"type": "nystrom", "lengthscale": 0.0}, "> 0"),
        ({"type": "nn", "epochs": 0}, ">= 1"),
        ({"type": "nn", "batch_size": 0}, ">= 1"),
        ({"type": "nn", "lr": 0.0}, "> 0"),
        ({"type": "nn", "hidden_dims": []}, "non-empty"),
        ({"type": "nn", "hidden_dims": [32, -4]}, ">= 1"),
        ({"type": "rbf", "center_method": "kmens"}, "did you mean 'kmeans'"),
        ({"type": "nystrom", "kernel": "matern99"}, "must be one of"),
    ])
    def test_bad_residual_values(self, spec, frag):
        with pytest.raises(ValueError, match=frag):
            V.validate_residual_spec("residual", spec)

    def test_constructor_options_accepted(self):
        assert V.validate_residual_spec(
            "residual", {"type": "rbf", "center_method": "random"})["type"] == "rbf"
        assert V.validate_residual_spec(
            "residual", {"type": "nystrom", "signal_variance": 2.0,
                         "kernel": "matern52"})["type"] == "nystrom"

    def test_lambda_and_enabled_context(self):
        # lambda_residual is ignored by NN and by variance_residual → rejected.
        with pytest.raises(ValueError, match="Unknown key"):
            V.validate_residual_spec("residual", {"type": "nn",
                                                  "lambda_residual": 1.0})
        with pytest.raises(ValueError, match="Unknown key"):
            V.validate_residual_spec("variance_residual",
                                     {"type": "rbf", "lambda_residual": 1.0},
                                     allow_nn=False, context="variance_residual")
        # enabled is a Stage-C concept only.
        with pytest.raises(ValueError, match="Unknown key"):
            V.validate_residual_spec("variance_residual",
                                     {"type": "rbf", "enabled": False},
                                     allow_nn=False, context="variance_residual")

    @pytest.mark.integration
    def test_analytic_residual_enabled_false_skips_stage_c(self):
        """enabled=False now disables an analytic residual too (unambiguous):
        Stage C records nothing because it was skipped (direct-trainer path)."""
        import jax.numpy as jnp
        X, y = _Xy(n=200, d=2, seed=5)
        x = jnp.asarray(X)
        yj = jnp.asarray(y)
        cfg = {"stages": ["A", "C"], "K1": 4, "K2": 0, "strategy": "variance",
               "lambda_order1": 0.01,
               "residual": {"type": "rbf", "n_centers": 20, "enabled": False}}
        _model, results = HiFiANOVATrainer(cfg).fit(x, yj, x, yj)
        assert "stage_C" not in results


class TestBroaderConfigCoverage:
    @pytest.mark.parametrize("cfg,frag", [
        ({"strategy": 42}, "strategy must be one of"),
        ({"strategy": {"order1": "bogus"}}, "must be one of"),
        ({"strategy": {"ordr1": "curvature"}}, "Unknown key"),
        ({"strategy": "sobolev_s"}, "must be one of"),
        ({"strategy": "spectral_a"}, "must be one of"),
        ({"strategy": "sobolevgarbage"}, "must be one of"),
        ({"pair_selection": (0, 1)}, "pair_selection must be a mode"),
        ({"pair_selection": [0, True]}, "not a bool"),
        ({"pair_threshold": float("nan")}, "finite"),
        ({"pair_threshold": -1.0}, ">= 0"),
        ({"variance_selection_margin": float("inf")}, "finite"),
        ({"var_pair_selection": "typo"}, "must be one of"),
        ({"var_triple_selection": "auto"}, "must be one of"),
        ({"basis_name": "gaussian"}, "must be one of"),
        ({"basis_type": "turbo"}, "must be one of"),
        ({"pair_candidates": "most"}, "must be one of"),
        ({"pair_selection": "bogus"}, "must be one of"),
        ({"triple_selection": "nope"}, "must be one of"),
        ({"precision": "float16"}, "must be one of"),
        ({"heteroscedastic_guard": 1}, "must be a bool"),
        ({"include_linear_1": 0}, "must be a bool"),
        ({"verbose": "yes"}, "must be a bool"),
    ])
    def test_bad_recognized_values(self, cfg, frag):
        base = {"stages": ["A", "B"], "K1": 5, "K2": 3, "strategy": "variance"}
        base.update(cfg)
        with pytest.raises(ValueError, match=frag):
            HiFiANOVATrainer(base)

    @pytest.mark.parametrize("cfg", [
        {"strategy": {"order1": "curvature", "order2": "smoothness",
                      "default": "variance"}},
        {"strategy": "sobolev_2"}, {"strategy": "spectral"},
        {"strategy": "sobolev"}, {"strategy": "spectral_1.5"},
        {"basis_name": "legendre"}, {"basis_type": "full"},
        {"pair_candidates": "either"},
        {"pair_selection": [0, 1, 2]},   # list[int] of active variable indices
        {"pair_selection": "bic"},
        {"triple_selection": "all"}, {"triple_selection": "two_active"},
        {"triple_selection": "one_active"}, {"var_pair_selection": "all"},
        {"heteroscedastic_guard": False}, {"include_linear_1": True},
        {"pair_threshold": 0.0}, {"precision": "float64"},
    ])
    def test_valid_recognized_values(self, cfg):
        base = {"stages": ["A", "B"], "K1": 5, "K2": 3, "strategy": "variance"}
        base.update(cfg)
        HiFiANOVATrainer(base)


class TestBasisPerVariableContainerGuard:
    @pytest.mark.parametrize("bpv", [
        np.array(["auto"]), np.array(["a", "b"]),
    ])
    def test_numpy_array_rejected_actionably(self, bpv):
        with pytest.raises(ValueError, match="'auto' or a mapping"):
            V.validate_basis_per_variable(bpv, D=3)


class TestStrategyDictAndPairSelection:
    """Strategy dicts only work on the uniform mean path; pair_selection is a
    list of active variable indices (range-checked at fit)."""

    def test_strategy_dict_rejected_on_stage_d(self):
        with pytest.raises(ValueError, match="single strategy string"):
            HiFiANOVATrainer({"stages": ["A", "B", "D"], "K1": 5, "K2": 3,
                              "strategy": {"order1": "curvature"}})

    def test_strategy_dict_rejected_on_mixed(self):
        with pytest.raises(ValueError, match="single strategy string"):
            HiFiANOVATrainer({
                "stages": ["A", "B"], "K1": 5, "K2": 3,
                "basis_per_variable": {0: {"basis": "legendre", "K": 3}},
                "strategy": {"order1": "curvature"}})

    def test_strategy_dict_ok_on_mean_only(self):
        HiFiANOVATrainer({"stages": ["A", "B"], "K1": 5, "K2": 3,
                          "strategy": {"order1": "curvature",
                                       "order2": "smoothness"}})

    @pytest.mark.integration
    def test_pair_selection_out_of_range_raises_at_fit(self):
        import jax.numpy as jnp
        X, y = _Xy(n=160, d=3, seed=7)
        x = jnp.asarray(X)
        yj = jnp.asarray(y)
        with pytest.raises(ValueError, match="out of range"):
            HiFiANOVATrainer(
                {"stages": ["A", "B"], "K1": 4, "K2": 2, "strategy": "variance",
                 "pair_selection": [0, 9]}).fit(x, yj, x, yj)

    @pytest.mark.integration
    def test_pair_selection_list_int_fits(self):
        import jax.numpy as jnp
        X, y = _Xy(n=220, d=3, seed=8)
        x = jnp.asarray(X)
        yj = jnp.asarray(y)
        model, _ = HiFiANOVATrainer(
            {"stages": ["A", "B"], "K1": 4, "K2": 2, "strategy": "variance",
             "pair_selection": [0, 1, 2]}).fit(x, yj, x, yj)
        assert model is not None
