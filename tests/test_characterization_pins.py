"""Characterization pins for the two golden-master blind spots (refactor prep).

The 2026-08-07 coverage audit (X2Code refactor session) found two paths with
NO golden and NO unit coverage:

  1. ``trainer.estimate_sobol`` — the standalone minimal-regularization Sobol
     estimator (never referenced by any test), and
  2. ``mode='auto'`` — the progressive stage-decision path (no functional fit
     test).

These tests pin both value-exactly on fixed inputs so the trainer/sobol
refactor cannot silently change them, plus the degenerate ``total_var <= 0``
fallback of ``compute_sobol_indices`` (also previously unpinned, including its
exact degenerate output *shape*).

Reference values were generated on ws2 (the primary dev runner). Tolerances:
float64 analytics are compared at rtol=1e-5 — above the measured cross-host
drift of float32-fit-derived analytics (<=3.2e-6 relative, DEC-035 follow-up)
and orders of magnitude below any real behavior change. Stage decisions and
output shapes are compared exactly.
"""

import numpy as np
import pytest

from hifi_anova.data.synthetic import generate_ishigami
from hifi_anova.data.preprocessing import preprocess_data
from hifi_anova.training.trainer import HiFiANOVATrainer, estimate_sobol
from hifi_anova.analysis.sobol import compute_sobol_indices

RTOL = 1e-5
ATOL = 1e-8

# Reference values generated on ws2 (see module docstring).
REF = {
    "fixed": {
        "s1": {"0": 0.3159123862793728, "1": 0.44326308567137956,
               "2": 5.548188551894817e-05},
        "s2": {"(0, 1)": 0.00016276515328520085,
               "(0, 2)": 0.2392106841055888,
               "(1, 2)": 0.0006053107421438653},
        "s3": {"(0, 1, 2)": 0.000790286162710866},
        "st": {"0": 0.5560761217009577, "1": 0.44482144772951954,
               "2": 0.2406617628959625},
        "total_model_variance": 13.699285620530556,
        "additivity_sum": 0.9874655571766382,
        "f0": 3.539129762404588,
    },
    "auto_lambda": {
        "lambda_order1": 1.0066557601624901e-10,
        "additivity_sum": 0.9908797747549981,
        "s1": {"0": 0.315544138154405, "1": 0.4424612145321722,
               "2": 4.462638865432278e-05},
    },
    "auto_stop_B": {
        "stages_run": ["stage_A", "stage_B"],
        "has_variance_model": False,
        "has_residual_net": False,
        "mean_first": {"0": 0.3212035728938409, "1": 0.4343662111854623,
                       "2": 8.140160295529664e-05},
        "mean_total": {"0": 0.564932442190496, "1": 0.43524931113306176,
                       "2": 0.2441670609941838},
    },
    "auto_add_D": {
        # Re-pinned at the Stage-D default flip (joint-GLS mean + mean-consistent
        # guard + min_noise_ratio=1e-2). mean_first shifts ~1e-3 (the mean is now
        # the joint-GLS solution); var_first is essentially unchanged.
        "stages_run": ["stage_A", "stage_B", "stage_D"],
        "has_variance_model": True,
        "var_first": {"0": 0.010719517423194776, "1": 0.009029124905416654,
                      "2": 0.9802513576713885},
        "mean_first": {"0": 0.31485481556369743, "1": 0.41324535250704314,
                       "2": 0.0031117702054866163},
    },
}


def _assert_block_close(actual: dict, expected: dict, rtol=RTOL, atol=ATOL):
    """Exact key-set match + numeric closeness (keys compared as str)."""
    actual_s = {str(k): float(v) for k, v in actual.items()}
    assert set(actual_s) == set(expected), (
        f"key sets differ: {sorted(actual_s)} vs {sorted(expected)}")
    for k, v_exp in expected.items():
        assert actual_s[k] == pytest.approx(v_exp, rel=rtol, abs=atol), (
            f"key {k}: {actual_s[k]} != {v_exp}")


class TestEstimateSobolPins:
    """Value-exact pins for the standalone estimate_sobol (blind spot #1)."""

    def test_fixed_lambda_with_third_order(self):
        ref = REF["fixed"]
        # estimate_sobol expects inputs on [0,1]: scale the Ishigami cube.
        Xi, yi, _ = generate_ishigami(n_samples=2000, noise_std=0.1, seed=0)
        Xi01 = (Xi + np.pi) / (2 * np.pi)
        r = estimate_sobol(Xi01, yi, K1=8, K2=4, K3=2, auto_lambda=False,
                           lambda1=1e-6, lambda2=1e-5, lambda3=1e-4)
        _assert_block_close(r["sobol_first_order"], ref["s1"])
        _assert_block_close(r["sobol_second_order"], ref["s2"])
        _assert_block_close(r["sobol_third_order"], ref["s3"])
        _assert_block_close(r["sobol_total_order"], ref["st"])
        assert r["total_model_variance"] == pytest.approx(
            ref["total_model_variance"], rel=RTOL)
        assert r["additivity_sum"] == pytest.approx(
            ref["additivity_sum"], rel=RTOL)
        assert r["f0"] == pytest.approx(ref["f0"], rel=RTOL)
        assert r["mode"] == "sobol_estimation"
        # Sanity beyond the pin: near-minimal regularization on [0,1] inputs
        # recovers the analytic Ishigami spectrum (S1≈.314, S2≈.442, S13≈.244)
        # and the additivity criterion (~1).
        assert r["additivity_sum"] == pytest.approx(1.0, abs=0.05)

    def test_auto_lambda_additivity_search(self):
        """Pins the scipy minimize_scalar additivity-criterion path."""
        ref = REF["auto_lambda"]
        Xi, yi, _ = generate_ishigami(n_samples=2000, noise_std=0.1, seed=0)
        Xi01 = (Xi + np.pi) / (2 * np.pi)
        r = estimate_sobol(Xi01, yi, K1=8, K2=4, auto_lambda=True)
        # The optimizer's argmin is pinned a bit looser than the analytics
        # (bounded Brent stops on an xatol, not machine precision).
        assert r["lambda_order1"] == pytest.approx(
            ref["lambda_order1"], rel=1e-3)
        assert r["additivity_sum"] == pytest.approx(
            ref["additivity_sum"], rel=1e-4)
        _assert_block_close(r["sobol_first_order"], ref["s1"],
                            rtol=1e-4, atol=1e-7)


class TestAutoModePins:
    """Functional pins for mode='auto' stage decisions (blind spot #2)."""

    @pytest.mark.integration
    def test_auto_stops_after_B_low_noise(self):
        """Low-noise homoscedastic Ishigami: auto adds B (pair structure),
        then stops — residual 0.5% ≤ 1% threshold (no Stage C) and
        max |corr(r², x)| ≈ 0.075 ≤ 0.1 (no Stage D). Chosen for robust
        decision margins (a rich-basis Friedman1 variant sat at corr ≈ 0.101
        — a knife-edge no characterization test should stand on)."""
        ref = REF["auto_stop_B"]
        Xi, yi, _ = generate_ishigami(n_samples=2000, noise_std=0.1, seed=0)
        data = preprocess_data(Xi, yi, seed=0)
        model, res = HiFiANOVATrainer(
            {"mode": "auto", "K1": 8, "K2": 4, "strategy": "variance",
             "lambda_order1": 1e-3, "lambda_order2": 1e-2,
             "verbose": False}).fit(
            data["x_train"], data["y_train"], data["x_val"], data["y_val"])

        stages_run = sorted(k for k in res if k.startswith("stage_"))
        assert stages_run == ref["stages_run"]
        assert (model.variance_model is not None) == ref["has_variance_model"]
        assert (model.residual_net is not None) == ref["has_residual_net"]

        sob = compute_sobol_indices(model, data["x_test"])
        _assert_block_close(sob["mean_sobol"]["first_order"],
                            ref["mean_first"], rtol=1e-4, atol=1e-7)
        _assert_block_close(sob["mean_sobol"]["total_order"],
                            ref["mean_total"], rtol=1e-4, atol=1e-7)

    @pytest.mark.integration
    def test_auto_adds_stage_D_heteroscedastic(self):
        """Hetero Ishigami: auto skips C (threshold 0.35 > residual fraction)
        but detects corr(r², x3) and adds Stage D."""
        ref = REF["auto_add_D"]
        Xh, yh, _ = generate_ishigami(n_samples=2500, heteroscedastic=True,
                                      seed=0)
        dh = preprocess_data(Xh, yh, seed=0)
        mh, rh = HiFiANOVATrainer(
            {"mode": "auto", "auto_threshold": 0.35, "K1": 8, "K2": 4,
             "strategy": "variance", "lambda_order1": 1e-3,
             "lambda_order2": 1e-2, "lambda_h": 0.1, "max_outer_iter": 8,
             "verbose": False}).fit(
            dh["x_train"], dh["y_train"], dh["x_val"], dh["y_val"])

        stages_run = sorted(k for k in rh if k.startswith("stage_"))
        assert stages_run == ref["stages_run"]
        assert (mh.variance_model is not None) == ref["has_variance_model"]

        sobh = compute_sobol_indices(mh, dh["x_test"])
        _assert_block_close(sobh["mean_sobol"]["first_order"],
                            ref["mean_first"], rtol=1e-4, atol=1e-7)
        if ref["var_first"] is not None:
            assert "log_variance_sobol" in sobh
            _assert_block_close(sobh["log_variance_sobol"]["first_order"],
                                ref["var_first"], rtol=1e-4, atol=1e-7)


class TestDegenerateSobolFallback:
    """Pins the total_var<=0 / v_core<=0 branches of compute_sobol_indices —
    including the exact (asymmetric) degenerate output shapes."""

    def test_constant_target_all_zero(self):
        # preprocess_data rejects a constant target, so drive the trainer
        # directly on [0,1] inputs: y ≡ 0 → w = 0 exactly → total_var = 0.
        rng = np.random.RandomState(0)
        x = rng.uniform(0.0, 1.0, size=(400, 3))
        y = np.zeros(400)
        model, _ = HiFiANOVATrainer(
            {"K1": 4, "strategy": "uniform", "lambda_order1": 1e-3,
             "stages": ["A"], "verbose": False}).fit(
            x[:300], y[:300], x[300:], y[300:])
        sob = compute_sobol_indices(model, x[:100])

        D = 3
        ms = sob["mean_sobol"]
        assert ms["first_order"] == {i: 0.0 for i in range(D)}
        assert ms["residual"] == 0.0
        assert ms["second_order"] == {}
        # total_order IS populated (with zeros) in the degenerate case...
        assert ms["total_order"] == {i: 0.0 for i in range(D)}
        # ...but the core block's total_order stays EMPTY (current behavior,
        # frozen here so a refactor cannot silently "fix" the asymmetry).
        core = sob["mean_sobol_core"]
        assert core["first_order"] == {i: 0.0 for i in range(D)}
        assert core["total_order"] == {}
        assert sob["mean_sobol_total"]["total_order"] == {}
        assert sob["fidelity"]["value"] == 1.0
        va = sob["variance_accounting"]
        assert va["total_model_variance"] == 0.0
