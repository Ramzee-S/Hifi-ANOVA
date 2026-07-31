"""Joint mean+variance regularization selection (DEC-012).

Exercises the location-scale smoothing-parameter selector: heteroscedastic
recovery, the false-positive (homoscedastic) case, the leverage-correction guard,
the two criteria (k-fold NLL and joint LAML), the LAML cross-block diagnostic, and
the boundary / df_h tripwire warnings. Test scenarios are chosen so the claim under
test is actually exercised (e.g. an *additive* mean for the homoscedastic case,
where a first-order fit is well-specified — Ishigami is never truly homoscedastic
under a first-order mean because of its x1·x3^4 interaction).
"""
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import pytest

from hifi_anova.data.synthetic import generate_ishigami
from hifi_anova.data.preprocessing import preprocess_data
from hifi_anova.core.features import build_first_order_features, basis_size
from hifi_anova.core.gram import build_gram_matrix
from hifi_anova.training.regularization import (
    build_regularization_vector, build_variance_regularization_vector)
from hifi_anova.training.joint_lambda import (
    optimize_joint_lambda, joint_laml, _joint_fit, _augment)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _feats(X, K):
    return np.asarray(build_first_order_features(
        jnp.asarray(X), K, include_linear=True, basis_name='fourier'), np.float64)


def _mshape(K1):
    return np.asarray(build_regularization_vector(3, K1, 0, 0, 'variance', 1.0, 1.0),
                      np.float64)


def _vshape(Kh):
    return np.asarray(build_variance_regularization_vector(
        3, Kh, 'variance', 1.0, include_linear_h1=True, basis_name='fourier'),
        np.float64)


def _var_sobol(w_h, Kh):
    bs = basis_size(Kh, True, 'fourier')
    G = np.asarray(build_gram_matrix(Kh, True, 'fourier'), np.float64)
    vi = np.array([max(0.0, w_h[i*bs:(i+1)*bs] @ G @ w_h[i*bs:(i+1)*bs])
                   for i in range(3)])
    t = vi.sum()
    return vi / t if t > 0 else vi


def _ishigami_hetero(n=1200, K1=8, Kh=3, seed=0):
    X, y, _ = generate_ishigami(n_samples=n, noise_std=0.0, heteroscedastic=True,
                                variance_variable=2, sigma_min=0.3, sigma_max=3.0,
                                seed=seed)
    d = preprocess_data(X, y, seed=seed)
    xtr = d['x_train']
    ytr = np.asarray(d['y_train'], np.float64)
    return _feats(xtr, K1), _feats(xtr, Kh), ytr, Kh


def _additive(n, seed, sig=1.0):
    """Purely additive mean (first-order-representable) + constant noise."""
    rng = np.random.RandomState(seed)
    X = rng.uniform(0, 1, size=(n, 3))
    f = (np.sin(2*np.pi*X[:, 0]) + 0.7*np.cos(2*np.pi*X[:, 1])
         + 0.5*np.sin(4*np.pi*X[:, 2]))
    y = f + sig * rng.normal(size=n)
    return X, y


# --------------------------------------------------------------------------- #
# 1. Heteroscedastic recovery (the headline benchmark)
# --------------------------------------------------------------------------- #

@pytest.mark.slow
@pytest.mark.parametrize("criterion", ["kfold_nll", "laml"])
def test_ishigami_heteroscedastic_recovery(criterion):
    """On the heteroscedastic Ishigami (x3 drives the variance), the selector puts
    x3 at the top of the variance-Sobol spectrum for both criteria."""
    Phi, Psi, y, Kh = _ishigami_hetero()
    r = optimize_joint_lambda(Phi, Psi, y, _mshape(8), _vshape(Kh),
                              criterion=criterion, n_grid=9, n_folds=4, seed=1)
    s = _var_sobol(r['w_h'], Kh)
    assert int(np.argmax(s)) == 2, f"x3 not dominant: {s}"
    assert s[2] > 0.6, f"x3 variance-Sobol share too low: {s}"
    assert r['lambda_h'] > 0 and np.isfinite(r['df_h'])


@pytest.mark.slow
def test_kfold_and_laml_select_compatible_lambda_h():
    """The two criteria pick lambda_h within ~1.5 decades of each other."""
    Phi, Psi, y, Kh = _ishigami_hetero()
    rk = optimize_joint_lambda(Phi, Psi, y, _mshape(8), _vshape(Kh),
                               criterion='kfold_nll', n_grid=9, n_folds=4, seed=1)
    rl = optimize_joint_lambda(Phi, Psi, y, _mshape(8), _vshape(Kh),
                               criterion='laml', n_grid=9, seed=1)
    assert abs(np.log10(rk['lambda_h']) - np.log10(rl['lambda_h'])) < 1.5


@pytest.mark.slow
def test_selection_beats_fixed_extremes_on_cv_nll():
    """The selected lambda_h achieves a cross-validated NLL no worse than fixed
    lambda_h at either end of the search range (the whole point of selecting it).
    Compared on the same k-fold estimator the selection used, so it is robust to
    the flatness of the NLL valley."""
    Phi, Psi, y, Kh = _ishigami_hetero()
    r = optimize_joint_lambda(Phi, Psi, y, _mshape(8), _vshape(Kh),
                              criterion='kfold_nll', n_grid=9, n_folds=4, seed=1)
    grid = [p for p in r['path']][:9]          # the grid sweep (refine point appended)
    lo_end, hi_end = grid[0]['nll'], grid[-1]['nll']
    assert r['nll'] <= lo_end + 1e-9
    assert r['nll'] <= hi_end + 1e-9
    # and it is a genuine improvement over at least one extreme (interior optimum).
    assert r['nll'] < max(lo_end, hi_end) - 1e-6


# --------------------------------------------------------------------------- #
# 2. False-positive: homoscedastic truth with a well-specified mean
# --------------------------------------------------------------------------- #

@pytest.mark.slow
def test_homoscedastic_false_positive():
    """With constant-variance data and an adequate additive mean, the criterion
    prefers the (near-)homoscedastic limit: lambda_h at the upper bound, df_h -> ~1,
    and the boundary warning fires."""
    X, y = _additive(1600, seed=0, sig=1.0)
    Phi, Psi = _feats(X, 6), _feats(X, 3)
    r = optimize_joint_lambda(Phi, Psi, y, _mshape(6), _vshape(3),
                              criterion='kfold_nll', n_grid=11, n_folds=4, seed=1,
                              lambda_h_bounds=(1e-2, 1e5))
    assert np.log10(r['lambda_h']) > 3.5, "did not push toward homoscedastic"
    assert r['df_h'] < 3.0, f"variance model not collapsed: df_h={r['df_h']}"
    assert any('UPPER bound' in w for w in r['warnings'])
    # Fitted variance recovers the true noise level.
    assert abs(np.mean(r['sigma2']) - 1.0) < 0.25


# --------------------------------------------------------------------------- #
# 3. Leverage-correction guard
# --------------------------------------------------------------------------- #

@pytest.mark.slow
def test_leverage_correction_recovers_sigma2():
    """At high leverage (rich mean, small N), the leverage correction recovers the
    noise variance; without it the variance model underestimates sigma^2 (it fits
    the shrunk residuals E[r^2] ~ sigma^2 (1-lev))."""
    X, y = _additive(260, seed=0, sig=1.0)
    Phi, Psi = _feats(X, 8), _feats(X, 2)      # 51-col mean on ~180 train rows
    m, v = _mshape(8), _vshape(2)
    r_on = optimize_joint_lambda(Phi, Psi, y, m, v, criterion='kfold_nll',
                                 n_grid=9, n_folds=4, seed=1, leverage_correct=True)
    r_off = optimize_joint_lambda(Phi, Psi, y, m, v, criterion='kfold_nll',
                                  n_grid=9, n_folds=4, seed=1, leverage_correct=False)
    s_on, s_off = np.mean(r_on['sigma2']), np.mean(r_off['sigma2'])
    assert s_off < s_on, "leverage-off did not underestimate relative to on"
    assert abs(s_on - 1.0) < abs(s_off - 1.0), "correction did not move toward truth"


# --------------------------------------------------------------------------- #
# 4. LAML cross-block diagnostic
# --------------------------------------------------------------------------- #

def test_laml_cross_block_diagnostic():
    """joint_laml returns a finite log-evidence and a cross-block ratio; the exact
    cross-block changes the value (it is nonzero in finite samples) but does not
    blow it up."""
    Phi, Psi, y, Kh = _ishigami_hetero(n=800)
    m, v = _mshape(8), _vshape(Kh)
    Pa = _augment(Phi)
    reg_m = np.concatenate([[0.0], 1.0 * m])
    fit = _joint_fit(Pa, Psi, y, reg_m, 10.0 * v)
    L_block = joint_laml(fit, cross=False)
    L_full = joint_laml(fit, cross=True)
    assert np.isfinite(L_block['laml']) and np.isfinite(L_full['laml'])
    assert L_block['laml'] != L_full['laml']
    assert 0.0 <= L_block['cross_ratio'] < 10.0
    assert abs(L_full['laml'] - L_block['laml']) < abs(L_block['laml'])


# --------------------------------------------------------------------------- #
# 5. Tripwires, guards, and API
# --------------------------------------------------------------------------- #

@pytest.mark.slow
def test_df_h_tripwire_warns_on_rich_variance_model():
    """A rich variance model (Kh=8, 51 cols) forced into the low-lambda_h regime on
    modest N trips the df_h > N/10 diagnostic."""
    Phi, Psi, y, Kh = _ishigami_hetero(n=500, Kh=8)   # 51-col variance model
    r = optimize_joint_lambda(Phi, Psi, y, _mshape(8), _vshape(8),
                              criterion='kfold_nll', n_grid=7, n_folds=3, seed=1,
                              lambda_h_bounds=(1e-3, 1e0))
    assert any('df_h' in w for w in r['warnings'])


def test_criterion_validation():
    Phi, Psi, y, Kh = _ishigami_hetero(n=400)
    with pytest.raises(ValueError):
        optimize_joint_lambda(Phi, Psi, y, _mshape(8), _vshape(Kh),
                              criterion='bogus')


def test_fixed_mean_lambda_and_split_mode():
    """fixed_lambda_mean is honored; split_nll runs as the cheap single-holdout."""
    Phi, Psi, y, Kh = _ishigami_hetero(n=600)
    r = optimize_joint_lambda(Phi, Psi, y, _mshape(8), _vshape(Kh),
                              criterion='split_nll', fixed_lambda_mean=0.05,
                              n_grid=7, seed=1)
    assert r['lambda_mean'] == pytest.approx(0.05)
    assert r['n_folds'] == 1


@pytest.mark.slow
def test_hyperprior_pulls_lambda_h():
    """A weak MAP-II hyperprior centered below the data optimum shifts lambda_h
    downward relative to no prior."""
    Phi, Psi, y, Kh = _ishigami_hetero(n=800)
    m, v = _mshape(8), _vshape(Kh)
    r0 = optimize_joint_lambda(Phi, Psi, y, m, v, criterion='kfold_nll',
                               n_grid=9, n_folds=3, seed=1)
    r1 = optimize_joint_lambda(Phi, Psi, y, m, v, criterion='kfold_nll',
                               n_grid=9, n_folds=3, seed=1,
                               hyperprior=(np.log10(r0['lambda_h']) - 2.0, 0.5))
    assert r1['lambda_h'] < r0['lambda_h']
