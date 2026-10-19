"""Multi-backend tests for the Student-t likelihood and ResidualFitter failure handling.

Run once per backend (the CI matrix sets ``KERAS_BACKEND``).

Background (2026-10): on a cluster checkout predating 9cda334, ``_lgamma``
did ``float(ops.convert_to_numpy(x))`` + ``scipy.special.gammaln``.

* Under JAX this raised ``TracerArrayConversionError`` inside
  ``jax.value_and_grad``. ``ResidualFitter.fit`` caught *every* exception,
  so each step's cost was inf and the parameters never moved. The fit
  "completed" and returned the initial Ω (ρ=0.1, dof=init) without
  reporting anything.
* Under TF the lgamma terms were constants, so dof was fitted with a
  biased gradient (the missing ψ terms).

The tests below check that the lgamma value and gradient are correct on
every backend, that a realistic Student-t fit moves and recovers dof, and
that ResidualFitter no longer swallows errors.
"""
import warnings

import numpy as np
import pandas as pd
import pytest
import keras
from keras import ops
from scipy.special import gammaln, digamma
from scipy.stats import multivariate_t

BACKEND = keras.backend.backend()


# ---------------------------------------------------------------------------
# lgamma: value and gradient
# ---------------------------------------------------------------------------

# Spans small values and the realistic Student-t arguments: (ν + k) / 2
# with k ~ 2000 voxels gives ~1000.
LGAMMA_X = np.array([0.05, 0.3, 0.5, 1.0, 1.7, 3.5, 7.0, 12.5, 50.0,
                     250.0, 1003.5, 5000.0], dtype=np.float32)


def test_lgamma_matches_scipy_gammaln():
    from braincoder.utils.backend import _lgamma, lgamma
    expected = gammaln(LGAMMA_X.astype(np.float64))
    for fn in (_lgamma, lgamma):
        got = np.asarray(ops.convert_to_numpy(
            fn(ops.convert_to_tensor(LGAMMA_X))), dtype=np.float64)
        np.testing.assert_allclose(got, expected, rtol=2e-5, atol=2e-5,
                                   err_msg=f'{fn.__name__} on {BACKEND}')


def test_lgamma_preserves_float32():
    from braincoder.utils.backend import _lgamma
    out = _lgamma(ops.convert_to_tensor(LGAMMA_X))
    assert 'float32' in str(out.dtype)


def test_lgamma_gradient_matches_digamma():
    """d lgamma(x)/dx = ψ(x), elementwise, through compute_gradients
    (i.e. inside jax.value_and_grad / tf.GradientTape / torch autograd)."""
    from braincoder.utils.backend import _lgamma, compute_gradients
    v = keras.Variable(LGAMMA_X.copy())
    _, grads = compute_gradients(lambda: ops.sum(_lgamma(v.value)), [v])
    g = np.asarray(ops.convert_to_numpy(grads[0]), dtype=np.float64)
    np.testing.assert_allclose(g, digamma(LGAMMA_X.astype(np.float64)),
                               rtol=1e-3, atol=1e-3,
                               err_msg=f'lgamma gradient on {BACKEND}')


def test_lgamma_inside_jit():
    """The original failure: host conversion inside a jitted trace."""
    if BACKEND != 'jax':
        pytest.skip('JAX-specific')
    import jax
    from braincoder.utils.backend import _lgamma
    f = jax.jit(lambda x: _lgamma(x))
    np.testing.assert_allclose(np.asarray(f(LGAMMA_X)),
                               gammaln(LGAMMA_X.astype(np.float64)),
                               rtol=2e-5, atol=2e-5)


# ---------------------------------------------------------------------------
# mvt_log_prob: value and dof gradient vs scipy
# ---------------------------------------------------------------------------

def _random_cov(k, rng):
    A = rng.standard_normal((k, k))
    return A @ A.T / k + np.eye(k)


def test_mvt_log_prob_matches_scipy():
    from braincoder.utils.backend import mvt_log_prob
    rng = np.random.default_rng(1)
    k, n, dof = 5, 40, 6.5
    cov = _random_cov(k, rng)
    x = multivariate_t(shape=cov, df=dof).rvs(n, random_state=rng)
    L = np.linalg.cholesky(cov)
    got = ops.convert_to_numpy(mvt_log_prob(
        ops.convert_to_tensor(x.astype(np.float32)),
        ops.convert_to_tensor(L.astype(np.float32)), dof))
    expected = multivariate_t(shape=cov, df=dof).logpdf(x)
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-3)


@pytest.mark.parametrize('k', [3, 200])
def test_mvt_log_prob_dof_gradient_matches_finite_difference(k):
    """∂/∂ν Σ log p(x | ν) must include 0.5[ψ((ν+k)/2) − ψ(ν/2)] per
    observation. With the old constant lgamma this term is lost (it is
    ~+2.9 per observation at k=2000, ν=7)."""
    from braincoder.utils.backend import mvt_log_prob, compute_gradients
    rng = np.random.default_rng(2)
    n, dof = 30, 7.0
    cov = np.eye(k)
    x = multivariate_t(shape=cov, df=dof).rvs(n, random_state=rng)

    def ll64(nu):
        return multivariate_t(shape=cov, df=nu).logpdf(x).sum()

    h = 1e-4
    fd = (ll64(dof + h) - ll64(dof - h)) / (2 * h)

    x_t = ops.convert_to_tensor(x.astype(np.float32))
    L_t = ops.convert_to_tensor(np.eye(k, dtype=np.float32))
    nu_var = keras.Variable(np.asarray([dof], dtype=np.float32))
    _, grads = compute_gradients(
        lambda: ops.sum(mvt_log_prob(x_t, L_t, nu_var.value[0])), [nu_var])
    g = float(ops.convert_to_numpy(grads[0])[0])

    # The missing lgamma contribution, for scale.
    missing = n * 0.5 * (digamma((dof + k) / 2) - digamma(dof / 2))
    assert abs(g - fd) < 0.05 * abs(missing) + 1e-2, (
        f'dof gradient {g:.4f} vs finite difference {fd:.4f} '
        f'(lgamma contribution {missing:.4f}) on {BACKEND}')


# ---------------------------------------------------------------------------
# ResidualFitter: recovery on synthetic Student-t residuals
# ---------------------------------------------------------------------------

def _make_model(n_vox=12, seed=0):
    from braincoder.models import GaussianPRF
    paradigm = np.linspace(-3, 3, 60, dtype=np.float32)[:, np.newaxis]
    pars = pd.DataFrame({
        'mu':        np.linspace(-2.5, 2.5, n_vox, dtype=np.float32),
        'sd':        np.full(n_vox, 1.2, dtype=np.float32),
        'amplitude': np.ones(n_vox, dtype=np.float32),
        'baseline':  np.zeros(n_vox, dtype=np.float32),
    })
    model = GaussianPRF(paradigm=paradigm, parameters=pars)
    model.init_pseudoWWT(paradigm, pars)
    return model, paradigm, pars


def _student_t_residuals(model, n_t, rho, sigma2, dof, seed=0):
    """Draw residuals ~ MVT(dof, 0, Ω) with Ω built exactly as
    ResidualFitter._get_omega builds it (normalised pseudo-WWT)."""
    rng = np.random.default_rng(seed)
    WWT = np.asarray(ops.convert_to_numpy(model.get_pseudoWWT()), dtype=np.float64)
    WWT = WWT / WWT.mean()
    n_vox = WWT.shape[0]
    tau = rng.uniform(0.8, 1.2, n_vox)
    omega = (rho * np.outer(tau, tau) + (1 - rho) * np.diag(tau ** 2)
             + sigma2 * WWT)
    resid = multivariate_t(shape=omega, df=dof).rvs(n_t, random_state=rng)
    return resid.astype(np.float32), omega


def test_residual_fitter_student_t_recovers_dof():
    from braincoder.optimize import ResidualFitter
    model, paradigm, pars = _make_model()
    true_rho, true_sigma2, true_dof = 0.4, 0.3, 5.0
    resid, true_omega = _student_t_residuals(model, 1500, true_rho,
                                             true_sigma2, true_dof)
    data = pd.DataFrame(resid[:60])  # only used for shapes
    fitter = ResidualFitter(model, data, paradigm=paradigm, parameters=pars)
    init_rho, init_sigma2, init_dof = 0.1, 1e-3, 1.0
    omega, dof = fitter.fit(method='t', residuals=resid,
                            init_rho=init_rho, init_sigma2=init_sigma2,
                            init_dof=init_dof, learning_rate=0.05,
                            max_n_iterations=600, min_n_iterations=100,
                            progressbar=False)
    p = fitter.fitted_omega_parameters
    costs = fitter.costs[:np.flatnonzero(fitter.costs)[-1] + 1]

    assert np.all(np.isfinite(costs)), f'non-finite costs on {BACKEND}'
    assert fitter.n_failed_steps == 0
    assert costs[-1] < costs[0]
    # Parameters moved away from their initial values ...
    assert abs(float(p['rho']) - init_rho) > 0.1
    assert abs(float(p['sigma2']) - init_sigma2) > 0.05
    assert abs(dof - init_dof) > 1.0
    # ... and land near the truth.
    assert abs(float(p['rho']) - true_rho) < 0.15, p
    assert 3.0 < dof < 8.5, f'dof={dof:.2f} (true {true_dof}) on {BACKEND}'
    assert np.all(np.isfinite(omega))


# ---------------------------------------------------------------------------
# ResidualFitter: errors must not be swallowed
# ---------------------------------------------------------------------------

@pytest.fixture
def small_fitter():
    from braincoder.optimize import ResidualFitter
    model, paradigm, pars = _make_model(n_vox=5)
    resid, _ = _student_t_residuals(model, 200, 0.3, 0.2, 6.0, seed=3)
    data = pd.DataFrame(resid[:60])
    return ResidualFitter(model, data, paradigm=paradigm, parameters=pars), resid


def test_residual_fitter_reraises_programming_error(small_fitter, monkeypatch):
    import braincoder.optimize.residual_fitter as rf
    fitter, resid = small_fitter

    def broken(*args, **kwargs):
        raise TypeError('bug in likelihood')

    monkeypatch.setattr(rf, 'mvt_log_prob', broken)
    with pytest.raises(TypeError, match='bug in likelihood'):
        fitter.fit(method='t', residuals=resid, max_n_iterations=50,
                   use_wwt=False, progressbar=False)


def test_residual_fitter_reraises_old_lgamma_bug(small_fitter, monkeypatch):
    """Reproduce the original failure: host conversion of a traced value.
    On JAX that is TracerArrayConversionError (a TypeError) and must
    propagate instead of producing a frozen, 'successful' fit."""
    if BACKEND != 'jax':
        pytest.skip('host conversion only fails under JAX tracing')
    import braincoder.optimize.residual_fitter as rf
    from braincoder.utils.backend import mvt_log_prob
    fitter, resid = small_fitter

    def old_style(x, L, dof):
        float(ops.convert_to_numpy(dof))   # what the old _lgamma did
        return mvt_log_prob(x, L, dof)

    monkeypatch.setattr(rf, 'mvt_log_prob', old_style)
    with pytest.raises(TypeError):
        fitter.fit(method='t', residuals=resid, max_n_iterations=50,
                   use_wwt=False, progressbar=False)


def test_residual_fitter_raises_after_consecutive_numerical_failures(
        small_fitter, monkeypatch):
    import braincoder.optimize.residual_fitter as rf
    fitter, resid = small_fitter

    def always_fails(*args, **kwargs):
        raise RuntimeError('cholesky failed')

    monkeypatch.setattr(rf, 'mvn_log_prob', always_fails)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        with pytest.raises(RuntimeError, match='consecutive'):
            fitter.fit(method='gauss', residuals=resid, max_n_iterations=200,
                       max_consecutive_failures=7, use_wwt=False,
                       progressbar=False)
    assert fitter.n_failed_steps == 7


def test_residual_fitter_raises_on_persistent_nan(small_fitter, monkeypatch):
    import braincoder.optimize.residual_fitter as rf
    fitter, resid = small_fitter
    real = rf.mvn_log_prob

    def nan_ll(x, L):
        return real(x, L) * np.float32(np.nan)

    monkeypatch.setattr(rf, 'mvn_log_prob', nan_ll)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        with pytest.raises(RuntimeError):
            fitter.fit(method='gauss', residuals=resid, max_n_iterations=200,
                       max_consecutive_failures=10, use_wwt=False,
                       progressbar=False)


def test_residual_fitter_never_finite_raises(small_fitter, monkeypatch):
    """Fewer iterations than the consecutive-failure limit: the
    'no finite cost ever' check must still catch it."""
    import braincoder.optimize.residual_fitter as rf
    fitter, resid = small_fitter

    def always_fails(*args, **kwargs):
        raise RuntimeError('cholesky failed')

    monkeypatch.setattr(rf, 'mvn_log_prob', always_fails)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        with pytest.raises(RuntimeError, match='finite cost'):
            fitter.fit(method='gauss', residuals=resid, max_n_iterations=5,
                       use_wwt=False, progressbar=False)


def test_residual_fitter_transient_failures_back_off_and_recover(
        small_fitter, monkeypatch):
    """Legitimate divergence handling stays: a few numerical failures are
    rolled back with a smaller learning rate, a warning is issued, and the
    fit completes."""
    import braincoder.optimize.residual_fitter as rf
    fitter, resid = small_fitter
    real = rf.mvn_log_prob
    calls = {'n': 0}

    def flaky(x, L):
        calls['n'] += 1
        if calls['n'] in (5, 6, 7):
            raise RuntimeError('transient cholesky failure')
        return real(x, L)

    monkeypatch.setattr(rf, 'mvn_log_prob', flaky)
    with pytest.warns(RuntimeWarning, match='rolled back'):
        omega, _ = fitter.fit(method='gauss', residuals=resid,
                              max_n_iterations=60, min_n_iterations=20,
                              use_wwt=False, progressbar=False)
    assert fitter.n_failed_steps == 3
    assert np.all(np.isfinite(omega))
