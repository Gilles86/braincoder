"""Tests for the GP prior and Bayesian fitter (Daghlian et al. 2025).

Three layers of coverage:
  * log_prob matches a manual numpy MVN log-prob (math sanity)
  * hyperparameter MLE recovers known (lengthscale, variance, nugget)
    from samples drawn from a known GP
  * BayesianParameterFitter recovers smooth-spatial-structure parameters
    better than classical at high noise (a mini version of paper Fig. 2)
"""
import numpy as np
import pandas as pd
import pytest

from braincoder.models import GaussianPRF
from braincoder.optimize.gp_prior import GeodesicGPPrior
from braincoder.optimize.bayesian_fitter import BayesianParameterFitter
from braincoder.utils.cortex import geodesic_distance_matrix


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def numpy_mvn_log_prob(values, K):
    """Reference MVN(0, K) log-prob via numpy."""
    L = np.linalg.cholesky(K)
    y = np.linalg.solve(L, values)
    mahal = float(y @ y)
    log_det = 2.0 * float(np.sum(np.log(np.diag(L))))
    n = len(values)
    return -0.5 * (mahal + log_det + n * np.log(2.0 * np.pi))


def line_distance_matrix(n, span=10.0):
    """Pairwise |x_i - x_j| for n equally spaced points on a line."""
    x = np.linspace(0.0, span, n)
    return np.abs(x[:, None] - x[None, :])


# ---------------------------------------------------------------------------
# Math sanity
# ---------------------------------------------------------------------------

def test_log_prob_matches_numpy_mvn():
    rng = np.random.default_rng(0)
    n = 25
    d = line_distance_matrix(n)
    l, v, nug = 2.5, 1.7, 0.2
    K_ref = v * np.exp(-d ** 2 / (2.0 * l ** 2)) + nug * np.eye(n)

    prior = GeodesicGPPrior(d, lengthscale_init=l, variance_init=v,
                            nugget_init=nug, jitter=0.0)

    values = rng.standard_normal(n).astype(np.float32)
    expected = numpy_mvn_log_prob(values.astype(np.float64),
                                  K_ref.astype(np.float64))
    got = float(prior.log_prob(values))

    assert np.isclose(got, expected, rtol=1e-3, atol=1e-3), \
        f"log_prob mismatch: got {got}, expected {expected}"


def test_freeze_cholesky_matches_live_log_prob():
    """Cached-Cholesky path must give the same log_prob as the live path.

    The cached path is used during fit_map to keep TF's CholeskyGrad
    out of the gradient graph. Sanity-check that it produces identical
    values, otherwise MAP fits would silently use a different prior.
    """
    rng = np.random.default_rng(7)
    n = 30
    d = line_distance_matrix(n, span=15.0)
    prior = GeodesicGPPrior(d, lengthscale_init=2.0, variance_init=1.5,
                            nugget_init=0.1)

    values = rng.standard_normal(n).astype(np.float32)
    live = float(prior.log_prob(values))

    prior.freeze_cholesky()
    cached = float(prior.log_prob(values))
    assert np.isclose(live, cached, rtol=1e-5, atol=1e-5), \
        f"cached log_prob {cached} differs from live {live}"

    prior.unfreeze_cholesky()
    assert prior._cached_L is None
    again = float(prior.log_prob(values))
    assert np.isclose(again, live, rtol=1e-5, atol=1e-5)


def test_smooth_values_have_higher_log_prob_than_noise():
    rng = np.random.default_rng(1)
    n = 40
    d = line_distance_matrix(n, span=20.0)
    prior = GeodesicGPPrior(d, lengthscale_init=3.0, variance_init=1.0,
                            nugget_init=0.05)

    x = np.linspace(0, 20, n)
    smooth = np.cos(0.5 * x).astype(np.float32)
    noise = rng.standard_normal(n).astype(np.float32)

    assert float(prior.log_prob(smooth)) > float(prior.log_prob(noise))


# ---------------------------------------------------------------------------
# Hyperparameter MLE
# ---------------------------------------------------------------------------

def test_hyperparameter_mle_recovers_truth():
    rng = np.random.default_rng(42)
    n = 80
    d = line_distance_matrix(n, span=20.0)

    true_l, true_v, true_nug = 4.0, 2.0, 0.1
    K = true_v * np.exp(-d ** 2 / (2 * true_l ** 2)) + true_nug * np.eye(n)
    L = np.linalg.cholesky(K + 1e-6 * np.eye(n))

    n_samples = 8
    samples = (L @ rng.standard_normal((n, n_samples))).astype(np.float32)

    ls, vs, nugs = [], [], []
    for s in range(n_samples):
        prior = GeodesicGPPrior(d, lengthscale_init=1.0, variance_init=0.5,
                                nugget_init=0.5)
        prior.fit_hyperparameters(samples[:, s], max_n_iterations=400,
                                  learning_rate=0.05, progressbar=False)
        ls.append(prior.lengthscale)
        vs.append(prior.variance)
        nugs.append(prior.nugget)

    # Lengthscale should be within a factor of 2 of truth — MLE on 80 points
    # from a single GP draw has wide error bars.
    assert 0.5 * true_l < np.mean(ls) < 2.0 * true_l, np.mean(ls)
    # Variance has the largest bias under finite samples; allow wide window.
    assert 0.3 * true_v < np.mean(vs) < 3.0 * true_v, np.mean(vs)
    # Nugget is typically the best-recovered.
    assert abs(np.mean(nugs) - true_nug) < 0.15, np.mean(nugs)


# ---------------------------------------------------------------------------
# Cortical surface helpers
# ---------------------------------------------------------------------------

def test_mesh_dijkstra_unit_square():
    """Mesh-graph Dijkstra on a unit square produces the expected distances.

    Catches the coo→csr-summing bug where interior edges would get
    double-counted, doubling weights.
    """
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float)
    f = np.array([[0, 1, 2], [1, 3, 2]], dtype=int)
    D = geodesic_distance_matrix(v, f, progressbar=False)

    # (0,0)-(1,0) adjacent → 1; (1,0)-(0,1) diagonal edge → √2;
    # (0,0)-(1,1) no direct edge → 2 (via either neighbour).
    expected = np.array([
        [0.0, 1.0, 1.0, 2.0],
        [1.0, 0.0, np.sqrt(2), 1.0],
        [1.0, np.sqrt(2), 0.0, 1.0],
        [2.0, 1.0, 1.0, 0.0],
    ])
    assert np.allclose(D, expected, atol=1e-5), D


def test_mesh_dijkstra_grid_ratio_bounded():
    """On a regular triangulated grid mesh/euclidean ratio is bounded by √2."""
    n = 25
    xs, ys = np.meshgrid(np.linspace(0, 10, n), np.linspace(0, 10, n))
    verts = np.stack([xs.ravel(), ys.ravel(),
                      np.zeros_like(xs).ravel()], axis=1).astype(np.float32)
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j; b = a + 1; c = a + n; d = c + 1
            faces.append([a, b, c]); faces.append([b, d, c])
    faces = np.array(faces, dtype=np.int64)

    rng = np.random.default_rng(0)
    idx = rng.choice(len(verts), size=80, replace=False)
    D = geodesic_distance_matrix(verts, faces, source_indices=idx,
                                 progressbar=False)

    eucl = np.sqrt(((verts[idx][:, None] - verts[idx][None, :]) ** 2).sum(-1))
    ratio = D[D > 0] / eucl[D > 0]
    assert ratio.min() >= 1.0 - 1e-4
    assert ratio.max() <= np.sqrt(2) + 1e-4
    # Symmetric, zero diagonal.
    assert np.allclose(D, D.T)
    assert np.allclose(np.diag(D), 0.0)


# ---------------------------------------------------------------------------
# Mini paper Fig. 2: parameter recovery wins under noise
# ---------------------------------------------------------------------------

def test_no_prior_ml_runs_and_recovers_params():
    """Empty-priors mode = pure ML with per-vertex sigma. Smoke test only."""
    rng = np.random.default_rng(8)
    n_vx = 15
    true_mu = np.linspace(-2, 2, n_vx).astype(np.float32)

    paradigm = pd.DataFrame({'x': np.linspace(-5, 5, 60, dtype=np.float32)})
    true_pars = pd.DataFrame({
        'mu': true_mu,
        'sd': np.full(n_vx, 1.0, dtype=np.float32),
        'amplitude': np.full(n_vx, 2.0, dtype=np.float32),
        'baseline': np.zeros(n_vx, dtype=np.float32),
    })
    model = GaussianPRF(paradigm=paradigm, parameters=true_pars)
    clean = model.predict(paradigm=paradigm, parameters=true_pars)
    noise = rng.standard_normal(clean.shape).astype(np.float32) * 0.3
    data = pd.DataFrame(clean.values + noise, columns=true_pars.index)

    fitter = BayesianParameterFitter(model, data, paradigm, priors={})
    fitter.classical_estimates = true_pars.copy()   # init from truth
    fitter.fit_map(max_n_iterations=100, progressbar=False)

    # Should stay close to truth at this noise level.
    assert np.allclose(fitter.map_estimates['mu'].values, true_mu, atol=0.3)
    assert fitter.map_sigma.mean() < 1.0   # rough sanity


@pytest.mark.parametrize("noise_sd", [0.8])
def test_map_beats_classical_at_high_noise(noise_sd):
    rng = np.random.default_rng(7)
    n_vx = 40
    x = np.linspace(0, 10, n_vx)
    d = np.abs(x[:, None] - x[None, :])
    true_mu = np.linspace(-3, 3, n_vx).astype(np.float32)

    paradigm = pd.DataFrame({'x': np.linspace(-5, 5, 80, dtype=np.float32)})
    true_pars = pd.DataFrame({
        'mu': true_mu,
        'sd': np.full(n_vx, 1.0, dtype=np.float32),
        'amplitude': np.full(n_vx, 2.0, dtype=np.float32),
        'baseline': np.zeros(n_vx, dtype=np.float32),
    })
    model = GaussianPRF(paradigm=paradigm, parameters=true_pars)
    clean = model.predict(paradigm=paradigm, parameters=true_pars)
    noise = rng.standard_normal(clean.shape).astype(np.float32) * noise_sd
    data = pd.DataFrame(clean.values + noise, columns=true_pars.index)

    # Reasonable init (skips the buggy get_init_pars path).
    init = true_pars.copy()
    init['mu'] = 0.0

    prior = GeodesicGPPrior(d, lengthscale_init=2.0,
                            variance_init=2.0, nugget_init=0.1)
    fitter = BayesianParameterFitter(model, data, paradigm,
                                     priors={'mu': prior})
    fitter.fit(max_n_iterations=300,
               classical_kwargs={'init_pars': init},
               progressbar=False)

    cls_rmse = float(np.sqrt(np.mean(
        (fitter.classical_estimates['mu'].values - true_mu) ** 2)))
    map_rmse = float(np.sqrt(np.mean(
        (fitter.map_estimates['mu'].values - true_mu) ** 2)))

    assert map_rmse < cls_rmse, \
        f"MAP RMSE {map_rmse:.3f} not better than classical {cls_rmse:.3f}"
    # Per-vertex sigma should be in the ballpark of the noise sd.
    assert 0.5 * noise_sd < fitter.map_sigma.mean() < 2.0 * noise_sd
