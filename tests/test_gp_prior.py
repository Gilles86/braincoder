"""Tests for the GP prior and Bayesian fitter (Daghlian et al. 2025).

Three layers of coverage:
  * log_prob matches a manual numpy MVN log-prob (math sanity)
  * hyperparameter MLE recovers known (lengthscale, variance, nugget)
    from samples drawn from a known GP
  * BayesianParameterFitter recovers smooth-spatial-structure parameters
    better than classical at high noise (a mini version of paper Fig. 2)
"""
from helpers import Bound, grid_for
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
    model = Bound(GaussianPRF(), paradigm, true_pars)
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
    model = Bound(GaussianPRF(), paradigm, true_pars)
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


def test_joint_hyperparams_updates_kernel_params():
    """joint_hyperparams=True puts ψ in the trainable set.

    Smoke test: the lengthscale moves from its initial value during
    fit_map (because Adam now has gradient access to it). Compare
    against the frozen-ψ path where the lengthscale must not move.
    """
    rng = np.random.default_rng(13)
    n_vx = 25
    x = np.linspace(0, 10, n_vx)
    d = np.abs(x[:, None] - x[None, :])
    true_mu = np.linspace(-3, 3, n_vx).astype(np.float32)

    paradigm = pd.DataFrame({'x': np.linspace(-5, 5, 60, dtype=np.float32)})
    true_pars = pd.DataFrame({
        'mu': true_mu,
        'sd': np.full(n_vx, 1.0, dtype=np.float32),
        'amplitude': np.full(n_vx, 2.0, dtype=np.float32),
        'baseline': np.zeros(n_vx, dtype=np.float32),
    })
    model = Bound(GaussianPRF(), paradigm, true_pars)
    clean = model.predict(paradigm=paradigm, parameters=true_pars)
    noise = rng.standard_normal(clean.shape).astype(np.float32) * 0.5
    data = pd.DataFrame(clean.values + noise, columns=true_pars.index)

    def make_fitter():
        prior = GeodesicGPPrior(d, lengthscale_init=2.0,
                                variance_init=2.0, nugget_init=0.1)
        fitter = BayesianParameterFitter(model, data, paradigm,
                                          priors={'mu': prior})
        fitter.classical_estimates = true_pars.copy()
        return fitter, prior

    # Frozen-ψ baseline: lengthscale must not move during fit_map.
    fitter_frozen, prior_frozen = make_fitter()
    l_before_frozen = prior_frozen.lengthscale
    fitter_frozen.fit_map(max_n_iterations=80, progressbar=False)
    l_after_frozen = prior_frozen.lengthscale
    assert l_after_frozen == pytest.approx(l_before_frozen), \
        "frozen path: lengthscale must not move"

    # Joint path: lengthscale should move (Adam has gradient access).
    fitter_joint, prior_joint = make_fitter()
    l_before_joint = prior_joint.lengthscale
    fitter_joint.fit_map(max_n_iterations=80, progressbar=False,
                          joint_hyperparams=True)
    l_after_joint = prior_joint.lengthscale
    assert l_after_joint != pytest.approx(l_before_joint), \
        f"joint path: lengthscale should move, but stayed at {l_after_joint}"


def test_shared_lengthscale_ties_priors():
    """shared_lengthscale=True: all priors share one lengthscale Variable.

    After fit_hyperparameters, every prior reports the same lengthscale.
    The mechanism is in-place Variable substitution, so each prior's
    `lengthscale` property reads the shared value. Per-prior variance
    and nugget remain independent.
    """
    rng = np.random.default_rng(11)
    n_vx = 30
    x = np.linspace(0, 10, n_vx)
    d = np.abs(x[:, None] - x[None, :])
    true_mu = np.linspace(-3, 3, n_vx).astype(np.float32)
    true_sd = np.full(n_vx, 1.0, dtype=np.float32)

    paradigm = pd.DataFrame({'x': np.linspace(-5, 5, 60, dtype=np.float32)})
    true_pars = pd.DataFrame({
        'mu': true_mu,
        'sd': true_sd,
        'amplitude': np.full(n_vx, 2.0, dtype=np.float32),
        'baseline': np.zeros(n_vx, dtype=np.float32),
    })
    model = Bound(GaussianPRF(), paradigm, true_pars)
    clean = model.predict(paradigm=paradigm, parameters=true_pars)
    noise = rng.standard_normal(clean.shape).astype(np.float32) * 0.4
    data = pd.DataFrame(clean.values + noise, columns=true_pars.index)

    priors = {
        'mu': GeodesicGPPrior(d, lengthscale_init=2.0,
                              variance_init=2.0, nugget_init=0.1),
        'sd': GeodesicGPPrior(d, lengthscale_init=3.0,    # different init
                              variance_init=0.5, nugget_init=0.1),
    }
    fitter = BayesianParameterFitter(model, data, paradigm, priors=priors)
    fitter.classical_estimates = true_pars.copy()
    fitter.fit_hyperparameters(shared_lengthscale=True, progressbar=False,
                                max_n_iterations=200)

    # Tied: both priors return the same lengthscale.
    assert priors['mu'].lengthscale == priors['sd'].lengthscale
    # Variance can still differ (it's not tied).
    assert priors['mu']._log_variance is not priors['sd']._log_variance
    # History recorded the shared value.
    assert '_shared' in fitter.hyperparameter_history
    assert 'shared_lengthscale' in fitter.hyperparameter_history['_shared']


# ---------------------------------------------------------------------------
# Mean function and transform
# ---------------------------------------------------------------------------

def test_zero_mean_is_the_default_and_unchanged():
    """mean='zero', transform='identity' is the original prior: no mean
    variable, three trainable hyperparameters, MVN(0, K) log-prob."""
    rng = np.random.default_rng(11)
    n = 20
    d = line_distance_matrix(n)
    prior = GeodesicGPPrior(d, lengthscale_init=2., variance_init=1.3,
                            nugget_init=.2, jitter=0.0)
    assert prior.mean_type == 'zero' and prior.mean == 0.0
    assert prior.transform == 'identity'
    assert len(prior.trainable_variables) == 3
    values = (20. + rng.standard_normal(n)).astype(np.float32)
    K = 1.3 * np.exp(-d ** 2 / (2 * 2. ** 2)) + .2 * np.eye(n)
    assert np.isclose(float(prior.log_prob(values)),
                      numpy_mvn_log_prob(values.astype(float), K), rtol=1e-3)


@pytest.mark.parametrize('transform', ['identity', 'log'])
def test_constant_and_fixed_mean_log_prob(transform):
    rng = np.random.default_rng(12)
    n = 20
    d = line_distance_matrix(n)
    K = 1.3 * np.exp(-d ** 2 / (2 * 2. ** 2)) + .2 * np.eye(n)
    values = np.exp(3. + .3 * rng.standard_normal(n)).astype(np.float32)
    z = np.log(values.astype(float)) if transform == 'log' else values.astype(float)
    for mean in ('constant', 2.5):
        prior = GeodesicGPPrior(d, lengthscale_init=2., variance_init=1.3,
                                nugget_init=.2, jitter=0.0, mean=mean,
                                mean_init=2.5 if mean == 'constant' else None,
                                transform=transform)
        assert np.isclose(prior.mean, 2.5)
        expected = numpy_mvn_log_prob(z - 2.5, K)
        assert np.isclose(float(prior.log_prob(values)), expected, rtol=1e-3), mean
        n_vars = 4 if mean == 'constant' else 3
        assert len(prior.trainable_variables) == n_vars


def test_constant_mean_initialised_at_median():
    d = line_distance_matrix(10)
    values = np.arange(10, 20, dtype=np.float32)
    prior = GeodesicGPPrior(d, mean='constant', transform='log')
    assert prior.needs_mean_init
    prior.initialize_mean(values)
    assert not prior.needs_mean_init
    assert np.isclose(prior.mean, np.median(np.log(values)), atol=1e-5)


def _demo_geometry(n_vox=150):
    """Geodesic distances between the best voxels of the public NPC demo."""
    try:
        from braincoder.utils.data import load_pratcarrabin2025_npc
        from scipy.spatial import cKDTree
        b = load_pratcarrabin2025_npc()
    except Exception as exc:          # offline: the bundle cannot be downloaded
        pytest.skip(f'demo bundle unavailable: {exc}')
    top = np.argsort(b['r2'].values)[::-1][:n_vox]
    xyz = b['voxel_coords_mm'][['x', 'y', 'z']].values[top]
    verts = b['surface_vertices'].astype(float)
    _, vtx = cKDTree(verts).query(xyz)
    D = geodesic_distance_matrix(verts, b['surface_faces'].astype(int),
                                 source_indices=vtx, progressbar=False,
                                 dtype=np.float64)
    from braincoder.optimize.gp_prior import _mds_embed_distances
    return _mds_embed_distances(D.astype(np.float32)).astype(np.float64)


def test_zero_mean_shrinks_offset_constant_mean_does_not():
    """A smooth map with mean 20 on real cortical (geodesic) distances.

    With the generating kernel, the posterior mean of the map given noisy
    estimates is pulled towards 0 under the zero-mean prior, and not under
    the constant-mean prior. With hyperparameters fitted by type-II ML, the
    constant-mean prior recovers the mean and the lengthscale; the zero-mean
    prior can only absorb the offset by inflating the lengthscale.
    """
    D = _demo_geometry()
    n = len(D)
    rng = np.random.default_rng(5)
    l_true, sd_true, m_true, noise_sd = 8., 4., 20., 3.
    K = sd_true ** 2 * np.exp(-D ** 2 / (2 * l_true ** 2))
    truth = m_true + np.linalg.cholesky(K + 1e-6 * np.eye(n)) @ rng.standard_normal(n)
    y = (truth + noise_sd * rng.standard_normal(n)).astype(np.float32)

    def posterior_mean(m, K_prior):
        return m + K_prior @ np.linalg.solve(K_prior + noise_sd ** 2 * np.eye(n), y - m)

    # 1. Known kernel: the zero-mean prior shrinks the map towards 0.
    post_zero = posterior_mean(0., K + 1e-6 * np.eye(n))
    post_const = posterior_mean(m_true, K + 1e-6 * np.eye(n))
    bias_const = post_const.mean() - truth.mean()
    shrink_zero = truth.mean() - post_zero.mean()
    assert abs(bias_const) < .5, bias_const
    assert shrink_zero > 1. and shrink_zero > 3 * abs(bias_const), (shrink_zero, bias_const)
    assert np.sqrt(np.mean((post_const - truth) ** 2)) < np.sqrt(np.mean((y - truth) ** 2))

    # 2. Type-II ML of the hyperparameters.
    fitted = {}
    for mean in ('zero', 'constant'):
        prior = GeodesicGPPrior(D.astype(np.float32), lengthscale_init=5.,
                                variance_init=float(np.var(y)),
                                nugget_init=.1 * float(np.var(y)),
                                psd_correction='none', mean=mean)
        prior.fit_hyperparameters(y, max_n_iterations=1000, learning_rate=.05,
                                  progressbar=False)
        fitted[mean] = prior.hyperparameters
    c, z = fitted['constant'], fitted['zero']
    assert abs(c['mean'] - m_true) < 2., c
    assert .5 * l_true < c['lengthscale'] < 2. * l_true, c
    assert z['lengthscale'] > 1.5 * c['lengthscale'], (z, c)


def test_bayesian_fitter_estimates_constant_mean():
    """The mean starts at the median of the classical estimates and is
    estimated in stage 2 and in joint type-II MAP."""
    rng = np.random.default_rng(3)
    n = 40
    x = np.linspace(0, 20, n)
    d = np.abs(x[:, None] - x[None])
    model = GaussianPRF()
    pars = pd.DataFrame(dict(mu=20 + 3 * np.sin(x / 4), sd=6., amplitude=1.,
                             baseline=0.)).astype(np.float32)
    paradigm = pd.DataFrame({'x': rng.uniform(5, 35, 120)}).astype(np.float32)
    data = pd.DataFrame(model.predict(paradigm=paradigm, parameters=pars).values
                        + rng.normal(0, .05, (120, n))).astype(np.float32)
    for joint in (False, True):
        prior = GeodesicGPPrior(d, lengthscale_init=3., variance_init=5.,
                                nugget_init=.5, mean='constant')
        fitter = BayesianParameterFitter(model, data, paradigm, priors={'mu': prior})
        fitter.classical_estimates = pars
        if not joint:
            fitter.fit_hyperparameters(progressbar=False, max_n_iterations=50)
        m0 = float(np.median(pars['mu']))
        fitter.fit_map(max_n_iterations=60, progressbar=False, joint_hyperparams=joint)
        assert not prior.needs_mean_init
        assert abs(prior.mean - m0) < 3.
        assert prior.mean != m0          # it was estimated, not only initialised
        assert np.isfinite(fitter.map_estimates.values).all()
