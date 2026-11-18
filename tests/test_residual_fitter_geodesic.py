"""ResidualFitter with a distance matrix: Ω is positive definite by construction.

Ω = ρ [α (e^{−β D̃} ∘ ττᵀ) + (1 − α) ττᵀ] + (1 − ρ) diag(τ²) + σ² WWᵀ

Two defects are guarded against here. β was an unconstrained variable, so a step
to β < 0 made e^{−β D} grow with distance and Ω indefinite. And the kernel was
computed on raw graph distances, on which an exponential kernel need not be
PSD, with `safe_cholesky` adding jitter on every evaluation to hide it.

Run once per backend (`KERAS_BACKEND=... pytest`).
"""
import numpy as np
import pandas as pd
import pytest
import keras
from keras import ops

from braincoder.models import GaussianPRF
from braincoder.optimize import ResidualFitter
from braincoder.optimize.gp_prior import mds_embed_distances
from braincoder.utils.backend import strict_cholesky
from braincoder.utils.cortex import geodesic_distance_matrix

BACKEND = keras.backend.backend()


def k32_distances():
    """Shortest-path metric of K_{3,2}: not a positive-definite metric space."""
    d = np.full((5, 5), 2.0)
    d[:3, 3:] = d[3:, :3] = 1.0
    np.fill_diagonal(d, 0.0)
    return d


def bumpy_mesh(n=12, amplitude=2.0, seed=0):
    """Triangulated grid with height noise, so mesh-edge paths are non-Euclidean."""
    rng = np.random.default_rng(seed)
    x, y = np.meshgrid(np.arange(n, dtype=float), np.arange(n, dtype=float))
    vertices = np.c_[x.ravel(), y.ravel(), amplitude * rng.standard_normal(n * n)]
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b, c, d = i * n + j, i * n + j + 1, (i + 1) * n + j, (i + 1) * n + j + 1
            faces += [[a, b, c], [b, d, c]]
    return vertices, np.asarray(faces)


def mesh_distances(n_sources=25, seed=0):
    vertices, faces = bumpy_mesh(seed=seed)
    sources = np.random.default_rng(seed).choice(len(vertices), n_sources, replace=False)
    return geodesic_distance_matrix(vertices, faces, source_indices=sources,
                                    progressbar=False, dtype=np.float64)


def omega_numpy(tau, rho, alpha, beta, sigma2, wwt, d):
    t = np.outer(tau, tau)
    return (rho * (alpha * np.exp(-beta * d) * t + (1 - alpha) * t)
            + (1 - rho) * np.diag(tau ** 2) + sigma2 * wwt)


def make_fitter(n_voxels, n_trials=10):
    paradigm = pd.DataFrame({'x': np.linspace(1, 10, n_trials)}, dtype=np.float32)
    pars = pd.DataFrame({'mu': np.linspace(2, 9, n_voxels), 'sd': 3., 'amplitude': 1.,
                         'baseline': 0.}, dtype=np.float32)
    data = pd.DataFrame(np.zeros((n_trials, n_voxels), dtype=np.float32))
    return ResidualFitter(GaussianPRF(), data, paradigm, parameters=pars)


def simulate_residuals(omega, n_trials, seed=1):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((n_trials, len(omega))) @ np.linalg.cholesky(omega).T).astype(np.float32)


# ---------------------------------------------------------------------------
# The MDS embedding
# ---------------------------------------------------------------------------

def test_k32_shows_why_the_embedding_is_needed():
    """The raw exponential kernel on this metric is indefinite; embedded it is not."""
    d = k32_distances()
    betas = np.logspace(-2, 1, 40)
    assert min(np.linalg.eigvalsh(np.exp(-b * d)).min() for b in betas) < -1e-3
    d_e = mds_embed_distances(d)
    assert min(np.linalg.eigvalsh(np.exp(-b * d_e)).min() for b in betas) > -1e-6


def test_mds_is_identity_on_euclidean_input():
    pts = np.random.default_rng(0).standard_normal((30, 3))
    d = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
    d_e, info = mds_embed_distances(d, return_info=True)
    np.testing.assert_allclose(d_e, d, atol=1e-4)
    assert info['negative_eigenvalue_fraction'] < 1e-8


def test_mds_matches_explicit_coordinates_and_only_lengthens():
    d = mesh_distances()
    d_e, info = mds_embed_distances(d, return_info=True)
    assert info['negative_eigenvalue_fraction'] > 0
    n = len(d)
    J = np.eye(n) - 1. / n
    w, V = np.linalg.eigh(-.5 * J @ (d ** 2) @ J)
    pos = w > 1e-10 * np.abs(w).max()
    c = V[:, pos] * np.sqrt(w[pos])
    explicit = np.sqrt(((c[:, None] - c[None]) ** 2).sum(-1))
    np.testing.assert_allclose(d_e, explicit, atol=1e-3)
    assert (d_e >= d - 1e-3).all()


def test_mds_keeps_coincident_points_coincident():
    d = mesh_distances(n_sources=10)
    # voxels 0 and 10 share a vertex: same row, distance 0
    idx = np.r_[np.arange(10), 0]
    d_dup = d[np.ix_(idx, idx)]
    d_e = mds_embed_distances(d_dup)
    assert d_e[0, 10] < 1e-2
    np.testing.assert_allclose(d_e[0], d_e[10], atol=1e-2)


def test_mds_refuses_unreachable_vertices():
    d = np.array([[0., 1., np.inf], [1., 0., np.inf], [np.inf, np.inf, 0.]])
    with pytest.raises(ValueError, match='non-finite'):
        mds_embed_distances(d)


def test_mds_memory_does_not_scale_cubically():
    """2000 points: a difference tensor would need ~64 GB, the Gram route ~100 MB."""
    pts = np.random.default_rng(0).standard_normal((2000, 2))
    d = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
    assert mds_embed_distances(d).shape == (2000, 2000)


# ---------------------------------------------------------------------------
# Ω is positive definite by construction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('duplicates', [False, True])
def test_omega_is_factorisable_without_jitter_over_the_whole_parameter_space(duplicates):
    d = mesh_distances(n_sources=25)
    if duplicates:   # voxels sharing a vertex: the kernel alone is singular
        idx = np.r_[np.arange(25), [0, 1, 2]]
        d = d[np.ix_(idx, idx)]
    d_e = mds_embed_distances(d).astype(np.float64)
    n = len(d)
    rng = np.random.default_rng(3)
    tau = rng.uniform(.5, 3, n)
    w = rng.standard_normal((6, n))
    wwt = w.T @ w
    wwt = wwt / wwt.mean()
    fitter = make_fitter(n)
    d_t = ops.convert_to_tensor(d_e.astype(np.float32))
    tau_t = ops.convert_to_tensor(tau[None, :].astype(np.float32))
    wwt_t = ops.convert_to_tensor(wwt.astype(np.float32))
    for rho in (1e-4, .1, .5, .9, .99):
        for alpha in (1e-6, .1, .5, .9, 1 - 1e-6):
            for beta in (1e-6, 1e-3, .1, 1., 10., 1e3):
                for sigma2 in (0., 1e-3, 1.):
                    om = omega_numpy(tau, rho, alpha, beta, sigma2, wwt, d_e)
                    assert np.linalg.eigvalsh(om).min() > 0, (rho, alpha, beta, sigma2)
                    # the same matrix through the backend, in float32, with no jitter
                    om_t = fitter._get_omega_distance(tau_t, rho, sigma2, wwt_t, alpha, beta, d_t)
                    np.testing.assert_allclose(ops.convert_to_numpy(om_t), om, rtol=2e-3, atol=2e-3)
                    L = ops.convert_to_numpy(strict_cholesky(om_t))
                    assert np.isfinite(L).all(), (rho, alpha, beta, sigma2)


def test_raw_distances_can_break_omega_but_the_fit_embeds_them():
    """Same parameters, with and without the embedding, on the K_{3,2} metric."""
    d = k32_distances()
    tau = np.ones(5)
    best = min((np.linalg.eigvalsh(omega_numpy(tau, 1 - 1e-6, 1 - 1e-6, b, 0., np.zeros((5, 5)), d)).min(), b)
               for b in np.logspace(-2, 1, 40))
    assert best[0] < 0
    d_e = mds_embed_distances(d).astype(np.float64)
    assert np.linalg.eigvalsh(omega_numpy(tau, 1 - 1e-6, 1 - 1e-6, best[1], 0., np.zeros((5, 5)), d_e)).min() > 0


def test_strict_cholesky_does_not_jitter():
    m = np.array([[2., .5], [.5, 1.]], dtype=np.float32)
    L = ops.convert_to_numpy(strict_cholesky(ops.convert_to_tensor(m)))
    np.testing.assert_allclose(L @ L.T, m, atol=1e-6)


# ---------------------------------------------------------------------------
# Nesting
# ---------------------------------------------------------------------------

def test_nesting_of_the_model_without_distances():
    """α = 0, or β = 0: Ω = ρττᵀ + (1 − ρ)diag(τ²) + σ²WWᵀ, the model without D."""
    n = 8
    rng = np.random.default_rng(0)
    d = mds_embed_distances(np.abs(rng.standard_normal((n, n)) + 3) * (1 - np.eye(n)) * .5
                            + (np.abs(rng.standard_normal((n, n)) + 3) * (1 - np.eye(n)) * .5).T)
    tau = ops.convert_to_tensor(rng.uniform(1, 2, (1, n)).astype(np.float32))
    w = rng.standard_normal((5, n))
    wwt = ops.convert_to_tensor((w.T @ w).astype(np.float32))
    f = make_fitter(n)
    base = ops.convert_to_numpy(f._get_omega(tau, .3, .1, wwt))
    d_t = ops.convert_to_tensor(d)
    a0 = ops.convert_to_numpy(f._get_omega_distance(tau, .3, .1, wwt, 0., .7, d_t))
    b0 = ops.convert_to_numpy(f._get_omega_distance(tau, .3, .1, wwt, .6, 0., d_t))
    np.testing.assert_allclose(a0, base, atol=1e-6)
    np.testing.assert_allclose(b0, base, atol=1e-6)


# ---------------------------------------------------------------------------
# The constraint, and recovery
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('bad', [0.0, -1.0])
def test_nonpositive_init_beta_is_refused(bad):
    f = make_fitter(5)
    with pytest.raises(ValueError, match='init_beta'):
        f.fit(D=k32_distances(), init_beta=bad, residuals=np.zeros((10, 5)) + np.eye(10, 5),
              use_wwt=False, max_n_iterations=2, progressbar=False)


def test_distance_options_that_would_be_ignored_are_refused():
    r = np.random.default_rng(0).standard_normal((10, 5)).astype(np.float32)
    with pytest.raises(ValueError, match='spherical'):
        make_fitter(5).fit(D=k32_distances(), spherical=True, residuals=r, use_wwt=False, progressbar=False)
    f = make_fitter(5)
    f.lambd = .5
    with pytest.raises(ValueError, match='lambd'):
        f.fit(D=k32_distances(), residuals=r, use_wwt=False, progressbar=False)
    with pytest.raises(ValueError, match='shape'):
        make_fitter(5).fit(D=np.zeros((4, 4)), residuals=r, use_wwt=False, progressbar=False)
    with pytest.raises(ValueError, match='psd_correction'):
        make_fitter(5).fit(D=k32_distances(), psd_correction='jitter', residuals=r, use_wwt=False, progressbar=False)


def test_beta_stays_positive_when_the_data_want_it_negative():
    """Noise correlations that *rise* with distance push an unconstrained β below 0."""
    n = 20
    x = np.linspace(0, 10, n)
    d = np.abs(x[:, None] - x[None])
    rng = np.random.default_rng(5)
    sign = np.where(x > 5, 1., -1.)
    r = (rng.standard_normal((400, 1)) * sign[None] + .5 * rng.standard_normal((400, n))).astype(np.float32)
    f = make_fitter(n)
    omega, _ = f.fit(D=d, residuals=r, use_wwt=False, init_beta=.1, max_n_iterations=400,
                     learning_rate=.05, progressbar=False)
    beta = float(f.fitted_omega_parameters['beta'])
    assert beta > 0
    assert f.n_failed_steps == 0
    assert f.omega_min_eigenvalue_ > 0
    assert np.linalg.eigvalsh(np.asarray(omega, dtype=np.float64)).min() > 0


@pytest.mark.parametrize('method', ['gauss', 't'])
def test_recovers_beta_and_returns_the_omega_of_the_fitted_parameters(method):
    n = 40
    pts = np.random.default_rng(0).uniform(0, 30, (n, 2))
    d = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
    tau = np.full(n, 1.5)
    beta, alpha, rho = .2, .8, .7
    truth = omega_numpy(tau, rho, alpha, beta, 0., np.zeros((n, n)), d)
    r = simulate_residuals(truth, 1500)
    f = make_fitter(n)
    omega, _ = f.fit(D=d, residuals=r, use_wwt=False, method=method, init_dof=30., init_alpha=.5,
                     init_rho=.5, max_n_iterations=1200, learning_rate=.03, progressbar=False)
    p = f.fitted_omega_parameters
    assert f.n_failed_steps == 0
    assert abs(float(p['beta']) - beta) / beta < .35
    assert abs(float(p['alpha']) - alpha) < .2
    assert abs(float(p['rho']) - rho) < .2
    # returned Ω is the one the reported parameters give, with no jitter
    rebuilt = omega_numpy(np.asarray(p['tau']).ravel(), float(p['rho']), float(p['alpha']),
                          float(p['beta']), float(p['sigma2']), np.zeros((n, n)),
                          f.distance_matrix_embedded_.astype(np.float64))
    np.testing.assert_allclose(np.asarray(omega, dtype=np.float64), rebuilt, rtol=1e-3, atol=1e-3)


def test_fit_without_distances_is_unchanged_by_default():
    """Default without D stays safe_cholesky, so old pipelines reproduce."""
    r = np.random.default_rng(0).standard_normal((60, 6)).astype(np.float32)
    f = make_fitter(6)
    f.fit(residuals=r, use_wwt=False, max_n_iterations=5, min_n_iterations=1, progressbar=False)
    assert not hasattr(f, 'omega_min_eigenvalue_')
    g = make_fitter(6)
    g.fit(residuals=r, use_wwt=False, max_n_iterations=5, min_n_iterations=1, progressbar=False, cholesky='strict')
    assert g.omega_min_eigenvalue_ > 0
