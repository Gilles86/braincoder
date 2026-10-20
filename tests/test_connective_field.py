"""Connective-field model, fitter and geodesic distances.

Runs on whichever Keras backend is active (CI runs TF, JAX and torch).
"""
import numpy as np
import pandas as pd
import pytest
from scipy.spatial import ConvexHull

from braincoder.models import GaussianConnectiveField
from braincoder.optimize import ConnectiveFieldFitter, exclusion_from_distances
from braincoder.utils.cortex import (block_distance_matrix, geodesic_distance_matrix,
                                     submesh)


# ---------------------------------------------------------------- meshes
def grid_mesh(nu, nv, spacing=1.0, embed=None):
    """Triangulated (u, v) grid; ``embed(u, v) -> xyz`` maps it to 3-D."""
    u, v = np.meshgrid(np.arange(nu) * spacing, np.arange(nv) * spacing, indexing='ij')
    u, v = u.ravel(), v.ravel()
    xyz = np.c_[u, v, np.zeros_like(u)] if embed is None else embed(u, v)
    faces = []
    for i in range(nu - 1):
        for j in range(nv - 1):
            a = i * nv + j
            b, c, d = a + 1, a + nv, a + nv + 1
            # alternate the diagonal so the mesh is not anisotropic
            faces += [[a, c, d], [a, d, b]] if (i + j) % 2 else [[a, c, b], [b, c, d]]
    return xyz.astype(float), np.array(faces), u, v


def hairpin(u, v, bank=15.0, radius=1.0):
    """Fold a strip into a sulcus: bank A (z=0), a half-turn of the given
    radius, bank B (z=2*radius) running back above A."""
    turn = np.pi * radius
    x = np.where(u <= bank, u, np.where(u <= bank + turn,
                                        bank + radius * np.sin((u - bank) / radius),
                                        bank - (u - bank - turn)))
    z = np.where(u <= bank, 0.0, np.where(u <= bank + turn,
                                          radius * (1 - np.cos((u - bank) / radius)),
                                          2 * radius))
    return np.c_[x, v, z]


def sphere_mesh(n=3000, r=50.0, seed=0):
    rng = np.random.default_rng(seed)
    P = rng.normal(size=(n, 3))
    P = r * P / np.linalg.norm(P, axis=1, keepdims=True)
    F = ConvexHull(P).simplices
    nrm = np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]])
    flip = (nrm * P[F].mean(1)).sum(1) < 0
    F[flip] = F[flip][:, [0, 2, 1]]
    return P, F


def simulate(D, centres, sigmas, n_samples=400, noise=0.5, seed=0, smooth=None):
    """Source signals (optionally spatially smooth over D) and CF targets."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, D.shape[0]))
    if smooth is not None:
        K = np.exp(-np.where(np.isfinite(D), D, np.inf) ** 2 / (2 * smooth ** 2))
        X = X @ K.T
        X = (X - X.mean(0)) / X.std(0)
    pars = pd.DataFrame({'center_index': centres, 'sigma': sigmas,
                         'baseline': 1.0, 'amplitude': 2.0}).astype(np.float32)
    model = GaussianConnectiveField(D, normalize='sum')
    Y = model.predict(X, pars).values
    Y = Y + noise * Y.std(0) * rng.normal(size=Y.shape)
    return X.astype(np.float32), pd.DataFrame(Y.astype(np.float32)), pars


# ---------------------------------------------------------------- distances
def test_heat_method_flat_and_sphere():
    V, F, u, v = grid_mesh(31, 31)
    src = np.array([15 * 31 + 15])
    D = geodesic_distance_matrix(V, F, src, np.arange(len(V)), method='heat', progressbar=False)
    E = np.linalg.norm(V[src] - V, axis=1)
    sel = (E > 3) & (E < 12)
    assert np.median(np.abs(D[0, sel] / E[sel] - 1)) < 0.03

    P, F = sphere_mesh()
    src = np.arange(5)
    G = 50 * np.arccos(np.clip(P[src] @ P.T / 2500, -1, 1))
    sel = (G > 5) & (G < 60)
    errs = {}
    for m in ('heat', 'dijkstra'):
        D = geodesic_distance_matrix(P, F, src, np.arange(len(P)), method=m, progressbar=False)
        errs[m] = np.median(np.abs(D[sel] / G[sel] - 1))
    assert errs['heat'] < 0.03
    assert errs['heat'] < errs['dijkstra']          # Dijkstra follows edges: biased high


def test_distance_matrix_square_submesh_and_blocks():
    V, F, u, v = grid_mesh(12, 12)
    idx = np.arange(0, 144, 7)
    D = geodesic_distance_matrix(V, F, idx, method='heat', progressbar=False)
    assert D.shape == (len(idx), len(idx))
    np.testing.assert_allclose(D, D.T)
    assert np.all(np.diag(D) == 0)
    # restricting to a submesh that still contains the paths changes nothing much
    mask = u < 11
    Vs, Fs, index = submesh(V, F, mask)
    assert len(Vs) == mask.sum() and Fs.max() < len(Vs)
    keep = idx[mask[idx]]
    D2 = geodesic_distance_matrix(V, F, keep, method='heat', vertex_mask=mask, progressbar=False)
    D1 = geodesic_distance_matrix(V, F, keep, method='heat', progressbar=False)
    assert np.median(np.abs(D2 - D1)) < 0.3
    # truncation and hemisphere blocks
    Dt = geodesic_distance_matrix(V, F, idx, method='dijkstra', max_distance=5,
                                  progressbar=False)
    assert np.isinf(Dt).any() and np.nanmax(Dt[np.isfinite(Dt)]) <= 5
    B = block_distance_matrix([D, D])
    assert B.shape == (2 * len(idx),) * 2
    assert np.isinf(B[:len(idx), len(idx):]).all()


# ---------------------------------------------------------------- model
def test_model_weights_and_prediction():
    V, F, u, v = grid_mesh(10, 10)
    D = geodesic_distance_matrix(V, F, method='dijkstra', progressbar=False)
    D = block_distance_matrix([D, D])               # two "hemispheres"
    pars = pd.DataFrame({'center_index': [5., 150.], 'sigma': [1.5, 3.],
                         'baseline': [0., 1.], 'amplitude': [1., 2.]})
    for norm in ('sum', 'density', None):
        m = GaussianConnectiveField(D, normalize=norm, truncate=3.0)
        W = m.get_cf_weights(pars).values
        assert (W[0, 100:] == 0).all() and (W[1, :100] == 0).all()   # no cross-hemisphere weight
        assert (W[0][D[5] > 4.5] == 0).all()                         # truncated at 3 sigma
        if norm == 'sum':
            np.testing.assert_allclose(W.sum(1), 1, rtol=1e-5)
        X = np.random.default_rng(0).normal(size=(50, 200)).astype(np.float32)
        pred = m.predict(X, pars).values
        np.testing.assert_allclose(pred, X @ W.T * pars['amplitude'].values + pars['baseline'].values,
                                   rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(m.predict(X, pars, batch_size=1).values, pred, rtol=1e-5, atol=1e-5)
    props = pd.DataFrame({'ecc': np.arange(200.), 'hemi': np.repeat(['L', 'R'], 100)})
    sp = m.source_properties(pars, props)
    assert list(sp['ecc']) == [5., 150.] and list(sp['hemi']) == ['L', 'R']


# ---------------------------------------------------------------- fitting
def test_grid_recovery_flat_mesh():
    V, F, u, v = grid_mesh(24, 24)
    D = geodesic_distance_matrix(V, F, method='heat', progressbar=False)
    rng = np.random.default_rng(1)
    centres = rng.choice(len(V), 40, replace=False)
    sigmas = rng.choice([1.0, 2.0, 4.0], 40)
    X, Y, true = simulate(D, centres, sigmas, noise=0.3)
    model = GaussianConnectiveField(D, normalize='sum')
    fitter = ConnectiveFieldFitter(model, Y, X)
    est = fitter.fit_grid([1.0, 2.0, 4.0], target_batch_size=16, candidate_batch_size=500,
                          progressbar=False)
    err = np.linalg.norm(V[est['center_index'].astype(int)] - V[centres], axis=1)
    assert np.median(err) <= 1.0
    assert (est['sigma'].values == sigmas).mean() > 0.8
    np.testing.assert_allclose(est['amplitude'], 2.0, rtol=0.25)
    np.testing.assert_allclose(est['baseline'], 1.0, atol=0.3)
    # grid-fit R2 equals r^2 of the search (closed-form OLS)
    np.testing.assert_allclose(fitter.get_rsq(est).values, fitter.grid_rsq_.values, atol=1e-3)


def test_geodesic_vs_euclidean_on_folded_mesh():
    """Two banks of a sulcus are 2 mm apart in 3-D but >= 2*(15-x)+pi mm
    apart along the surface. Euclidean CFs pool across the fold."""
    V, F, u, v = grid_mesh(34, 8, spacing=1.0, embed=hairpin)
    D_geo = geodesic_distance_matrix(V, F, method='heat', progressbar=False)
    D_euc = np.linalg.norm(V[:, None] - V[None], axis=2).astype(np.float32)
    bank_a, bank_b = u < 14, u > 15 + np.pi + 1
    sigma = 2.0

    # (1) A Euclidean Gaussian centred on bank A puts substantial weight on bank B
    centres = np.flatnonzero(bank_a & (u >= 8) & (v >= 2) & (v <= 5))
    pars = pd.DataFrame({'center_index': centres, 'sigma': sigma,
                         'baseline': 0., 'amplitude': 1.}).astype(np.float32)
    w_geo = GaussianConnectiveField(D_geo, normalize='sum').get_cf_weights(pars).values
    w_euc = GaussianConnectiveField(D_euc, normalize='sum').get_cf_weights(pars).values
    assert w_geo[:, bank_b].sum(1).max() < 0.01
    assert w_euc[:, bank_b].sum(1).mean() > 0.2

    # (2) Fitting: data generated by geodesic CFs on surface-smooth sources is
    # recovered by the geodesic model; the Euclidean model fits worse
    X, Y, true = simulate(D_geo, centres, np.full(len(centres), sigma), noise=0.3,
                          smooth=1.5, n_samples=500, seed=3)
    res = {}
    for name, D in (('geo', D_geo), ('euc', D_euc)):
        fitter = ConnectiveFieldFitter(GaussianConnectiveField(D, normalize='sum'), Y, X)
        est = fitter.fit_grid([1.0, 2.0, 4.0], progressbar=False)
        c = est['center_index'].astype(int).values
        res[name] = dict(r=fitter.grid_r_.values,
                         err=np.abs(u[c] - u[centres]) + np.abs(v[c] - v[centres]))
    assert np.median(res['geo']['err']) <= 1.0
    assert res['geo']['r'].mean() > res['euc']['r'].mean()


def test_exclusion_leave_neighbourhood_out():
    V, F, u, v = grid_mesh(20, 20)
    D = geodesic_distance_matrix(V, F, method='heat', progressbar=False)
    rng = np.random.default_rng(2)
    X = rng.normal(size=(300, len(V))).astype(np.float32)
    targets = rng.choice(len(V), 10, replace=False)
    # targets are copies of sources (+ noise): without exclusion they find themselves
    Y = pd.DataFrame(X[:, targets] + 0.1 * rng.normal(size=(300, 10)).astype(np.float32))
    fitter = ConnectiveFieldFitter(GaussianConnectiveField(D), Y, X)
    est = fitter.fit_grid([1.0, 2.0], progressbar=False)
    assert (est['center_index'].astype(int).values == targets).all()
    excl = exclusion_from_distances(D[targets], radius=3.0)
    est2 = fitter.fit_grid([1.0, 2.0], exclude=excl, progressbar=False)
    c2 = est2['center_index'].astype(int).values
    assert (D[targets, c2] >= 3.0).all()
    assert fitter.grid_r_.abs().max() < 0.4          # nothing left to find
    # dict form gives the same answer
    est3 = fitter.fit_grid([1.0, 2.0], exclude={j: excl[j].indices for j in range(10)},
                           progressbar=False)
    np.testing.assert_array_equal(est3['center_index'].values, est2['center_index'].values)


def test_refine_sigma_and_cv():
    V, F, u, v = grid_mesh(16, 16)
    D = geodesic_distance_matrix(V, F, method='heat', progressbar=False)
    rng = np.random.default_rng(4)
    centres = rng.choice(len(V), 12, replace=False)
    true_sigma = np.full(12, 2.6, dtype=np.float32)
    X, Y, true = simulate(D, centres, true_sigma, noise=0.2, n_samples=400, smooth=1.0, seed=5)
    model = GaussianConnectiveField(D, normalize='sum', truncate=None)
    fitter = ConnectiveFieldFitter(model, Y, X)
    grid = fitter.fit_grid([1.0, 2.0, 4.0], progressbar=False)
    refined = fitter.refine(grid, max_n_iterations=600, min_n_iterations=100,
                            learning_rate=0.02, progressbar=False)
    assert (refined['center_index'].values == grid['center_index'].values).all()
    assert np.median(np.abs(refined['sigma'] - 2.6)) < np.median(np.abs(grid['sigma'] - 2.6))
    assert (fitter.get_rsq(refined).values >= fitter.get_rsq(grid).values - 1e-3).all()

    groups = np.repeat(np.arange(4), 100)
    cv = fitter.fit_cv(groups, [1.0, 2.0, 4.0])
    assert cv.shape == (12, 2)
    assert (cv['cv_r2'] > 0.5).all()
    # noise-only targets get cv_r2 around or below zero
    noise = pd.DataFrame(rng.normal(size=(400, 5)).astype(np.float32))
    cv0 = ConnectiveFieldFitter(model, noise, X).fit_cv(groups, [1.0, 2.0, 4.0])
    assert (cv0['cv_r2'] < 0.05).all()


def test_exclusion_covariance_matches_samples():
    """Covariance-space re-scoring of excluded targets (the default) equals
    rebuilding the pooled signals over samples, including with non-centred,
    spatially smooth sources and two 'hemispheres'."""
    V, F, u, v = grid_mesh(18, 18)
    D1 = geodesic_distance_matrix(V, F, method='heat', progressbar=False)
    D = block_distance_matrix([D1, D1])
    rng = np.random.default_rng(7)
    n = len(V)
    centres = np.r_[rng.choice(n, 10, replace=False), n + rng.choice(n, 10, replace=False)]
    X, Y, _ = simulate(D, centres, rng.choice([1.0, 2.0, 4.0], 20), noise=1.0,
                       smooth=1.5, n_samples=350, seed=8)
    X = X + rng.normal(size=(1, X.shape[1])).astype(np.float32) * 3   # non-zero source means
    excl = exclusion_from_distances(D[centres], radius=2.5)
    model = GaussianConnectiveField(D, normalize='sum', truncate=3.0)
    fitter = ConnectiveFieldFitter(model, Y, X)
    res = {}
    for m in ('samples', 'covariance'):
        est = fitter.fit_grid([1.0, 2.0, 4.0], exclude=excl, exclusion_method=m,
                              target_batch_size=7, progressbar=False)
        res[m] = (est.copy(), fitter.grid_r_.values.copy())
    np.testing.assert_array_equal(res['samples'][0]['center_index'].values,
                                  res['covariance'][0]['center_index'].values)
    np.testing.assert_array_equal(res['samples'][0]['sigma'].values,
                                  res['covariance'][0]['sigma'].values)
    np.testing.assert_allclose(res['samples'][1], res['covariance'][1], atol=2e-4)
    c = res['covariance'][0]['center_index'].astype(int).values
    assert (D[centres, c] >= 2.5).all()
    with pytest.raises(ValueError):
        fitter.fit_grid([1.0], exclude=excl, exclusion_method='nope', progressbar=False)
