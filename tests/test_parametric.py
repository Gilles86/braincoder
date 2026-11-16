"""Stimulus parameterisations (braincoder.parametric) and the parametric MAP fitter.

Run once per backend: KERAS_BACKEND=tensorflow|jax|torch pytest tests/test_parametric.py
"""
import numpy as np
import pandas as pd
import pytest
from keras import ops

from braincoder.models import GaussianPRF2D, GaussianPRF2DWithHRF
from braincoder.hrf import SPMHRFModel
from braincoder.noise import LowRankNoise
from braincoder.optimize import ParametricStimulusFitter
from braincoder.parametric import (BarStimulus, forward_backward, get_parametric_stimulus_pdf, marginalize)
from braincoder.utils.backend import compute_gradients


def reference_bar(theta, p, width, x, y, aperture=None):
    """The candidate-bank renderer the moving-bar decoding used (paper repo, bar_class.render)."""
    t = np.deg2rad(theta)
    im = (np.abs(x * np.cos(t) + y * np.sin(t) - p) <= width / 2.).astype(np.float32)
    return im if aperture is None else im * aperture


def make_grid(n=24, r=6.):
    x, y = np.meshgrid(np.linspace(-r, r, n), np.linspace(-r, r, n))
    return x.astype(np.float32), y.astype(np.float32)


def toy(n_vox=200, seed=0, n=24, r=6.):
    rng = np.random.default_rng(seed)
    x, y = make_grid(n, r)
    grid = pd.DataFrame({'x': x.ravel(), 'y': y.ravel()})
    pars = pd.DataFrame({'x': rng.uniform(-4.5, 4.5, n_vox), 'y': rng.uniform(-4.5, 4.5, n_vox),
                         'sd': rng.uniform(.8, 2., n_vox), 'baseline': 0., 'amplitude': 1.}).astype(np.float32)
    return x, y, grid, pars


def test_hard_bar_matches_candidate_bank():
    x, y = make_grid(30, 8.)
    ap = (np.sqrt(x ** 2 + y ** 2) <= 7.).astype(np.float32)
    stim = BarStimulus(np.c_[x.ravel(), y.ravel()], edge='hard', aperture=ap.ravel())
    g = stim.grid(orientation=np.arange(0, 180, 15.), position=np.arange(-8, 8.01, .5), width=[1., 2., 3., 4.])
    table, ims = stim.candidates(g, include_blank=True)
    assert np.isnan(table.iloc[0].values).all() and ims[0].sum() == 0
    ref = np.array([reference_bar(r.orientation, r.position, r.width, x, y, ap).ravel() for r in table.iloc[1:].itertuples()])
    assert np.array_equal(ims[1:], ref)
    # keras path (float32): equal except possibly at exact edge ties
    k = np.asarray(ops.convert_to_numpy(stim.to_image(g.values[:200])))
    assert np.mean(k != ref[:200]) < 1e-3


def test_symmetry_and_canonicalize():
    x, y = make_grid()
    for edge in ('hard', 'soft'):
        stim = BarStimulus(np.c_[x.ravel(), y.ravel()], edge=edge)
        a = stim.images(np.array([[30., 2., 1.5]]))
        b = stim.images(np.array([[210., -2., 1.5]]))
        np.testing.assert_allclose(a, b, atol=1e-5)
    c = stim.canonicalize(pd.DataFrame({'orientation': [210., -30., 30.], 'position': [-2., 1., 1.], 'width': 1.}))
    np.testing.assert_allclose(c.orientation, [30., 150., 30.])
    np.testing.assert_allclose(c.position, [2., -1., 1.])


def test_soft_bar_gradients_finite():
    import keras
    x, y = make_grid()
    stim = BarStimulus(np.c_[x.ravel(), y.ravel()], edge='soft', contrast=True)
    phi = np.array([[0., 0., 1., .8], [45., 3., 2., .5], [100., -5., .5, 1.]])   # incl. a centre line through pixels
    u = keras.Variable(stim.to_unconstrained(phi))
    w = ops.convert_to_tensor(np.random.default_rng(0).normal(size=(stim.n_pixels,)).astype(np.float32))
    loss, (g,) = compute_gradients(lambda: ops.sum(stim.to_image(stim.from_unconstrained(u)) * w[None]), [u])
    g = np.asarray(ops.convert_to_numpy(g))
    assert np.isfinite(g).all() and np.abs(g).sum() > 0
    # the soft bar approaches the hard one away from its edges
    hard = BarStimulus(np.c_[x.ravel(), y.ravel()], edge='hard').images(phi[:, :3])
    soft = BarStimulus(np.c_[x.ravel(), y.ravel()], edge='soft', edge_width=.01).images(phi[:, :3])
    assert np.mean(np.abs(hard - soft) > .5) < .01


def test_grid_posterior_recovers_bar():
    x, y, grid, pars = toy()
    model = GaussianPRF2D(grid_coordinates=grid, rf_normalization='peak')
    stim = BarStimulus(grid, edge='hard')
    truth = pd.DataFrame({'orientation': [0., 45., 90., 135.], 'position': [-2., 1., 0., 3.], 'width': [1., 2., 1., 2.]})
    images = stim.images(truth)
    rng = np.random.default_rng(1)
    clean = model.predict(paradigm=images, parameters=pars).values
    noise = LowRankNoise.from_residuals(rng.normal(size=(400, len(pars))) * .3, k=0)
    g = stim.grid(orientation=np.arange(0, 180, 15.), position=np.arange(-5, 5.01, .5), width=[1., 2.])
    for data, sharp in ((clean + rng.normal(size=clean.shape) * 1e-3, True), (clean + rng.normal(size=clean.shape) * .3, False)):
        pdf = get_parametric_stimulus_pdf(model, data, stim, g, parameters=pars, noise_model=noise, include_blank=True,
                                          blank_prior=.2)
        np.testing.assert_allclose(pdf.sum(1), 1.)
        best = pdf.columns.to_frame(index=False).iloc[pdf.values.argmax(1)].reset_index(drop=True)
        if sharp:
            pd.testing.assert_frame_equal(best, truth, check_dtype=False)
        else:
            assert np.abs(best.position - truth.position).max() <= 1.
        m = marginalize(pdf, 'position')
        assert m.shape[1] == len(np.unique(g.position)) + 1          # + blank
        np.testing.assert_allclose(m.sum(1), 1.)
    # equals the same computation done by hand from get_stimulus_pdf
    table, ims = stim.candidates(g)
    lik = model.get_stimulus_pdf(data, ims, parameters=pars, noise_model=noise, normalize=False).values
    manual = lik / lik.sum(1, keepdims=True)
    pdf = get_parametric_stimulus_pdf(model, data, stim, g, parameters=pars, noise_model=noise)
    np.testing.assert_allclose(pdf.values, manual, rtol=1e-6, atol=1e-12)


def test_forward_backward_matches_enumeration():
    import itertools
    rng = np.random.default_rng(0)
    K, n = 3, 5
    E = rng.normal(size=(n, K)) * 2
    Tm = rng.dirichlet(np.ones(K), size=K)
    init = rng.dirichlet(np.ones(K))
    m, logz = forward_backward(E, np.log(Tm), np.log(init))
    marg = np.zeros((n, K))
    Z = 0.
    for s in itertools.product(range(K), repeat=n):
        w = init[s[0]] * np.exp(E[0, s[0]])
        for t in range(1, n):
            w *= Tm[s[t - 1], s[t]] * np.exp(E[t, s[t]])
        Z += w
        for t in range(n):
            marg[t, s[t]] += w
    np.testing.assert_allclose(m, marg / Z, rtol=1e-10)
    np.testing.assert_allclose(logz, np.log(Z), rtol=1e-10)


@pytest.mark.parametrize('noisy', [False, True])
def test_fitter_per_volume_recovers_phi(noisy):
    x, y, grid, pars = toy(300)
    model = GaussianPRF2D(grid_coordinates=grid, rf_normalization='peak')
    stim = BarStimulus(grid, edge='soft', edge_width=.25)
    truth = pd.DataFrame({'orientation': [10., 60., 100., 170.], 'position': [-2.3, 1.2, 0.4, 3.1], 'width': [1.2, 2.1, 1.6, 1.]})
    rng = np.random.default_rng(2)
    data = model.predict(paradigm=stim.images(truth), parameters=pars).values
    sd = .2 if noisy else .01
    data = data + rng.normal(size=data.shape) * sd
    noise = LowRankNoise.from_residuals(rng.normal(size=(500, len(pars))) * sd, k=0)
    start = truth + np.array([8., .6, .4])                       # off-grid start, as from a coarse grid
    fit = ParametricStimulusFitter(model, stim, data, parameters=pars, noise_model=noise)
    est = fit.fit(start, max_n_iterations=600, learning_rate=.05)
    assert fit.costs[-1] < fit.costs[0]
    tol = (2., .1, .15) if not noisy else (6., .4, .5)
    d_or = np.abs((est.orientation - truth.orientation + 90) % 180 - 90)
    assert d_or.max() < tol[0]
    assert np.abs(est.position - truth.position).max() < tol[1]
    assert np.abs(est.width - truth.width).max() < tol[2]


def test_fitter_sequence_with_hrf_positions_only():
    x, y, grid, pars = toy(300)
    tr = 1.5
    model = GaussianPRF2DWithHRF(grid_coordinates=grid, hrf_model=SPMHRFModel(tr=tr), rf_normalization='peak')
    stim = BarStimulus(grid, edge='soft', edge_width=.25)
    n = 40
    truth = pd.DataFrame({'orientation': 30., 'position': np.linspace(-4, 4, n), 'width': 1.5})
    rng = np.random.default_rng(3)
    data = model.predict(paradigm=stim.images(truth), parameters=pars).values + rng.normal(size=(n, len(pars))) * .05
    noise = LowRankNoise.from_residuals(rng.normal(size=(500, len(pars))) * .05, k=0)
    start = truth.assign(position=np.clip(truth.position + rng.normal(size=n) * .8, -5, 5))
    fit = ParametricStimulusFitter(model, stim, data, parameters=pars, noise_model=noise)
    est = fit.fit(start, max_n_iterations=800, learning_rate=.05, fixed=['orientation', 'width'],
                  smoothness={'position': 1.})
    np.testing.assert_allclose(est.orientation, 30., atol=1e-4)
    # the last frames' bars have barely entered the BOLD yet; judge the rest
    err = np.abs(est.position - truth.position).values[:n - 6]
    assert np.median(err) < .15 and err.max() < .6
