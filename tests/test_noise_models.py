"""Tests for braincoder.noise.LowRankNoise and braincoder.utils.calibration.

Run once per backend (the CI matrix sets ``KERAS_BACKEND``). The fitting and
the grid likelihood are numpy (float64); the backend enters through the model
predictions and through ``LowRankNoise.log_prob_backend``.
"""
import numpy as np
import pandas as pd
import pytest
import keras
from keras import ops
from scipy.stats import multivariate_normal, multivariate_t

from braincoder.models import GaussianPRF
from braincoder.noise import LowRankNoise, fit_lowrank_omega
from braincoder.utils.backend import mvn_log_prob, mvt_log_prob, safe_cholesky
from braincoder.utils.calibration import (calibration_summary, hdi_coverage,
                                          hpd_level, log_score, marginalize, pit)

BACKEND = keras.backend.backend()
GRID = np.linspace(-5, 5, 21).astype(np.float32)


def _random_noise(V=30, k=3, seed=0, dof=None):
    rng = np.random.default_rng(seed)
    F = rng.normal(size=(V, k)) * 0.4
    psi = rng.uniform(0.2, 1.0, V)
    s = rng.uniform(0.5, 2.0, V)
    return LowRankNoise(s, F, psi, dof=dof)


def _simulate(seed=0, V=60, k_true=3, n_runs=10, n_per=40, n_test_runs=5,
              amplitude=0.6, shared=0.6, dof=None):
    """1-D Gaussian PRF population with low-rank + diagonal noise."""
    rng = np.random.default_rng(seed)
    pars = pd.DataFrame({'mu': rng.uniform(-5, 5, V), 'sd': rng.uniform(1, 3, V),
                         'amplitude': np.full(V, amplitude),
                         'baseline': np.zeros(V)}).astype(np.float32)
    L = rng.normal(size=(V, k_true))
    L = L / np.sqrt((L ** 2).sum(1, keepdims=True)) * np.sqrt(shared)
    psi = 1 - (L ** 2).sum(1)
    sds = rng.uniform(0.5, 1.5, V)
    model = GaussianPRF(parameters=pars)

    def sim(nr):
        n = nr * n_per
        x = rng.choice(GRID, n)
        mu = model.predict(paradigm=x[:, None], parameters=pars).values
        z = rng.normal(size=(n, k_true)) @ L.T + rng.normal(size=(n, V)) * np.sqrt(psi)
        if dof is not None:
            z = z / np.sqrt(rng.chisquare(dof, size=(n, 1)) / dof)
        Y = pd.DataFrame((mu + z * sds).astype(np.float32), columns=pars.index)
        return Y, pd.DataFrame({'x': x}), np.repeat(np.arange(nr), n_per)

    return model, pars, sim(n_runs), sim(n_test_runs)


# ---------------------------------------------------------------------------
# (a) factored likelihood == dense likelihood
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('dof', [None, 5.0])
def test_woodbury_matches_dense(dof):
    noise = _random_noise(dof=dof)
    omega, dof_ = noise.to_dense()
    omega = omega.astype(np.float64)
    R = np.random.default_rng(1).normal(size=(50, noise.n_voxels)) * 1.5
    if dof is None:
        ref = multivariate_normal(np.zeros(noise.n_voxels), omega).logpdf(R)
    else:
        ref = multivariate_t(np.zeros(noise.n_voxels), omega, df=dof).logpdf(R)
    np.testing.assert_allclose(noise.log_prob(R), ref, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(noise.logdet, np.linalg.slogdet(omega)[1], rtol=1e-6)
    np.testing.assert_allclose(noise.mahalanobis(R),
                               np.einsum('ij,ij->i', R, np.linalg.solve(omega, R.T).T),
                               rtol=1e-6)


@pytest.mark.parametrize('dof', [None, 5.0])
def test_grid_log_likelihood_matches_rowwise(dof):
    noise = _random_noise(dof=dof)
    rng = np.random.default_rng(2)
    Y = rng.normal(size=(7, noise.n_voxels))
    P = rng.normal(size=(11, noise.n_voxels))
    table = noise.grid_log_likelihood(Y, P)
    ref = np.array([[noise.log_prob((y - p)[None])[0] for p in P] for y in Y])
    np.testing.assert_allclose(table, ref, rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize('dof', [None, 5.0])
def test_backend_log_prob_matches_dense_backend(dof):
    """Keras-ops factored log-prob == braincoder's dense mvn/mvt_log_prob."""
    noise = _random_noise(dof=dof)
    omega, _ = noise.to_dense()
    R = np.random.default_rng(3).normal(size=(20, noise.n_voxels)).astype(np.float32)
    L = safe_cholesky(ops.convert_to_tensor(omega), jitter=0.)
    Rt = ops.convert_to_tensor(R)
    dense = mvn_log_prob(Rt, L) if dof is None else mvt_log_prob(Rt, L, dof)
    fact = noise.log_prob_backend(Rt)
    np.testing.assert_allclose(ops.convert_to_numpy(fact), ops.convert_to_numpy(dense),
                               rtol=1e-3, atol=1e-2)
    np.testing.assert_allclose(ops.convert_to_numpy(fact), noise.log_prob(R),
                               rtol=1e-4, atol=1e-3)


def test_rank_zero_is_diagonal():
    rng = np.random.default_rng(4)
    R = rng.normal(size=(200, 10)) * np.arange(1, 11)
    noise = LowRankNoise.from_residuals(R, k=0, dof=None)
    assert noise.k == 0
    omega, dof = noise.to_dense()
    assert dof is None
    np.testing.assert_allclose(np.diag(omega), R.var(0) * (noise.scale[0] ** 2 / R.std(0)[0] ** 2),
                               rtol=1e-4)
    assert np.allclose(omega - np.diag(np.diag(omega)), 0)


def test_from_residuals_recovers_dof_and_legacy_path():
    rng = np.random.default_rng(5)
    V, n, nu = 15, 4000, 5.0
    F = rng.normal(size=(V, 2)) * 0.5
    C = F @ F.T + np.diag(np.full(V, 0.5))
    R = multivariate_t(np.zeros(V), C, df=nu, seed=1).rvs(n)
    noise = LowRankNoise.from_residuals(R, k=2, dof='fit')
    assert 3.5 < noise.dof < 7.5
    omega, dof = fit_lowrank_omega(R, k=2)
    assert omega.shape == (V, V) and abs(dof - noise.dof) < 1e-6


# ---------------------------------------------------------------------------
# (b) nested CV recovers k and gives calibrated held-out coverage
# (c) T ≈ 1 when correctly specified, T > 1 when the tuning is misspecified
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def simulated():
    return _simulate(seed=0)


def test_cv_recovers_rank_and_calibrates(simulated):
    model, pars, (Ytr, Xtr, gtr), (Yte, Xte, _) = simulated
    noise = LowRankNoise.fit(model, Ytr, Xtr, pars, stimulus_range=GRID, groups=gtr)
    # true k = 3; the grid has 2 and 5
    assert noise.k in (2, 5, 10), noise.summary()
    # rank 0 (diagonal) must decode worse than the chosen rank
    assert noise.cv_scores_.loc[0].max() < noise.cv_scores_.loc[noise.k].max()
    assert 0.5 <= noise.temperature <= 2.0, noise.summary()

    pdf = model.get_stimulus_pdf(Yte, GRID, parameters=pars, noise_model=noise)
    summ = calibration_summary(pdf, Xte['x'].values)
    assert 0.88 <= summ['coverage_0.95'] <= 0.99, summ
    assert 0.40 <= summ['coverage_0.5'] <= 0.60, summ
    assert summ['log_score'] > 0
    assert 0.25 <= summ['pit_sd'] <= 0.33
    assert 'chosen by' in noise.summary()


def test_temperature_above_one_when_misspecified(simulated):
    model, pars, (Ytr, Xtr, gtr), (Yte, Xte, _) = simulated
    bad = pars.copy()
    bad['sd'] = bad['sd'] * 0.5          # tuning curves half as wide as the truth
    noise = LowRankNoise.fit(model, Ytr, Xtr, bad, stimulus_range=GRID, groups=gtr)
    assert noise.temperature > 1.4, noise.summary()

    pdf_T = model.get_stimulus_pdf(Yte, GRID, parameters=bad, noise_model=noise)
    pdf_1 = model.get_stimulus_pdf(Yte, GRID, parameters=bad, noise_model=noise,
                                   temperature=1.0)
    cov_T = hdi_coverage(pdf_T, Xte['x'].values)[0.95]
    cov_1 = hdi_coverage(pdf_1, Xte['x'].values)[0.95]
    assert cov_1 < cov_T
    assert cov_T >= 0.88


def test_student_t_noise_and_fixed_k_T(simulated):
    model, pars, _, _ = simulated
    _, _, (Ytr, Xtr, gtr), (Yte, Xte, _) = _simulate(seed=1, dof=4.0)
    noise = LowRankNoise.fit(model, Ytr, Xtr, pars, stimulus_range=GRID, groups=gtr,
                             k_grid=(0, 2, 5))
    assert noise.dof is not None and noise.dof < 15
    fixed = LowRankNoise.fit(model, Ytr, Xtr, pars, k=3, temperature=2.0)
    assert fixed.k == 3 and fixed.temperature == 2.0 and fixed.cv_scores_ is None
    gauss = LowRankNoise.fit(model, Ytr, Xtr, pars, k=3, temperature=1.0, dof=None)
    assert gauss.dof is None


def test_inner_parameters_dict_and_callable(simulated):
    model, pars, (Ytr, Xtr, gtr), _ = simulated
    inner = {g: pars for g in np.unique(gtr)}
    a = LowRankNoise.fit(model, Ytr, Xtr, pars, stimulus_range=GRID, groups=gtr,
                         k_grid=(0, 2, 5))
    b = LowRankNoise.fit(model, Ytr, Xtr, pars, stimulus_range=GRID, groups=gtr,
                         k_grid=(0, 2, 5), inner_parameters=inner)
    c = LowRankNoise.fit(model, Ytr, Xtr, pars, stimulus_range=GRID, groups=gtr,
                         k_grid=(0, 2, 5),
                         inner_parameters=lambda d, p, g: pars)
    pd.testing.assert_frame_equal(a.cv_scores_, b.cv_scores_)
    pd.testing.assert_frame_equal(a.cv_scores_, c.cv_scores_)
    assert b.cv_info_['inner_parameters'] == 'dict'


def test_flat_temperature_gives_uniform_posterior(simulated):
    model, pars, (Ytr, Xtr, gtr), (Yte, Xte, _) = simulated
    noise = LowRankNoise.fit(model, Ytr, Xtr, pars, k=2, temperature=np.inf)
    pdf = model.get_stimulus_pdf(Yte, GRID, parameters=pars, noise_model=noise)
    assert np.allclose(pdf.values, pdf.values[0, 0])
    np.testing.assert_allclose(log_score(pdf, Xte['x'].values), 0, atol=1e-12)


def test_voxel_alignment(simulated):
    model, pars, (Ytr, Xtr, gtr), (Yte, Xte, _) = simulated
    noise = LowRankNoise.fit(model, Ytr, Xtr, pars, k=2, temperature=1.0)
    shuffled = Yte[Yte.columns[::-1]]
    a = noise.grid_log_likelihood(Yte, np.zeros((3, Yte.shape[1])))
    b = noise.grid_log_likelihood(shuffled, np.zeros((3, Yte.shape[1])))
    np.testing.assert_allclose(a, b)
    with pytest.raises(ValueError):
        noise.grid_log_likelihood(Yte.iloc[:, :-1], np.zeros((3, Yte.shape[1] - 1)))


# ---------------------------------------------------------------------------
# (d) backward compatibility of get_stimulus_pdf
# ---------------------------------------------------------------------------

def test_get_stimulus_pdf_backward_compatible(simulated):
    model, pars, (Ytr, Xtr, gtr), (Yte, Xte, _) = simulated
    noise = LowRankNoise.fit(model, Ytr, Xtr, pars, k=2, temperature=1.0, dof=6.0)
    omega, dof = noise.to_dense()

    # old call, no new arguments: must equal a direct dense computation
    pdf_old = model.get_stimulus_pdf(Yte, GRID, parameters=pars, omega=omega, dof=dof)
    pred = model.predict(paradigm=GRID[:, None], parameters=pars).values
    L = safe_cholesky(ops.convert_to_tensor(omega))
    ll = np.stack([ops.convert_to_numpy(mvt_log_prob(
        ops.convert_to_tensor((Yte.values - p).astype(np.float32)), L, dof)) for p in pred], 1)
    ref = np.exp(ll - ll.max(1, keepdims=True))
    ref /= np.trapezoid(ref, GRID, axis=1)[:, None] if hasattr(np, 'trapezoid') else \
        np.trapz(ref, GRID, axis=1)[:, None]
    np.testing.assert_allclose(pdf_old.values, ref, rtol=2e-3, atol=1e-4)
    assert list(pdf_old.columns) == list(GRID)
    assert pdf_old.index.equals(Yte.index)

    # explicit temperature=1 is identical to the default
    pdf_t1 = model.get_stimulus_pdf(Yte, GRID, parameters=pars, omega=omega, dof=dof,
                                    temperature=1.0)
    pd.testing.assert_frame_equal(pdf_old, pdf_t1)

    # the factored noise model gives the same posterior as its dense omega
    pdf_noise = model.get_stimulus_pdf(Yte, GRID, parameters=pars, noise_model=noise)
    np.testing.assert_allclose(pdf_noise.values, pdf_old.values, rtol=5e-3, atol=1e-4)

    # temperature on the dense path == temperature on the factored path
    pdf_dense_T = model.get_stimulus_pdf(Yte, GRID, parameters=pars, omega=omega, dof=dof,
                                         temperature=3.0)
    pdf_noise_T = model.get_stimulus_pdf(Yte, GRID, parameters=pars, noise_model=noise,
                                         temperature=3.0)
    np.testing.assert_allclose(pdf_noise_T.values, pdf_dense_T.values, rtol=5e-3, atol=1e-4)


def test_get_stimulus_pdf_2d_grid_noise_model():
    from braincoder.models import GaussianPointPRF2D
    rng = np.random.default_rng(0)
    V = 12
    pars = pd.DataFrame({'mu_x': rng.uniform(-1, 1, V), 'mu_y': rng.uniform(-1, 1, V),
                         'sd_x': np.full(V, .5), 'sd_y': np.full(V, .5),
                         'amplitude': np.ones(V), 'baseline': np.zeros(V)}).astype(np.float32)
    xs = np.linspace(-1, 1, 5)
    grid2 = np.array([(a, b) for a in xs for b in xs], dtype=np.float32)
    model = GaussianPointPRF2D(parameters=pars)
    Y = pd.DataFrame(rng.normal(size=(8, V)).astype(np.float32))
    noise = LowRankNoise(np.ones(V), rng.normal(size=(V, 2)) * .3, np.full(V, .5))
    pdf = model.get_stimulus_pdf(Y, grid2, noise_model=noise,
                                 include_multidimensional_stimulus_index=True)
    assert pdf.shape == (8, len(grid2))
    omega, _ = noise.to_dense()
    pdf_dense = model.get_stimulus_pdf(Y, grid2, omega=omega,
                                       include_multidimensional_stimulus_index=True)
    np.testing.assert_allclose(pdf.values, pdf_dense.values, rtol=5e-3, atol=1e-5)
    m = marginalize(pdf, 0)
    np.testing.assert_allclose(m.sum(axis=1), 1)
    cov = hdi_coverage(pdf, grid2[:8])
    assert set(cov.index) == {0.5, 0.95}


# ---------------------------------------------------------------------------
# calibration utilities
# ---------------------------------------------------------------------------

def test_calibration_metrics_uniform_for_calibrated_posteriors():
    """Truth drawn from the posterior itself ⇒ PIT / HPD ~ U(0, 1)."""
    rng = np.random.default_rng(0)
    n, S = 4000, 21
    logits = rng.normal(size=(n, S)) * 2
    P = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    truth_idx = np.array([rng.choice(S, p=p) for p in P])
    pdf = pd.DataFrame(P, columns=pd.Index(GRID, name='stimulus'))
    truth = GRID[truth_idx]
    p = pit(pdf, truth, rng=1)
    h = hpd_level(pdf, truth, rng=2)
    for v in (p, h):
        assert abs(v.mean() - 0.5) < 0.02
        assert abs(v.std() - np.sqrt(1 / 12)) < 0.015
    cov = hdi_coverage(pdf, truth)
    assert abs(cov[0.95] - 0.95) < 0.02 and abs(cov[0.5] - 0.5) < 0.03
    # flat posterior has log score 0 relative to uniform
    flat = pd.DataFrame(np.ones((5, S)), columns=GRID)
    np.testing.assert_allclose(log_score(flat, GRID[:5]), 0, atol=1e-12)


def test_calibration_overconfident_detected():
    rng = np.random.default_rng(1)
    n, S = 2000, 21
    truth_idx = rng.integers(0, S, n)
    centre = np.clip(truth_idx + rng.integers(-4, 5, n), 0, S - 1)
    P = np.exp(-0.5 * ((np.arange(S)[None] - centre[:, None]) / 0.7) ** 2)
    pdf = pd.DataFrame(P, columns=GRID)
    s = calibration_summary(pdf, GRID[truth_idx])
    assert s['coverage_0.95'] < 0.8
    assert s['pit_sd'] > 0.32
