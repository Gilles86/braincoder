"""Inputs that used to give plausible-looking wrong results must now be right or raise."""
import warnings

import numpy as np
import pandas as pd
import pytest

from braincoder.hrf import SPMHRFModel
from braincoder.models import (DivisiveNormalizationGaussianPRF2D, DivisiveNormalizationGaussianPRF2DWithHRF, GaussianPRF, GaussianPRF2D,
                               GaussianPRF2DWithHRF, GaussianPRFWithHRF, LinearModel,
                               LinearModelWithBaseline, LinearModelWithBaselineHRF)
from braincoder.optimize import ParameterFitter, ResidualFitter, WeightFitter
from braincoder.utils import get_rsq

SEARCH = dict(mu=np.linspace(5, 35, 11), sd=np.linspace(3, 15, 4), amplitude=[1.], baseline=[0.])
GRID = np.linspace(0, 40, 41).astype(np.float32)


@pytest.fixture(scope='module')
def sim():
    rng = np.random.default_rng(0)
    V, n = 20, 120
    pars = pd.DataFrame({'mu': rng.uniform(5, 35, V), 'sd': rng.uniform(4, 10, V),
                         'amplitude': rng.uniform(.5, 2, V), 'baseline': rng.normal(0, .3, V)},
                        index=pd.Index([f'v{i}' for i in range(V)], name='voxel')).astype(np.float32)
    paradigm = pd.DataFrame({'x': rng.uniform(5, 35, n).astype(np.float32)},
                            index=pd.Index(np.arange(n), name='trial'))
    data = GaussianPRF(parameters=pars).simulate(paradigm=paradigm, parameters=pars, noise=.3)
    data.index = paradigm.index
    return data, paradigm, pars


def true_r2(data, predictions):
    d, p = data.values, predictions.values
    return np.nanmean(1 - ((d - p) ** 2).sum(0) / ((d - d.mean(0)) ** 2).sum(0))


def test_get_rsq_refuses_nonoverlapping_labels(sim):
    data, paradigm, pars = sim
    predictions = GaussianPRF().predict(paradigm, pars)
    assert get_rsq(data, predictions).mean() < 1
    with pytest.raises(ValueError, match='labels'):
        get_rsq(data.set_axis(np.arange(1000, 1000 + len(data)), axis=0), predictions)
    with pytest.raises(ValueError, match='labels'):
        get_rsq(data.set_axis(np.arange(len(pars)), axis=1), predictions)


def test_all_nan_residual_gives_nan_r2(sim):
    data, paradigm, pars = sim
    nan_data = data.copy()
    nan_data.iloc[:, 0] = np.nan
    assert np.isnan(get_rsq(nan_data, GaussianPRF().predict(paradigm, pars)).iloc[0])


def test_numpy_paradigm_takes_the_index_of_the_data(sim):
    data, paradigm, _ = sim
    shifted = data.set_axis(np.arange(1, len(data) + 1), axis=0)
    x = paradigm.values
    fitter = ParameterFitter(GaussianPRF(paradigm=x), shifted, x)
    init = fitter.refine_baseline_and_amplitude(fitter.fit_grid(**SEARCH, use_correlation_cost=True))
    assert fitter.get_rsq(init).mean() > .3
    assert init['amplitude'].nunique() > 1


def test_labelled_paradigm_mismatch_warns_and_pairs_by_position(sim):
    data, paradigm, _ = sim
    shifted = data.set_axis(np.arange(1000, 1000 + len(data)), axis=0)
    with pytest.warns(RuntimeWarning, match='index of the paradigm'):
        fitter = ParameterFitter(GaussianPRF(paradigm=paradigm), shifted, paradigm)
    init = fitter.refine_baseline_and_amplitude(fitter.fit_grid(**SEARCH, use_correlation_cost=True))
    assert fitter.get_rsq(init).mean() > .3


def test_paradigm_length_mismatch_raises(sim):
    data, paradigm, _ = sim
    with pytest.raises(ValueError, match='rows'):
        ParameterFitter(GaussianPRF(paradigm=paradigm), data, paradigm.iloc[:-1])


@pytest.mark.parametrize('noise_model', ['ssq', 'gaussian'])
def test_masked_voxels_are_nan_and_leave_others_alone(sim, noise_model):
    data, paradigm, pars = sim
    dirty = data.copy()
    dirty.iloc[3, 1] = np.nan
    dirty.iloc[:, 2] = 0.
    fitter = ParameterFitter(GaussianPRF(paradigm=paradigm), dirty, paradigm)
    with pytest.warns(RuntimeWarning, match='2 of 20 voxels'):
        fit = fitter.fit(init_pars=pars, max_n_iterations=20, min_n_iterations=20,
                         noise_model=noise_model, progressbar=False)
    assert fit.iloc[[1, 2]].isna().all(axis=None) and fitter.r2.iloc[[1, 2]].isna().all()
    assert fit.drop(index=fit.index[[1, 2]]).notna().all(axis=None)
    assert fitter.r2.mean() > .3
    # the result can be used as a starting point again
    with pytest.warns(RuntimeWarning):
        again = fitter.fit(init_pars=fit, max_n_iterations=5, min_n_iterations=5, progressbar=False)
    assert again.iloc[0].notna().all()


def test_masked_voxels_do_not_poison_shared_parameters(sim):
    data, paradigm, pars = sim
    dirty = data.copy()
    dirty.iloc[:, 0] = np.nan
    fitter = ParameterFitter(GaussianPRF(paradigm=paradigm), dirty, paradigm)
    with pytest.warns(RuntimeWarning):
        fit = fitter.fit(init_pars=pars, shared_pars=['sd'], max_n_iterations=10,
                         min_n_iterations=10, progressbar=False)
    assert fit.iloc[1:].notna().all(axis=None)


def test_refine_explains_zero_amplitude(sim):
    data, paradigm, pars = sim
    fitter = ParameterFitter(GaussianPRF(paradigm=paradigm), data, paradigm)
    with pytest.raises(ValueError, match='nonzero'):
        fitter.refine_baseline_and_amplitude(pars.assign(amplitude=0.))


def test_decoding_refuses_nonfinite_parameters(sim):
    data, _, pars = sim
    bad = pars.copy()
    bad.iloc[0] = np.nan
    omega = np.eye(len(pars), dtype=np.float32)
    with pytest.raises(ValueError, match='NaN or inf'):
        GaussianPRF().get_stimulus_pdf(data, GRID, parameters=bad, omega=omega)


def test_dof_comes_from_the_model_with_omega(sim):
    data, _, pars = sim
    omega = np.eye(len(pars), dtype=np.float32) * .09
    model = GaussianPRF(parameters=pars)
    gaussian = model.get_stimulus_pdf(data, GRID, omega=omega)
    student = model.get_stimulus_pdf(data, GRID, omega=omega, dof=3.)
    assert not np.allclose(gaussian.values, student.values)
    model.omega, model.dof = omega, 3.
    assert np.allclose(model.get_stimulus_pdf(data, GRID).values, student.values)
    assert np.allclose(model.get_stimulus_pdf(data, GRID, omega=omega).values, gaussian.values)


def test_constructor_accepts_slightly_non_psd_omega(sim):
    _, _, pars = sim
    omega = np.eye(len(pars))
    omega[0, 0] = -1e-4
    model = GaussianPRF(parameters=pars, omega=omega)
    assert np.isfinite(model.omega_chol).all()


def test_hrf_model_decodes_like_its_plain_twin(sim):
    data, _, pars = sim
    omega = np.eye(len(pars), dtype=np.float32) * .09
    plain = GaussianPRF(parameters=pars)
    hrf = GaussianPRFWithHRF(hrf_model=SPMHRFModel(tr=1.6), parameters=pars)
    with pytest.warns(UserWarning, match='ignores the HRF'):
        decoded = hrf.get_stimulus_pdf(data, GRID, parameters=pars, omega=omega)
    assert np.allclose(decoded.values, plain.get_stimulus_pdf(data, GRID, parameters=pars, omega=omega).values)
    with pytest.warns(UserWarning):
        grid_pred = hrf.predict_stimulus_grid(GRID, parameters=pars)
    assert np.allclose(grid_pred, plain.predict_stimulus_grid(GRID, parameters=pars), atol=1e-5)


def test_hrf_decoding_with_flexible_hrf_columns(sim):
    data, _, pars = sim
    omega = np.eye(len(pars), dtype=np.float32) * .09
    flexible = pars.assign(hrf_delay=5., hrf_dispersion=1.).astype(np.float32)
    hrf = GaussianPRFWithHRF(hrf_model=SPMHRFModel(tr=1.6), flexible_hrf_parameters=True, parameters=flexible)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        decoded = hrf.get_stimulus_pdf(data, GRID, parameters=flexible, omega=omega)
    plain = GaussianPRF(parameters=pars).get_stimulus_pdf(data, GRID, parameters=pars, omega=omega)
    assert np.allclose(decoded.values, plain.values)


@pytest.mark.parametrize('method', ['get_fisher_information', 'get_expected_uncertainty'])
def test_hrf_models_refuse_stimulus_wise_analyses(sim, method):
    _, _, pars = sim
    hrf = GaussianPRFWithHRF(hrf_model=SPMHRFModel(tr=1.6), parameters=pars)
    with pytest.raises(NotImplementedError, match='without HRF'):
        getattr(hrf, method)(GRID, omega=np.eye(len(pars), dtype=np.float32))


def test_to_linear_model_on_hrf_model_matches_the_prediction():
    rng = np.random.default_rng(1)
    g = np.linspace(-5, 5, 15)
    coords = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2).astype(np.float32)
    image = (rng.random((80, len(coords))) < .1).astype(np.float32)
    pars = pd.DataFrame({'x': rng.uniform(-3, 3, 6), 'y': rng.uniform(-3, 3, 6), 'sd': rng.uniform(1, 2, 6),
                         'baseline': rng.normal(0, 1, 6), 'amplitude': rng.uniform(1, 2, 6)}).astype(np.float32)
    model = GaussianPRF2DWithHRF(grid_coordinates=coords, paradigm=image, hrf_model=SPMHRFModel(tr=1.6), parameters=pars)
    linear = model.to_linear_model()
    assert isinstance(linear, LinearModelWithBaselineHRF)
    assert np.allclose(linear.predict().values, model.predict().values, atol=1e-3)


def test_get_rf_uses_the_parameters_it_is_given():
    g = np.linspace(-5, 5, 15)
    coords = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2).astype(np.float32)
    pars = pd.DataFrame({'x': [0.], 'y': [0.], 'sd': [1.], 'baseline': [0.], 'amplitude': [1.]}).astype(np.float32)
    model = GaussianPRF2D(grid_coordinates=coords, parameters=pars,
                          paradigm=np.zeros((2, len(coords)), dtype=np.float32))
    assert not np.allclose(model.get_rf(parameters=pars.assign(x=3.)), model.get_rf())


def test_dn_hrf_model_decodes_without_the_hrf():
    g = np.linspace(-5, 5, 15)
    coords = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2).astype(np.float32)
    pars = pd.DataFrame({'x': [0., 1.], 'y': [0., 1.], 'sd': [1., 1.], 'rf_amplitude': [1., 1.],
                         'srf_amplitude': [.1, .1], 'srf_size': [2., 2.], 'neural_baseline': [1., 1.],
                         'surround_baseline': [1., 1.], 'bold_baseline': [.5, .5]}).astype(np.float32)
    images = (np.random.default_rng(5).random((4, len(coords))) < .3).astype(np.float32)
    model = DivisiveNormalizationGaussianPRF2DWithHRF(grid_coordinates=coords, paradigm=images,
                                                      hrf_model=SPMHRFModel(tr=1.6), parameters=pars)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pred = model.predict_stimulus_grid(images, parameters=pars)
    assert pred.shape == (4, 2) and np.isfinite(pred).all() and pred.std() > 1e-3


def test_dn_denominator_uses_absolute_amplitude():
    g = np.linspace(-5, 5, 15)
    coords = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2).astype(np.float32)
    images = np.ones((1, len(coords)), dtype=np.float32)
    base = dict(x=0., y=0., sd=2., srf_amplitude=5., srf_size=2., neural_baseline=0., surround_baseline=1.)
    model = DivisiveNormalizationGaussianPRF2D(grid_coordinates=coords, paradigm=images)
    positive = model.predict(paradigm=images, parameters=pd.DataFrame([{**base, 'rf_amplitude': 1.}]).astype(np.float32))
    negative = model.predict(paradigm=images, parameters=pd.DataFrame([{**base, 'rf_amplitude': -1.}]).astype(np.float32))
    assert np.allclose(negative.values, -positive.values, atol=1e-5)


def test_linear_model_simulate_without_parameters():
    rng = np.random.default_rng(2)
    X = pd.DataFrame(rng.normal(size=(50, 2)).astype(np.float32), columns=['a', 'b'])
    weights = pd.DataFrame(rng.normal(size=(2, 5)).astype(np.float32))
    assert LinearModel(paradigm=X, weights=weights).simulate(noise=.1).shape == (50, 5)


def test_residual_fitter_pairs_by_position_and_finds_bad_input(sim):
    data, paradigm, pars = sim
    model = GaussianPRF(paradigm=paradigm)
    model.init_pseudoWWT(GRID, pars)
    shifted = data.set_axis(np.arange(1000, 1000 + len(data)), axis=0)
    omega, _ = ResidualFitter(model, shifted, paradigm, parameters=pars).fit(max_n_iterations=5, progressbar=False)
    assert np.isfinite(omega).all()
    dirty = data.copy()
    dirty.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match='NaN or inf'):
        ResidualFitter(model, dirty, paradigm, parameters=pars).fit(max_n_iterations=5, progressbar=False)
    with pytest.raises(ValueError, match='no voxels'):
        ResidualFitter(model, data.iloc[:, :0], paradigm, parameters=pars).fit(max_n_iterations=5, progressbar=False)


def test_weight_fitter_intercept_recovers_slope():
    rng = np.random.default_rng(3)
    x = rng.uniform(2, 8, (200, 1)).astype(np.float32)        # all positive, far from mean zero
    slopes = rng.normal(0, 1, 4)
    offsets = rng.uniform(5, 20, 4)
    y = pd.DataFrame(x @ slopes[None] + offsets + rng.normal(0, .1, (200, 4))).astype(np.float32)
    model = LinearModel(paradigm=x)
    through_origin = WeightFitter(model, None, y, x).fit()
    weights, intercept = WeightFitter(model, None, y, x).fit(fit_intercept=True)
    assert np.allclose(weights.values[0], slopes, atol=.03)
    assert np.allclose(intercept.values, offsets, atol=.2)
    assert not np.allclose(through_origin.values[0], slopes, atol=.5)


def test_weight_fitter_subtracts_known_baseline():
    rng = np.random.default_rng(4)
    x = rng.uniform(2, 8, (200, 1)).astype(np.float32)
    baseline = pd.DataFrame({'baseline': [10., 20.]}).astype(np.float32)
    y = pd.DataFrame(x @ np.array([[1., -2.]]) + baseline['baseline'].values + rng.normal(0, .1, (200, 2))).astype(np.float32)
    weights = WeightFitter(LinearModelWithBaseline(paradigm=x), baseline, y, x).fit()
    assert np.allclose(weights.values[0], [1., -2.], atol=.03)
