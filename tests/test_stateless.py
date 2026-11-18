"""Models hold configuration only: using them must leave them unchanged."""
import copy
import pickle

import numpy as np
import pandas as pd
import pytest

from braincoder.hrf import SPMHRFModel
from braincoder.models import (GaussianPRF, GaussianPRF2D, GaussianPRF2DWithHRF, GaussianPRFWithHRF,
                               LinearModel, LogGaussianPRF, RegressionGaussianPRF, VonMisesPRF)
from braincoder.noise import LowRankNoise
from braincoder.optimize import ParameterFitter, ResidualFitter, WeightFitter

GRID = np.linspace(0, 40, 41).astype(np.float32)


def configuration(model):
    """Attributes that define a model, leaving out bound methods and internal memos."""
    return {k: v for k, v in vars(model).items() if not callable(v) and not k.startswith('_')}


def assert_unchanged(model, before):
    now = configuration(model)
    assert now.keys() == before.keys()
    for key in before:
        assert pickle.dumps(now[key]) == pickle.dumps(before[key]), key


@pytest.fixture(scope='module')
def sim():
    rng = np.random.default_rng(0)
    pars = pd.DataFrame({'mu': rng.uniform(5, 35, 12), 'sd': rng.uniform(4, 10, 12),
                         'amplitude': 1., 'baseline': 0.},
                        index=pd.Index([f'v{i}' for i in range(12)], name='voxel')).astype(np.float32)
    paradigm = pd.DataFrame({'x': rng.uniform(5, 35, 60).astype(np.float32)},
                            index=pd.Index(np.arange(60), name='trial'))
    data = GaussianPRF().simulate(paradigm, pars, noise=.3)
    data.index = paradigm.index
    return data, paradigm, pars


@pytest.mark.parametrize('kwarg', ['paradigm', 'data', 'parameters', 'weights', 'omega'])
@pytest.mark.parametrize('make', [GaussianPRF, LinearModel, VonMisesPRF, LogGaussianPRF,
                                  lambda **k: GaussianPRFWithHRF(hrf_model=SPMHRFModel(tr=1.), **k),
                                  lambda **k: GaussianPRF2D(grid_coordinates=np.zeros((4, 2)), **k),
                                  lambda **k: GaussianPRF2DWithHRF(grid_coordinates=np.zeros((4, 2)),
                                                                   hrf_model=SPMHRFModel(tr=1.), **k)])
def test_constructors_refuse_state_with_a_pointer_to_the_new_api(make, kwarg):
    with pytest.raises(TypeError, match='no longer takes'):
        make(**{kwarg: 1})


def test_models_have_no_state_attributes(sim):
    data, paradigm, pars = sim
    model = GaussianPRF()
    model.predict(paradigm, pars)
    for name in ('parameters', 'weights', 'omega', 'dof', 'paradigm', 'data', 'omega_chol'):
        assert not hasattr(model, name), name


def test_methods_need_their_arguments(sim):
    data, paradigm, pars = sim
    model = GaussianPRF()
    with pytest.raises(TypeError):
        model.predict()
    with pytest.raises(ValueError, match='needs parameters'):
        model.predict(paradigm)
    with pytest.raises(ValueError, match='needs parameters'):
        model.simulate(paradigm)
    with pytest.raises(ValueError, match='needs parameters'):
        model.get_stimulus_pdf(data, GRID, omega=np.eye(12))
    with pytest.raises(ValueError, match='noise model'):
        model.get_stimulus_pdf(data, GRID, parameters=pars)


def test_full_workflow_leaves_the_model_unchanged(sim):
    data, paradigm, pars = sim
    model = GaussianPRF()
    before = copy.deepcopy(configuration(model))

    fitter = ParameterFitter(model, data, paradigm)
    grid = fitter.fit_grid(mu=np.linspace(5, 35, 7), sd=[5., 10.], amplitude=[1.], baseline=[0.],
                           use_correlation_cost=True)
    fit = fitter.fit(init_pars=fitter.refine_baseline_and_amplitude(grid), max_n_iterations=10,
                     min_n_iterations=5, progressbar=False)
    model.predict(paradigm, fit)
    model.simulate(paradigm, fit, noise=.1)
    omega, dof = ResidualFitter(model, data, paradigm, parameters=fit).fit(
        stimulus_range=GRID, method='t', init_dof=10., max_n_iterations=5, progressbar=False)
    model.get_stimulus_pdf(data, GRID, parameters=fit, omega=omega, dof=dof)
    model.get_pseudoWWT(GRID, fit)
    noise = LowRankNoise.fit(model, data, paradigm, fit, stimulus_range=GRID, groups=np.repeat(np.arange(3), 20),
                             k=0, temperature=1.)
    model.get_stimulus_pdf(data, GRID, parameters=fit, noise_model=noise)

    assert_unchanged(model, before)


def test_fitters_do_not_share_results_through_the_model(sim):
    data, paradigm, pars = sim
    model = GaussianPRF()
    shifted = pars.assign(mu=pars.mu + 5.)
    ResidualFitter(model, data, paradigm, parameters=shifted).fit(stimulus_range=GRID, max_n_iterations=3, progressbar=False)
    np.testing.assert_allclose(model.predict(paradigm, pars).values,
                               GaussianPRF().predict(paradigm, pars).values)


def test_hrf_model_is_not_changed_by_a_flexible_model():
    hrf = SPMHRFModel(tr=1.6)
    before = pickle.dumps({k: v for k, v in vars(hrf).items() if not callable(v) and k != 'transformations'})
    GaussianPRFWithHRF(hrf_model=hrf, flexible_hrf_parameters=True)
    assert pickle.dumps({k: v for k, v in vars(hrf).items() if not callable(v) and k != 'transformations'}) == before


def test_weight_fitter_leaves_the_model_unchanged(sim):
    data, paradigm, _ = sim
    centers = pd.DataFrame({'mu': np.linspace(0, 40, 5), 'sd': 8., 'amplitude': 1., 'baseline': 0.}).astype(np.float32)
    model = GaussianPRF()
    before = copy.deepcopy(configuration(model))
    weights = WeightFitter(model, centers, data, paradigm).fit(alpha=1.)
    assert model.predict(paradigm, centers, weights).shape == data.shape
    assert model.get_WWT(weights).shape == (12, 12)
    assert_unchanged(model, before)


def test_regression_model_predicts_from_the_paradigm_it_is_given():
    rng = np.random.default_rng(1)
    paradigm = pd.DataFrame({'x': rng.uniform(5, 35, 80), 'condition': np.repeat([0., 1.], 40)}).astype(np.float32)
    model = RegressionGaussianPRF(design_data=paradigm, regressors={'mu': '0 + C(condition)'})
    assert ('mu_unbounded', 'C(condition)[0.0]') in model.parameter_labels
    pars = pd.DataFrame(np.zeros((3, len(model.parameter_labels))), columns=model.parameter_labels).astype(np.float32)
    pars[('mu_unbounded', 'C(condition)[0.0]')] = 10.
    pars[('mu_unbounded', 'C(condition)[1.0]')] = 30.
    pars[('sd_unbounded', 'Intercept')] = 5.
    pars[('amplitude_unbounded', 'Intercept')] = 1.

    both = model.predict(paradigm, pars)
    first = model.predict(paradigm.iloc[:40], pars)
    second = model.predict(paradigm.iloc[40:], pars)
    assert both.shape == (80, 3)
    np.testing.assert_allclose(both.values, np.concatenate([first.values, second.values]), atol=1e-5)
    # A different paradigm gives different predictions, whatever was used last.
    assert not np.allclose(model.predict(paradigm.assign(condition=1.), pars).values, both.values)
    fit = ParameterFitter(model, both.iloc[:, :], paradigm).fit(init_pars=pars, max_n_iterations=3,
                                                                min_n_iterations=2, progressbar=False)
    assert fit.shape == pars.shape


def test_linear_and_xy_conversions_return_their_values():
    from braincoder.models import GaussianPRF2DAngle
    g = np.linspace(-5, 5, 12)
    coords = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2).astype(np.float32)
    images = (np.random.default_rng(2).random((20, len(coords))) < .2).astype(np.float32)
    angle = pd.DataFrame({'theta': [0., 1.], 'ecc': [1., 2.], 'sd': [1., 1.5], 'baseline': [0., .5],
                          'amplitude': [1., 2.]}).astype(np.float32)
    model = GaussianPRF2DAngle(grid_coordinates=coords)
    xy_model, xy = model.to_xy_model(angle)
    assert isinstance(xy_model, GaussianPRF2D) and list(xy.columns) == GaussianPRF2D.parameter_labels
    linear, baseline, weights = xy_model.to_linear_model(xy)
    np.testing.assert_allclose(linear.predict(images, baseline, weights).values,
                               xy_model.predict(images, xy).values, atol=1e-3)
