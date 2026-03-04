"""Unit tests for braincoder encoding models.

Tests cover: predict/simulate shapes and values, noiseless simulation,
parameter recovery via ParameterFitter, and import paths.
"""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import pearsonr


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def gauss_paradigm():
    """1-D stimulus paradigm covering [-5, 5]."""
    return np.linspace(-5, 5, 100, dtype=np.float32)[:, np.newaxis]


@pytest.fixture
def gauss_parameters():
    """5-voxel GaussianPRF parameters spread across the paradigm range."""
    return pd.DataFrame({
        'mu':        np.linspace(-3, 3, 5, dtype=np.float32),
        'sd':        np.ones(5, dtype=np.float32),
        'amplitude': np.ones(5, dtype=np.float32) * 2,
        'baseline':  np.zeros(5, dtype=np.float32),
    })


@pytest.fixture
def image_paradigm():
    """Small random image stack: 30 timepoints × 8 × 8 pixels."""
    rng = np.random.default_rng(0)
    return rng.standard_normal((30, 8, 8)).astype(np.float32)


@pytest.fixture
def gauss2d_parameters():
    """4-voxel GaussianPRF2D parameters."""
    return pd.DataFrame({
        'x':         np.array([-0.5, 0.0, 0.5, -0.3], dtype=np.float32),
        'y':         np.array([0.0, 0.4, -0.2, 0.1], dtype=np.float32),
        'sd':        np.array([0.3, 0.4, 0.3, 0.35], dtype=np.float32),
        'baseline':  np.zeros(4, dtype=np.float32),
        'amplitude': np.ones(4, dtype=np.float32),
    })


@pytest.fixture
def linear_paradigm():
    """Simple design matrix: 60 timepoints, 3 regressors."""
    rng = np.random.default_rng(1)
    return pd.DataFrame(rng.standard_normal((60, 3)).astype(np.float32))


@pytest.fixture
def linear_weights(linear_paradigm):
    """Random weights mapping 3 regressors → 4 voxels."""
    rng = np.random.default_rng(2)
    n_feat = linear_paradigm.shape[1]
    return pd.DataFrame(rng.standard_normal((n_feat, 4)).astype(np.float32))


# ---------------------------------------------------------------------------
# Import tests
# ---------------------------------------------------------------------------

class TestImports:
    """All public names must be importable from their documented paths."""

    def test_models_package(self):
        from braincoder.models import (
            EncodingModel, EncodingRegressionModel, HRFEncodingModel,
            GaussianPRF, RegressionGaussianPRF, VonMisesPRF, LogGaussianPRF,
            GaussianPRFWithHRF, LogGaussianPRFWithHRF, AlphaGaussianPRF,
            RegressionAlphaGaussianPRF, GaussianPRFOnGaussianSignal,
            GaussianPRF2D, GaussianPRF2DAngle,
            GaussianPRF2DWithHRF, GaussianPRF2DAngleWithHRF,
            DifferenceOfGaussiansPRF2D, DifferenceOfGaussiansPRF2DWithHRF,
            DivisiveNormalizationGaussianPRF2D, DivisiveNormalizationGaussianPRF2DWithHRF,
            DiscreteModel, LinearModel, LinearModelWithBaseline, LinearModelWithBaselineHRF,
        )

    def test_optimize_package(self):
        from braincoder.optimize import (
            WeightFitter, ParameterFitter, ResidualFitter,
            StimulusFitter, CustomStimulusFitter,
            SzinteStimulus, SzinteStimulus2, make_aperture_stimuli,
        )

    def test_top_level_imports(self):
        from braincoder import (
            GaussianPRF, GaussianPRF2D, LinearModel, LinearModelWithBaseline,
            ParameterFitter, WeightFitter,
        )

    def test_submodule_class_identity(self):
        """Classes imported from sub-paths must be the same objects."""
        from braincoder.models import GaussianPRF as A
        from braincoder.models.prf_1d import GaussianPRF as B
        assert A is B

        from braincoder.optimize import ParameterFitter as C
        from braincoder.optimize.parameter_fitter import ParameterFitter as D
        assert C is D


# ---------------------------------------------------------------------------
# GaussianPRF
# ---------------------------------------------------------------------------

class TestGaussianPRF:

    def test_predict_shape(self, gauss_paradigm, gauss_parameters):
        from braincoder.models import GaussianPRF
        model = GaussianPRF(paradigm=gauss_paradigm, parameters=gauss_parameters)
        pred = model.predict()
        assert pred.shape == (len(gauss_paradigm), len(gauss_parameters))

    def test_predict_finite(self, gauss_paradigm, gauss_parameters):
        from braincoder.models import GaussianPRF
        model = GaussianPRF(paradigm=gauss_paradigm, parameters=gauss_parameters)
        pred = model.predict()
        assert np.all(np.isfinite(pred.values))

    def test_simulate_shape(self, gauss_paradigm, gauss_parameters):
        from braincoder.models import GaussianPRF
        model = GaussianPRF(paradigm=gauss_paradigm, parameters=gauss_parameters)
        sim = model.simulate(noise=0.1)
        assert sim.shape == (len(gauss_paradigm), len(gauss_parameters))

    def test_noiseless_simulate_equals_predict(self, gauss_paradigm, gauss_parameters):
        from braincoder.models import GaussianPRF
        model = GaussianPRF(paradigm=gauss_paradigm, parameters=gauss_parameters)
        pred = model.predict()
        sim = model.simulate(noise=0.0)
        np.testing.assert_allclose(sim.values, pred.values, rtol=1e-5, atol=1e-5)

    def test_noise_changes_output(self, gauss_paradigm, gauss_parameters):
        from braincoder.models import GaussianPRF
        model = GaussianPRF(paradigm=gauss_paradigm, parameters=gauss_parameters)
        pred = model.predict()
        sim = model.simulate(noise=1.0)
        assert not np.allclose(sim.values, pred.values)

    def test_predict_peak_at_mu(self, gauss_paradigm):
        """Each voxel's predicted maximum should be closest to its mu."""
        from braincoder.models import GaussianPRF
        mus = np.array([-3.0, 0.0, 3.0], dtype=np.float32)
        parameters = pd.DataFrame({
            'mu': mus,
            'sd': np.ones(3, dtype=np.float32),
            'amplitude': np.ones(3, dtype=np.float32),
            'baseline': np.zeros(3, dtype=np.float32),
        })
        model = GaussianPRF(paradigm=gauss_paradigm, parameters=parameters)
        pred = model.predict()
        stimulus_values = gauss_paradigm[:, 0]
        for i, mu in enumerate(mus):
            peak_stimulus = stimulus_values[pred.iloc[:, i].argmax()]
            assert abs(peak_stimulus - mu) < 0.5


# ---------------------------------------------------------------------------
# GaussianPRF2D
# ---------------------------------------------------------------------------

class TestGaussianPRF2D:

    def test_predict_shape(self, image_paradigm, gauss2d_parameters):
        from braincoder.models import GaussianPRF2D
        model = GaussianPRF2D(paradigm=image_paradigm, parameters=gauss2d_parameters)
        pred = model.predict()
        assert pred.shape == (len(image_paradigm), len(gauss2d_parameters))

    def test_predict_finite(self, image_paradigm, gauss2d_parameters):
        from braincoder.models import GaussianPRF2D
        model = GaussianPRF2D(paradigm=image_paradigm, parameters=gauss2d_parameters)
        pred = model.predict()
        assert np.all(np.isfinite(pred.values))

    def test_simulate_shape(self, image_paradigm, gauss2d_parameters):
        from braincoder.models import GaussianPRF2D
        model = GaussianPRF2D(paradigm=image_paradigm, parameters=gauss2d_parameters)
        sim = model.simulate(noise=0.1)
        assert sim.shape == (len(image_paradigm), len(gauss2d_parameters))

    def test_noiseless_simulate_equals_predict(self, image_paradigm, gauss2d_parameters):
        from braincoder.models import GaussianPRF2D
        model = GaussianPRF2D(paradigm=image_paradigm, parameters=gauss2d_parameters)
        pred = model.predict()
        sim = model.simulate(noise=0.0)
        np.testing.assert_allclose(sim.values, pred.values, rtol=1e-5, atol=1e-5)

    def test_auto_grid_coordinates(self, image_paradigm, gauss2d_parameters):
        """Grid coordinates inferred from paradigm shape should match manual ones."""
        from braincoder.models import GaussianPRF2D
        n_t, h, w = image_paradigm.shape
        grid = np.array(
            np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h)),
            dtype=np.float32,
        )
        grid = np.swapaxes(grid, 2, 1).reshape(2, -1).T

        m_auto = GaussianPRF2D(paradigm=image_paradigm, parameters=gauss2d_parameters)
        m_manual = GaussianPRF2D(
            grid_coordinates=grid, paradigm=image_paradigm, parameters=gauss2d_parameters
        )
        np.testing.assert_allclose(
            m_auto.predict().values, m_manual.predict().values, rtol=1e-5, atol=1e-5
        )


# ---------------------------------------------------------------------------
# LinearModel
# ---------------------------------------------------------------------------

class TestLinearModel:

    def test_predict_shape(self, linear_paradigm, linear_weights):
        from braincoder.models import LinearModel
        model = LinearModel(paradigm=linear_paradigm)
        pred = model.predict(weights=linear_weights)
        assert pred.shape == (len(linear_paradigm), linear_weights.shape[1])

    def test_predict_equals_matmul(self, linear_paradigm, linear_weights):
        """predict() should equal paradigm @ weights exactly."""
        from braincoder.models import LinearModel
        model = LinearModel(paradigm=linear_paradigm)
        pred = model.predict(weights=linear_weights)
        expected = linear_paradigm.values @ linear_weights.values
        np.testing.assert_allclose(pred.values, expected, rtol=1e-5, atol=1e-5)

    def test_rejects_parameters(self, linear_paradigm):
        from braincoder.models import LinearModel
        with pytest.raises(ValueError):
            LinearModel(paradigm=linear_paradigm, parameters=pd.DataFrame({'x': [1.0]}))


# ---------------------------------------------------------------------------
# LinearModelWithBaseline
# ---------------------------------------------------------------------------

class TestLinearModelWithBaseline:

    @pytest.fixture
    def baseline_parameters(self, linear_weights):
        n_vox = linear_weights.shape[1]
        return pd.DataFrame({
            'baseline': np.array([0.5, -0.3, 0.1, 0.8], dtype=np.float32),
        })

    def test_predict_shape(self, linear_paradigm, linear_weights, baseline_parameters):
        from braincoder.models import LinearModelWithBaseline
        model = LinearModelWithBaseline(paradigm=linear_paradigm, parameters=baseline_parameters)
        pred = model.predict(weights=linear_weights)
        assert pred.shape == (len(linear_paradigm), linear_weights.shape[1])

    def test_predict_equals_matmul_plus_baseline(
        self, linear_paradigm, linear_weights, baseline_parameters
    ):
        """predict() should equal (paradigm @ weights) + baseline."""
        from braincoder.models import LinearModelWithBaseline
        model = LinearModelWithBaseline(paradigm=linear_paradigm, parameters=baseline_parameters)
        pred = model.predict(weights=linear_weights)
        expected = (
            linear_paradigm.values @ linear_weights.values
            + baseline_parameters['baseline'].values
        )
        np.testing.assert_allclose(pred.values, expected, rtol=1e-5, atol=1e-5)

    def test_noiseless_simulate_equals_predict(
        self, linear_paradigm, linear_weights, baseline_parameters
    ):
        from braincoder.models import LinearModelWithBaseline
        model = LinearModelWithBaseline(paradigm=linear_paradigm, parameters=baseline_parameters)
        pred = model.predict(weights=linear_weights)
        sim = model.simulate(weights=linear_weights, noise=0.0)
        np.testing.assert_allclose(sim.values, pred.values, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# ParameterFitter (integration)
# ---------------------------------------------------------------------------

class TestParameterFitter:

    def test_gauss_prf_parameter_recovery(self, gauss_paradigm):
        """Fitted mu values should correlate strongly with ground truth."""
        from braincoder.models import GaussianPRF
        from braincoder.optimize import ParameterFitter

        rng = np.random.default_rng(0)
        n_voxels = 15
        true_pars = pd.DataFrame({
            'mu':        np.linspace(-4, 4, n_voxels, dtype=np.float32),
            'sd':        np.ones(n_voxels, dtype=np.float32),
            'amplitude': np.ones(n_voxels, dtype=np.float32),
            'baseline':  np.zeros(n_voxels, dtype=np.float32),
        })

        model = GaussianPRF(paradigm=gauss_paradigm, parameters=true_pars)
        data = model.simulate(noise=0.1)

        fitter = ParameterFitter(model, data, gauss_paradigm)
        init_pars = pd.DataFrame({
            'mu':        np.zeros(n_voxels, dtype=np.float32),
            'sd':        np.ones(n_voxels, dtype=np.float32) * 2,
            'amplitude': np.ones(n_voxels, dtype=np.float32),
            'baseline':  np.zeros(n_voxels, dtype=np.float32),
        })
        estimated = fitter.fit(init_pars=init_pars, max_n_iterations=200, progressbar=False)

        r_mu, _ = pearsonr(true_pars['mu'], estimated['mu'])
        assert r_mu > 0.9, f"mu recovery correlation too low: {r_mu:.3f}"

    def test_fit_grid_gauss_prf(self, gauss_paradigm):
        """fit_grid should return a DataFrame with the right shape."""
        from braincoder.models import GaussianPRF
        from braincoder.optimize import ParameterFitter

        true_pars = pd.DataFrame({
            'mu':        np.linspace(-3, 3, 5, dtype=np.float32),
            'sd':        np.ones(5, dtype=np.float32),
            'amplitude': np.ones(5, dtype=np.float32),
            'baseline':  np.zeros(5, dtype=np.float32),
        })
        model = GaussianPRF(paradigm=gauss_paradigm, parameters=true_pars)
        data = model.simulate(noise=0.2)

        fitter = ParameterFitter(model, data, gauss_paradigm)
        grid_pars = fitter.fit_grid(
            mu=np.linspace(-5, 5, 11, dtype=np.float32),
            sd=np.array([1.0], dtype=np.float32),
            amplitude=np.array([1.0], dtype=np.float32),
            baseline=np.array([0.0], dtype=np.float32),
        )

        assert grid_pars.shape == (5, 4)
        assert list(grid_pars.columns) == ['mu', 'sd', 'amplitude', 'baseline']


# ---------------------------------------------------------------------------
# WeightFitter (integration)
# ---------------------------------------------------------------------------

class TestWeightFitter:

    def test_weight_recovery(self, linear_paradigm, linear_weights):
        """WeightFitter should recover ground-truth weights from noiseless data."""
        from braincoder.models import LinearModel
        from braincoder.optimize import WeightFitter

        model = LinearModel(paradigm=linear_paradigm)
        # LinearModel.simulate() requires parameters; use predict() + manual DataFrame instead
        pred = model.predict(weights=linear_weights)
        data = pd.DataFrame(pred.values, columns=pred.columns, index=pred.index)

        fitter = WeightFitter(model, parameters=None, data=data, paradigm=linear_paradigm)
        estimated_weights = fitter.fit(alpha=0.0)

        np.testing.assert_allclose(
            estimated_weights.values, linear_weights.values, rtol=1e-4, atol=1e-4
        )
