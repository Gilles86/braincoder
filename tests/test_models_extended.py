"""Extended unit tests for braincoder models and optimizers.

Covers: VonMisesPRF, LogGaussianPRF, AlphaGaussianPRF, GaussianPRFWithHRF,
GaussianPRF2DAngle, DifferenceOfGaussiansPRF2D, simulate() behaviour,
and ResidualFitter.
"""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import pearsonr


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def gauss_paradigm():
    return np.linspace(-5, 5, 100, dtype=np.float32)[:, np.newaxis]


@pytest.fixture
def log_paradigm():
    """Positive-valued paradigm for log-normal models."""
    return np.linspace(0.2, 10, 100, dtype=np.float32)[:, np.newaxis]


@pytest.fixture
def circular_paradigm():
    """Angles 0 → 2π for VonMisesPRF."""
    return np.linspace(0, 2 * np.pi, 80, dtype=np.float32)[:, np.newaxis]


@pytest.fixture
def image_paradigm():
    rng = np.random.default_rng(0)
    return rng.standard_normal((40, 10, 10)).astype(np.float32)


@pytest.fixture
def spm_hrf():
    from braincoder.hrf import SPMHRFModel
    return SPMHRFModel(tr=1.0)


# ---------------------------------------------------------------------------
# VonMisesPRF
# ---------------------------------------------------------------------------

class TestVonMisesPRF:

    @pytest.fixture
    def model(self, circular_paradigm):
        from braincoder.models import VonMisesPRF
        pars = pd.DataFrame({
            'mu':        np.array([0.0, np.pi, np.pi / 2], dtype=np.float32),
            'kappa':     np.array([3., 3., 3.], dtype=np.float32),
            'amplitude': np.ones(3, dtype=np.float32),
            'baseline':  np.zeros(3, dtype=np.float32),
        })
        return VonMisesPRF(paradigm=circular_paradigm, parameters=pars)

    def test_predict_shape(self, model, circular_paradigm):
        pred = model.predict()
        assert pred.shape == (len(circular_paradigm), 3)

    def test_predict_finite(self, model):
        assert np.all(np.isfinite(model.predict().values))

    def test_simulate_shape(self, model, circular_paradigm):
        sim = model.simulate(noise=0.1)
        assert sim.shape == (len(circular_paradigm), 3)

    def test_noiseless_simulate_equals_predict(self, model):
        pred = model.predict()
        sim = model.simulate(noise=0.0)
        np.testing.assert_allclose(sim.values, pred.values, rtol=1e-5, atol=1e-5)

    def test_peak_near_mu(self, circular_paradigm):
        """Each voxel should peak at its preferred angle (mu)."""
        from braincoder.models import VonMisesPRF
        mus = np.array([0.0, np.pi, np.pi / 2], dtype=np.float32)
        pars = pd.DataFrame({
            'mu': mus, 'kappa': [5., 5., 5.],
            'amplitude': [1., 1., 1.], 'baseline': [0., 0., 0.],
        }, dtype=np.float32)
        m = VonMisesPRF(paradigm=circular_paradigm, parameters=pars)
        pred = m.predict()
        stim = circular_paradigm[:, 0]
        for i, mu in enumerate(mus):
            peak = stim[pred.iloc[:, i].argmax()]
            # Allow a half-bin tolerance
            bin_width = stim[1] - stim[0]
            assert abs(peak - mu) < 2 * bin_width

    def test_amplitude_scales_response(self, circular_paradigm):
        from braincoder.models import VonMisesPRF
        pars_low = pd.DataFrame({'mu': [0.], 'kappa': [3.], 'amplitude': [1.], 'baseline': [0.]}, dtype=np.float32)
        pars_high = pd.DataFrame({'mu': [0.], 'kappa': [3.], 'amplitude': [2.], 'baseline': [0.]}, dtype=np.float32)
        pred_low = VonMisesPRF(paradigm=circular_paradigm, parameters=pars_low).predict()
        pred_high = VonMisesPRF(paradigm=circular_paradigm, parameters=pars_high).predict()
        np.testing.assert_allclose(pred_high.values, 2 * pred_low.values, rtol=1e-5)


# ---------------------------------------------------------------------------
# AxialVonMisesPRF
# ---------------------------------------------------------------------------

class TestAxialVonMisesPRF:
    """Tests for the π-periodic (axial) Von Mises pRF.

    Key property: because gabor orientations are π-periodic, a voxel tuned
    to μ should respond identically to orientations μ and μ + π.
    """

    @pytest.fixture
    def orientation_paradigm(self):
        """Orientations 0 → π (half-circle, gabor space)."""
        return np.linspace(0, np.pi, 80, dtype=np.float32)[:, np.newaxis]

    @pytest.fixture
    def model(self, orientation_paradigm):
        from braincoder.models import AxialVonMisesPRF
        pars = pd.DataFrame({
            'mu':        np.array([0.0, np.pi / 4, np.pi / 2], dtype=np.float32),
            'kappa':     np.array([3., 3., 3.], dtype=np.float32),
            'amplitude': np.ones(3, dtype=np.float32),
            'baseline':  np.zeros(3, dtype=np.float32),
        })
        return AxialVonMisesPRF(paradigm=orientation_paradigm, parameters=pars)

    def test_predict_shape(self, model, orientation_paradigm):
        pred = model.predict()
        assert pred.shape == (len(orientation_paradigm), 3)

    def test_predict_finite(self, model):
        assert np.all(np.isfinite(model.predict().values))

    def test_pi_periodicity(self):
        """Core correctness: pdf(x=0, mu=0) must equal pdf(x=π, mu=0)."""
        from braincoder.utils.math import axial_von_mises_pdf
        kappa = np.float32(3.0)
        v0  = float(axial_von_mises_pdf(np.float32(0.0),    np.float32(0.0), kappa))
        vpi = float(axial_von_mises_pdf(np.float32(np.pi),  np.float32(0.0), kappa))
        np.testing.assert_allclose(v0, vpi, rtol=1e-5,
            err_msg='axial_von_mises_pdf must be π-periodic: pdf(0,mu=0) != pdf(π,mu=0)')

    def test_minimum_at_quarter_period(self):
        """Minimum should be at x = μ + π/2 (quarter-period away)."""
        from braincoder.utils.math import axial_von_mises_pdf
        kappa = np.float32(3.0)
        v_peak = float(axial_von_mises_pdf(np.float32(0.0),      np.float32(0.0), kappa))
        v_min  = float(axial_von_mises_pdf(np.float32(np.pi / 2), np.float32(0.0), kappa))
        assert v_peak > v_min

    def test_pi_periodic_differs_from_2pi_periodic(self):
        """AxialVonMisesPRF must give a different (correct) result than VonMisesPRF at x=π."""
        from braincoder.utils.math import axial_von_mises_pdf, von_mises_pdf
        kappa = np.float32(3.0)
        axial_at_pi = float(axial_von_mises_pdf(np.float32(np.pi), np.float32(0.0), kappa))
        old_at_pi   = float(von_mises_pdf(np.float32(np.pi),       np.float32(0.0), kappa))
        # old formula puts the minimum at π; axial puts the peak there
        assert axial_at_pi > old_at_pi

    def test_peak_at_mu_and_mu_plus_pi(self, orientation_paradigm):
        """Each voxel should peak at both μ and μ + π (mod π = μ)."""
        from braincoder.models import AxialVonMisesPRF
        mu = np.float32(np.pi / 4)
        pars = pd.DataFrame({
            'mu': [mu], 'kappa': [5.], 'amplitude': [1.], 'baseline': [0.],
        }, dtype=np.float32)
        m = AxialVonMisesPRF(paradigm=orientation_paradigm, parameters=pars)
        pred = m.predict().iloc[:, 0]
        stim = orientation_paradigm[:, 0]
        bin_width = stim[1] - stim[0]
        peak_x = stim[pred.argmax()]
        assert abs(peak_x - float(mu)) < 2 * bin_width

    def test_normalisation(self, orientation_paradigm):
        """PDF should integrate to ~1 over [0, π)."""
        from braincoder.utils.math import axial_von_mises_pdf
        x = orientation_paradigm[:, 0].astype(np.float32)
        kappa = np.float32(2.0)
        mu    = np.float32(np.pi / 4)
        pdf_vals = np.array([float(axial_von_mises_pdf(xi, mu, kappa)) for xi in x])
        integral = np.trapz(pdf_vals, x)
        np.testing.assert_allclose(integral, 1.0, atol=0.02,
            err_msg='axial_von_mises_pdf should integrate to ~1 over [0, π)')


# ---------------------------------------------------------------------------
# LogGaussianPRF
# ---------------------------------------------------------------------------

class TestLogGaussianPRF:

    def test_predict_shape(self, log_paradigm):
        from braincoder.models import LogGaussianPRF
        pars = pd.DataFrame({'mu': [1., 2., 3.], 'sd': [0.5, 0.5, 0.5],
                             'amplitude': [1., 1., 1.], 'baseline': [0., 0., 0.]}, dtype=np.float32)
        m = LogGaussianPRF(paradigm=log_paradigm, parameters=pars)
        assert m.predict().shape == (len(log_paradigm), 3)

    def test_predict_finite(self, log_paradigm):
        from braincoder.models import LogGaussianPRF
        pars = pd.DataFrame({'mu': [1., 2.], 'sd': [0.5, 0.5],
                             'amplitude': [1., 1.], 'baseline': [0., 0.]}, dtype=np.float32)
        pred = LogGaussianPRF(paradigm=log_paradigm, parameters=pars).predict()
        assert np.all(np.isfinite(pred.values))

    def test_noiseless_simulate_equals_predict(self, log_paradigm):
        from braincoder.models import LogGaussianPRF
        pars = pd.DataFrame({'mu': [2.], 'sd': [0.5],
                             'amplitude': [1.], 'baseline': [0.]}, dtype=np.float32)
        m = LogGaussianPRF(paradigm=log_paradigm, parameters=pars)
        np.testing.assert_allclose(m.simulate(noise=0.).values, m.predict().values, atol=1e-5)

    def test_peak_increases_with_mu(self, log_paradigm):
        """Higher mu → peak at larger stimulus value."""
        from braincoder.models import LogGaussianPRF
        pars = pd.DataFrame({'mu': [0.5, 1.5, 2.5], 'sd': [0.3, 0.3, 0.3],
                             'amplitude': [1., 1., 1.], 'baseline': [0., 0., 0.]}, dtype=np.float32)
        pred = LogGaussianPRF(paradigm=log_paradigm, parameters=pars).predict()
        peak_locs = log_paradigm[pred.values.argmax(0), 0]
        assert peak_locs[0] < peak_locs[1] < peak_locs[2]

    def test_mode_fwhm_parameterisation(self, log_paradigm):
        """mode_fwhm_natural parameterisation produces valid finite predictions."""
        from braincoder.models import LogGaussianPRF
        pars = pd.DataFrame({'mode': [2., 4.], 'fwhm': [1., 1.5],
                             'amplitude': [1., 1.], 'baseline': [0., 0.]}, dtype=np.float32)
        m = LogGaussianPRF(paradigm=log_paradigm, parameters=pars,
                           parameterisation='mode_fwhm_natural')
        pred = m.predict()
        assert pred.shape == (len(log_paradigm), 2)
        assert np.all(np.isfinite(pred.values))


# ---------------------------------------------------------------------------
# AlphaGaussianPRF
# ---------------------------------------------------------------------------

class TestAlphaGaussianPRF:

    @pytest.fixture
    def alpha_paradigm(self):
        return np.linspace(1, 10, 80, dtype=np.float32)[:, np.newaxis]

    def test_predict_shape(self, alpha_paradigm):
        from braincoder.models import AlphaGaussianPRF
        pars = pd.DataFrame({'mu': [3., 5., 7.], 'sd': [1., 1., 1.], 'alpha': [0.5, 0.5, 0.5],
                             'amplitude': [1., 1., 1.], 'baseline': [0., 0., 0.]}, dtype=np.float32)
        m = AlphaGaussianPRF(paradigm=alpha_paradigm, parameters=pars)
        assert m.predict().shape == (len(alpha_paradigm), 3)

    def test_predict_finite(self, alpha_paradigm):
        from braincoder.models import AlphaGaussianPRF
        pars = pd.DataFrame({'mu': [5.], 'sd': [1.], 'alpha': [0.5],
                             'amplitude': [1.], 'baseline': [0.]}, dtype=np.float32)
        pred = AlphaGaussianPRF(paradigm=alpha_paradigm, parameters=pars).predict()
        assert np.all(np.isfinite(pred.values))

    def test_noiseless_simulate_equals_predict(self, alpha_paradigm):
        from braincoder.models import AlphaGaussianPRF
        pars = pd.DataFrame({'mu': [5.], 'sd': [1.], 'alpha': [0.5],
                             'amplitude': [1.], 'baseline': [0.]}, dtype=np.float32)
        m = AlphaGaussianPRF(paradigm=alpha_paradigm, parameters=pars)
        np.testing.assert_allclose(m.simulate(noise=0.).values, m.predict().values, atol=1e-5)

    def test_alpha_zero_stability(self, alpha_paradigm):
        """alpha very near 0 should not produce NaN (log fallback path)."""
        from braincoder.models import AlphaGaussianPRF
        pars = pd.DataFrame({'mu': [5.], 'sd': [1.], 'alpha': [0.001],
                             'amplitude': [1.], 'baseline': [0.]}, dtype=np.float32)
        pred = AlphaGaussianPRF(paradigm=alpha_paradigm, parameters=pars).predict()
        assert np.all(np.isfinite(pred.values))

    def test_peak_tracks_mu(self, alpha_paradigm):
        """Peak response location should track mu."""
        from braincoder.models import AlphaGaussianPRF
        mus = np.array([3., 5., 7.], dtype=np.float32)
        pars = pd.DataFrame({'mu': mus, 'sd': [1., 1., 1.], 'alpha': [0.5, 0.5, 0.5],
                             'amplitude': [1., 1., 1.], 'baseline': [0., 0., 0.]}, dtype=np.float32)
        pred = AlphaGaussianPRF(paradigm=alpha_paradigm, parameters=pars).predict()
        peak_locs = alpha_paradigm[pred.values.argmax(0), 0]
        assert peak_locs[0] < peak_locs[1] < peak_locs[2]


# ---------------------------------------------------------------------------
# GaussianPRFWithHRF
# ---------------------------------------------------------------------------

class TestGaussianPRFWithHRF:

    def test_predict_shape(self, gauss_paradigm, spm_hrf):
        from braincoder.models import GaussianPRFWithHRF
        pars = pd.DataFrame({'mu': [-2., 0., 2.], 'sd': [1., 1., 1.],
                             'amplitude': [1., 1., 1.], 'baseline': [0., 0., 0.]}, dtype=np.float32)
        m = GaussianPRFWithHRF(paradigm=gauss_paradigm, parameters=pars, hrf_model=spm_hrf)
        assert m.predict().shape == (len(gauss_paradigm), 3)

    def test_predict_finite(self, gauss_paradigm, spm_hrf):
        from braincoder.models import GaussianPRFWithHRF
        pars = pd.DataFrame({'mu': [0.], 'sd': [1.], 'amplitude': [1.], 'baseline': [0.]}, dtype=np.float32)
        pred = GaussianPRFWithHRF(paradigm=gauss_paradigm, parameters=pars, hrf_model=spm_hrf).predict()
        assert np.all(np.isfinite(pred.values))

    def test_hrf_changes_predictions(self, gauss_paradigm, spm_hrf):
        """HRF-convolved predictions should differ from plain GaussianPRF."""
        from braincoder.models import GaussianPRF, GaussianPRFWithHRF
        pars = pd.DataFrame({'mu': [0., 2.], 'sd': [1., 1.],
                             'amplitude': [1., 1.], 'baseline': [0., 0.]}, dtype=np.float32)
        pred_plain = GaussianPRF(paradigm=gauss_paradigm, parameters=pars).predict()
        pred_hrf = GaussianPRFWithHRF(paradigm=gauss_paradigm, parameters=pars, hrf_model=spm_hrf).predict()
        assert not np.allclose(pred_plain.values, pred_hrf.values)

    def test_noiseless_simulate_equals_predict(self, gauss_paradigm, spm_hrf):
        from braincoder.models import GaussianPRFWithHRF
        pars = pd.DataFrame({'mu': [0.], 'sd': [1.], 'amplitude': [1.], 'baseline': [0.]}, dtype=np.float32)
        m = GaussianPRFWithHRF(paradigm=gauss_paradigm, parameters=pars, hrf_model=spm_hrf)
        np.testing.assert_allclose(m.simulate(noise=0.).values, m.predict().values, atol=1e-5)

    def test_requires_hrf_model(self, gauss_paradigm):
        from braincoder.models import GaussianPRFWithHRF
        pars = pd.DataFrame({'mu': [0.], 'sd': [1.], 'amplitude': [1.], 'baseline': [0.]}, dtype=np.float32)
        with pytest.raises((ValueError, TypeError)):
            GaussianPRFWithHRF(paradigm=gauss_paradigm, parameters=pars)


# ---------------------------------------------------------------------------
# GaussianPRF2DAngle
# ---------------------------------------------------------------------------

class TestGaussianPRF2DAngle:

    @pytest.fixture
    def model(self, image_paradigm):
        from braincoder.models import GaussianPRF2DAngle
        pars = pd.DataFrame({
            'theta':     np.array([0.0, np.pi / 2, np.pi], dtype=np.float32),
            'ecc':       np.array([0.3, 0.4, 0.3], dtype=np.float32),
            'sd':        np.array([0.3, 0.3, 0.3], dtype=np.float32),
            'baseline':  np.zeros(3, dtype=np.float32),
            'amplitude': np.ones(3, dtype=np.float32),
        })
        return GaussianPRF2DAngle(paradigm=image_paradigm, parameters=pars)

    def test_predict_shape(self, model, image_paradigm):
        assert model.predict().shape == (len(image_paradigm), 3)

    def test_predict_finite(self, model):
        assert np.all(np.isfinite(model.predict().values))

    def test_noiseless_simulate_equals_predict(self, model):
        np.testing.assert_allclose(
            model.simulate(noise=0.).values, model.predict().values, atol=1e-5
        )

    def test_polar_to_cartesian_consistency(self, image_paradigm):
        """RF peak for theta=0, ecc=r should be close to (r, 0) in grid space."""
        from braincoder.models import GaussianPRF2DAngle
        r = 0.3
        pars_angle = pd.DataFrame({'theta': [0.], 'ecc': [r], 'sd': [0.3],
                                   'baseline': [0.], 'amplitude': [1.]}, dtype=np.float32)
        model = GaussianPRF2DAngle(paradigm=image_paradigm, parameters=pars_angle)
        rf = model.get_rf()  # shape: (n_voxels, n_grid_spaces)
        peak_idx = np.argmax(rf[0])
        peak_x = model.grid_coordinates['x'].values[peak_idx]
        peak_y = model.grid_coordinates['y'].values[peak_idx]
        assert abs(peak_x - r) < 0.3, f"RF peak x={peak_x:.3f}, expected ~{r}"
        assert abs(peak_y) < 0.3, f"RF peak y={peak_y:.3f}, expected ~0"


# ---------------------------------------------------------------------------
# DifferenceOfGaussiansPRF2D
# ---------------------------------------------------------------------------

class TestDifferenceOfGaussiansPRF2D:

    @pytest.fixture
    def dog_pars(self):
        return pd.DataFrame({
            'x':            np.array([0., 0.3], dtype=np.float32),
            'y':            np.array([0., 0.], dtype=np.float32),
            'sd':           np.array([0.25, 0.25], dtype=np.float32),
            'baseline':     np.zeros(2, dtype=np.float32),
            'amplitude':    np.ones(2, dtype=np.float32),
            'srf_amplitude': np.array([0.3, 0.3], dtype=np.float32),
            'srf_size':     np.array([2., 2.], dtype=np.float32),
        })

    def test_predict_shape(self, image_paradigm, dog_pars):
        from braincoder.models import DifferenceOfGaussiansPRF2D
        m = DifferenceOfGaussiansPRF2D(paradigm=image_paradigm, parameters=dog_pars)
        assert m.predict().shape == (len(image_paradigm), 2)

    def test_predict_finite(self, image_paradigm, dog_pars):
        from braincoder.models import DifferenceOfGaussiansPRF2D
        pred = DifferenceOfGaussiansPRF2D(paradigm=image_paradigm, parameters=dog_pars).predict()
        assert np.all(np.isfinite(pred.values))

    def test_noiseless_simulate_equals_predict(self, image_paradigm, dog_pars):
        from braincoder.models import DifferenceOfGaussiansPRF2D
        m = DifferenceOfGaussiansPRF2D(paradigm=image_paradigm, parameters=dog_pars)
        np.testing.assert_allclose(m.simulate(noise=0.).values, m.predict().values, atol=1e-5)

    def test_surround_suppresses_response(self, image_paradigm):
        """DoG with strong surround should have smaller max response than plain Gaussian."""
        from braincoder.models import GaussianPRF2D, DifferenceOfGaussiansPRF2D
        xy_pars = pd.DataFrame({'x': [0.], 'y': [0.], 'sd': [0.25],
                                'baseline': [0.], 'amplitude': [1.]}, dtype=np.float32)
        dog_pars = pd.DataFrame({'x': [0.], 'y': [0.], 'sd': [0.25], 'baseline': [0.],
                                 'amplitude': [1.], 'srf_amplitude': [0.5], 'srf_size': [2.]},
                                dtype=np.float32)
        pred_gauss = GaussianPRF2D(paradigm=image_paradigm, parameters=xy_pars).predict()
        pred_dog = DifferenceOfGaussiansPRF2D(paradigm=image_paradigm, parameters=dog_pars).predict()
        # DoG peak should be lower in absolute value than Gaussian peak
        assert pred_dog.values.max() < pred_gauss.values.max()


# ---------------------------------------------------------------------------
# simulate() edge-case behaviour
# ---------------------------------------------------------------------------

class TestSimulateBehaviour:

    @pytest.fixture
    def simple_model(self, gauss_paradigm):
        from braincoder.models import GaussianPRF
        pars = pd.DataFrame({'mu': [-2., 0., 2.], 'sd': [1., 1., 1.],
                             'amplitude': [1., 1., 1.], 'baseline': [0., 0., 0.]}, dtype=np.float32)
        return GaussianPRF(paradigm=gauss_paradigm, parameters=pars)

    def test_n_repeats_shape(self, simple_model, gauss_paradigm):
        sim = simple_model.simulate(noise=0.1, n_repeats=4)
        assert sim.shape == (4 * len(gauss_paradigm), 3)

    def test_n_repeats_multiindex(self, simple_model):
        sim = simple_model.simulate(noise=0.1, n_repeats=3)
        assert 'repeat' in sim.index.names

    def test_n_repeats_1_same_as_default(self, simple_model):
        sim_default = simple_model.simulate(noise=0.0)
        sim_one = simple_model.simulate(noise=0.0, n_repeats=1)
        np.testing.assert_allclose(sim_default.values, sim_one.values, atol=1e-6)

    def test_gaussian_noise_scalar(self, simple_model):
        """With scalar noise, repeated calls should give different results."""
        s1 = simple_model.simulate(noise=1.0)
        s2 = simple_model.simulate(noise=1.0)
        assert not np.allclose(s1.values, s2.values)

    def test_covariance_noise_matrix(self, simple_model):
        """Matrix noise (covariance) should produce correlated residuals."""
        from braincoder.models import GaussianPRF
        pred = simple_model.predict()
        # Strong correlation between voxels 0 and 1 via off-diagonal covariance
        omega = np.array([[1., 0.9, 0.], [0.9, 1., 0.], [0., 0., 1.]], dtype=np.float32)
        sim = simple_model.simulate(noise=omega)
        resid = sim.values - pred.values
        r01, _ = pearsonr(resid[:, 0], resid[:, 1])
        r02, _ = pearsonr(resid[:, 0], resid[:, 2])
        assert r01 > 0.5, f"Expected high correlation between voxels 0,1, got {r01:.2f}"
        assert abs(r02) < 0.5, f"Expected low correlation between voxels 0,2, got {r02:.2f}"

    def test_student_t_noise(self, simple_model):
        """Student-t noise (dof < inf) should still produce finite predictions."""
        sim = simple_model.simulate(noise=0.5, dof=3.0)
        assert sim.shape == simple_model.predict().shape
        assert np.all(np.isfinite(sim.values))


# ---------------------------------------------------------------------------
# ResidualFitter
# ---------------------------------------------------------------------------

class TestResidualFitter:

    @pytest.fixture
    def fitted_model_and_data(self, gauss_paradigm):
        from braincoder.models import GaussianPRF
        n_voxels = 6
        pars = pd.DataFrame({
            'mu':        np.linspace(-3, 3, n_voxels, dtype=np.float32),
            'sd':        np.ones(n_voxels, dtype=np.float32),
            'amplitude': np.ones(n_voxels, dtype=np.float32),
            'baseline':  np.zeros(n_voxels, dtype=np.float32),
        })
        model = GaussianPRF(paradigm=gauss_paradigm, parameters=pars)
        model.init_pseudoWWT(gauss_paradigm, pars)
        data = model.simulate(noise=0.3)
        return model, data

    def test_omega_shape(self, fitted_model_and_data, gauss_paradigm):
        from braincoder.optimize import ResidualFitter
        model, data = fitted_model_and_data
        n_voxels = data.shape[1]
        fitter = ResidualFitter(model, data, paradigm=gauss_paradigm)
        omega, dof = fitter.fit(max_n_iterations=50, min_n_iterations=20,
                                progressbar=False)
        assert omega.shape == (n_voxels, n_voxels)
        assert dof is None  # Gaussian method returns None for dof

    def test_omega_is_positive_definite(self, fitted_model_and_data, gauss_paradigm):
        from braincoder.optimize import ResidualFitter
        model, data = fitted_model_and_data
        fitter = ResidualFitter(model, data, paradigm=gauss_paradigm)
        omega, _ = fitter.fit(max_n_iterations=50, min_n_iterations=20,
                              progressbar=False)
        # All eigenvalues should be positive for a valid covariance matrix
        eigvals = np.linalg.eigvalsh(omega)
        assert np.all(eigvals > 0), f"omega has non-positive eigenvalues: {eigvals}"

    def test_omega_is_symmetric(self, fitted_model_and_data, gauss_paradigm):
        from braincoder.optimize import ResidualFitter
        model, data = fitted_model_and_data
        fitter = ResidualFitter(model, data, paradigm=gauss_paradigm)
        omega, _ = fitter.fit(max_n_iterations=50, min_n_iterations=20,
                              progressbar=False)
        np.testing.assert_allclose(omega, omega.T, atol=1e-5)

    def test_spherical_omega_is_diagonal(self, fitted_model_and_data, gauss_paradigm):
        from braincoder.optimize import ResidualFitter
        model, data = fitted_model_and_data
        n_voxels = data.shape[1]
        fitter = ResidualFitter(model, data, paradigm=gauss_paradigm)
        omega, _ = fitter.fit(max_n_iterations=50, min_n_iterations=20,
                              progressbar=False, spherical=True)
        # Off-diagonal elements should be ~0
        off_diag = omega - np.diag(np.diag(omega))
        np.testing.assert_allclose(off_diag, 0., atol=1e-5)

    def test_fitted_parameters_stored(self, fitted_model_and_data, gauss_paradigm):
        from braincoder.optimize import ResidualFitter
        model, data = fitted_model_and_data
        fitter = ResidualFitter(model, data, paradigm=gauss_paradigm)
        fitter.fit(max_n_iterations=50, min_n_iterations=20, progressbar=False)
        assert hasattr(fitter, 'fitted_omega_parameters')
        assert 'tau' in fitter.fitted_omega_parameters
        assert 'rho' in fitter.fitted_omega_parameters


# ---------------------------------------------------------------------------
# ParameterFitter — shared/fixed parameters
# ---------------------------------------------------------------------------

class TestParameterFitterAdvanced:

    @pytest.fixture
    def paradigm_and_data(self, gauss_paradigm):
        from braincoder.models import GaussianPRF
        n_voxels = 10
        pars = pd.DataFrame({
            'mu':        np.linspace(-3, 3, n_voxels, dtype=np.float32),
            'sd':        np.ones(n_voxels, dtype=np.float32),
            'amplitude': np.ones(n_voxels, dtype=np.float32),
            'baseline':  np.zeros(n_voxels, dtype=np.float32),
        })
        model = GaussianPRF(paradigm=gauss_paradigm, parameters=pars)
        data = model.simulate(noise=0.1)
        return model, data, pars, gauss_paradigm

    def test_fixed_pars(self, paradigm_and_data):
        """Fitting with fixed sd should not change sd from init_pars."""
        from braincoder.optimize import ParameterFitter
        model, data, true_pars, paradigm = paradigm_and_data
        fitter = ParameterFitter(model, data, paradigm)
        fixed_sd = 1.0
        init_pars = pd.DataFrame({
            'mu':        np.zeros(10, dtype=np.float32),
            'sd':        np.full(10, fixed_sd, dtype=np.float32),
            'amplitude': np.ones(10, dtype=np.float32),
            'baseline':  np.zeros(10, dtype=np.float32),
        })
        result = fitter.fit(init_pars=init_pars, fixed_pars=['sd'],
                            max_n_iterations=30, progressbar=False)
        np.testing.assert_allclose(result['sd'].values,
                                   np.full(10, fixed_sd), rtol=1e-4)

    def test_invalid_fixed_par_raises(self, paradigm_and_data):
        from braincoder.optimize import ParameterFitter
        model, data, true_pars, paradigm = paradigm_and_data
        fitter = ParameterFitter(model, data, paradigm)
        init_pars = pd.DataFrame({'mu': np.zeros(10), 'sd': np.ones(10),
                                  'amplitude': np.ones(10), 'baseline': np.zeros(10)}, dtype=np.float32)
        with pytest.raises(ValueError):
            fitter.fit(init_pars=init_pars, fixed_pars=['nonexistent_param'],
                       max_n_iterations=5, progressbar=False)

    def test_get_predictions_after_fit(self, paradigm_and_data):
        """get_predictions() should return a DataFrame of the right shape."""
        from braincoder.optimize import ParameterFitter
        model, data, true_pars, paradigm = paradigm_and_data
        fitter = ParameterFitter(model, data, paradigm)
        init_pars = pd.DataFrame({'mu': np.zeros(10), 'sd': np.ones(10),
                                  'amplitude': np.ones(10), 'baseline': np.zeros(10)}, dtype=np.float32)
        fitter.fit(init_pars=init_pars, max_n_iterations=10, progressbar=False)
        preds = fitter.get_predictions()
        assert preds.shape == data.shape

    def test_get_rsq_after_fit(self, paradigm_and_data):
        """R² should be in a reasonable range after a brief fit."""
        from braincoder.optimize import ParameterFitter
        model, data, true_pars, paradigm = paradigm_and_data
        fitter = ParameterFitter(model, data, paradigm)
        init_pars = true_pars.copy()
        fitter.fit(init_pars=init_pars, max_n_iterations=10, progressbar=False)
        r2 = fitter.get_rsq()
        # Starting from true params, R² should be high even with few iterations
        assert r2.mean() > 0.5
