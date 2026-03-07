import tensorflow as tf
import tensorflow_probability as tfp
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
from ..utils import norm, format_data, format_paradigm, format_parameters, format_weights, logit, restrict_radians, lognormalpdf_n, von_mises_pdf, lognormal_pdf_mode_fwhm, norm2d
from tensorflow_probability import distributions as tfd
from ..utils.math import aggressive_softplus, aggressive_softplus_inverse, norm
import scipy.stats as ss
from ..stimuli import Stimulus, OneDimensionalRadialStimulus, OneDimensionalGaussianStimulus, OneDimensionalStimulusWithAmplitude, OneDimensionalRadialStimulusWithAmplitude, ImageStimulus, TwoDimensionalStimulus
from patsy import dmatrix, build_design_matrices
from .base import EncodingModel, EncodingRegressionModel, HRFEncodingModel

class GaussianPRF(EncodingModel):
    """One-dimensional population receptive field with Gaussian tuning."""

    parameter_labels = ['mu', 'sd', 'amplitude', 'baseline']

    def __init__(self, paradigm=None, data=None, parameters=None,
                 weights=None, omega=None, allow_neg_amplitudes=False, verbosity=logging.INFO,
                 model_stimulus_amplitude=False,
                 **kwargs):
        """Configure Gaussian pRF with optional stimulus amplitude modeling."""

        if not hasattr(self, 'transformations'):
            if allow_neg_amplitudes:
                self.transformations = ['identity', 'aggressive_softplus', 'identity', 'identity']
            else:
                self.transformations = ['identity', 'aggressive_softplus', 'aggressive_softplus', 'identity']

        self.stimulus_type = self._get_stimulus_type(model_stimulus_amplitude=model_stimulus_amplitude)
        self._basis_predictions = self._get_basis_predictions(model_stimulus_amplitude=model_stimulus_amplitude)

        super().__init__(paradigm=paradigm, data=data, parameters=parameters,
                         weights=weights, omega=omega, verbosity=logging.INFO, **kwargs)


    def _get_stimulus_type(self, model_stimulus_amplitude=False):
        """Select stimulus class depending on whether amplitude is modeled."""
        if model_stimulus_amplitude:
            return OneDimensionalStimulusWithAmplitude
        else:
            return Stimulus

    def _get_basis_predictions(self, model_stimulus_amplitude=False):
        """Return the appropriate basis prediction function handle."""
        if model_stimulus_amplitude:
            return self._basis_predictions_with_amplitude
        else:
            return self._basis_predictions_without_amplitude

    def basis_predictions(self, paradigm=None, parameters=None):
        """Convenience wrapper returning numpy array of basis predictions."""

        paradigm = self.get_paradigm(paradigm)
        parameters = self._get_parameters(parameters)

        paradigm_ = self._get_paradigm(paradigm)[np.newaxis, ...]
        parameters_ = parameters.values[np.newaxis, ...] if parameters is not None else None

        # predictions = self._predict(paradigm_, parameters_, weights_)[0]
        # if hasattr(parameters, 'values'):
        #     parameters = parameters.values

        # parameters = np.float32(parameters)

        return self._basis_predictions(paradigm_, parameters_)[0]

    def get_init_pars(self, data, paradigm, confounds=None):
        """Heuristic initialization for mu/sd/amplitude/baseline."""

        paradigm = self._get_paradigm(paradigm)
        data = format_data(data)

        if confounds is not None:
            beta = tf.linalg.lstsq(confounds, data)
            predictions = (confounds @ beta)
            data -= predictions

        if hasattr(data, 'values'):
            data = data.values

        baselines = tf.reduce_min(data, 0)
        data_ = (data - baselines)

        mus = tf.reduce_sum((data_ * self.stimulus._generate_stimulus(paradigm.values)), 0) / tf.reduce_sum(data_, 0)
        sds = tf.sqrt(tf.reduce_sum(data_ * (self.stimulus._generate_stimulus(paradigm.values) - mus)
                                    ** 2, 0) / tf.reduce_sum(data_, 0))
        amplitudes = tf.reduce_max(data_, 0)

        parameters = tf.concat([mus[:, tf.newaxis],
                                sds[:, tf.newaxis],
                                amplitudes[:, tf.newaxis],
                                baselines[:, tf.newaxis]], 1)

        return parameters

    @tf.function
    def _basis_predictions_without_amplitude(self, paradigm, parameters):
        """Gaussian tuning without stimulus amplitude modulation."""
        # paradigm: n_batches x n_timepoints x n_stimulus_features
        # parameters:: n_batches x n_voxels x n_parameters

        # norm: n_batches x n_timepoints x n_voxels

        # output: n_batches x n_timepoints x n_voxels
        return norm(paradigm[..., tf.newaxis, 0],
                    parameters[:, tf.newaxis, :, 0],
                    parameters[:, tf.newaxis, :, 1]) * \
            parameters[:, tf.newaxis, :, 2] + parameters[:, tf.newaxis, :, 3]

    @tf.function
    def _basis_predictions_with_amplitude(self, paradigm, parameters):
        """Gaussian tuning optionally scaled by stimulus amplitude channel."""
        # paradigm: n_batches x n_timepoints x n_stimulus_features
        # parameters:: n_batches x n_voxels x n_parameters

        # norm: n_batches x n_timepoints x n_voxels

        # output: n_batches x n_timepoints x n_voxels
        return norm(paradigm[..., tf.newaxis, 0],
                    parameters[:, tf.newaxis, :, 0],
                    parameters[:, tf.newaxis, :, 1]) * \
            parameters[:, tf.newaxis, :, 2] * paradigm[:, :, tf.newaxis, 1] + parameters[:, tf.newaxis, :, 3]


    def init_pseudoWWT(self, stimulus_range, parameters):
        """Cache WᵀW approximation by integrating basis responses over range."""

        stimulus_range = stimulus_range.astype(np.float32)
        W = self.basis_predictions(stimulus_range, parameters)

        pseudoWWT = tf.tensordot(W, W, (0, 0))
        self._pseudoWWT = tf.where(tf.math.is_nan(pseudoWWT), tf.zeros_like(pseudoWWT),
                                   pseudoWWT)
        return self._pseudoWWT

    def get_pseudoWWT(self):
        """Return cached pseudo WᵀW matrix or compute via weights."""

        if self.weights is not None:
            return self.weights.T.dot(self.weights).values

        if hasattr(self, '_pseudoWWT'):
            return self._pseudoWWT
        else:
            raise ValueError(
                'First initialize WWT for a specific stimulus range using init_pseudoWWT!')

    def get_WWT(self):
        """Alias for :meth:`get_pseudoWWT`."""
        return self.get_pseudoWWT()

class RegressionGaussianPRF(EncodingRegressionModel, GaussianPRF):
    """Gaussian pRF whose parameters are modeled by regression covariates."""


class VonMisesPRF(GaussianPRF):
    """Circular pRF with von Mises tuning (e.g., for polar angle stimuli)."""

    parameter_labels = ['mu', 'kappa', 'amplitude', 'baseline']
    stimulus_type = OneDimensionalRadialStimulus

    def __init__(self, paradigm=None, data=None, parameters=None,
                 weights=None, omega=None, allow_neg_amplitudes=False,
                 model_stimulus_amplitude=False,
                 **kwargs):

        super().__init__(paradigm=paradigm, data=data, parameters=parameters,
                            weights=weights, omega=omega, allow_neg_amplitudes=allow_neg_amplitudes,
                            model_stimulus_amplitude=model_stimulus_amplitude,
                            **kwargs)

    def _get_stimulus_type(self, model_stimulus_amplitude=False):
        """Select radial stimulus variant (with optional amplitude channel)."""
        if model_stimulus_amplitude:
            return OneDimensionalRadialStimulusWithAmplitude
        else:
            return OneDimensionalRadialStimulus

    def _get_stimulus(self, **kwargs):
        """Instantiate the radial stimulus (ignores n_dimensions — always 1D)."""
        return self.stimulus_type()

    @tf.function
    def _basis_predictions_without_amplitude(self, paradigm, parameters):
        """Von Mises tuning without stimulus amplitude modulation."""
        # paradigm: n_batches x n_timepoints x n_stimulus_features
        # parameters:: n_batches x n_voxels x n_parameters

        # norm: n_batches x n_timepoints x n_voxels

        # output: n_batches x n_timepoints x n_voxels
        return von_mises_pdf(paradigm[..., tf.newaxis, 0],
                    parameters[:, tf.newaxis, :, 0],
                    parameters[:, tf.newaxis, :, 1]) * \
            parameters[:, tf.newaxis, :, 2] + parameters[:, tf.newaxis, :, 3]

    @tf.function
    def _basis_predictions_with_amplitude(self, paradigm, parameters):
        """Von Mises tuning scaled by a stimulus amplitude channel."""
        # paradigm: n_batches x n_timepoints x n_stimulus_features
        # parameters:: n_batches x n_voxels x n_parameters

        # norm: n_batches x n_timepoints x n_voxels

        # output: n_batches x n_timepoints x n_voxels
        return von_mises_pdf(paradigm[..., tf.newaxis, 0],
                    parameters[:, tf.newaxis, :, 0],
                    parameters[:, tf.newaxis, :, 1]) * \
            parameters[:, tf.newaxis, :, 2] * paradigm[..., tf.newaxis, 1] + parameters[:, tf.newaxis, :, 3]

    def init_pseudoWWT(self, stimulus_range, parameters):
        """Precompute pseudo WᵀW for circular stimulus space."""

        if stimulus_range.ndim == 2:
            stimulus_range = stimulus_range[:, [0]]

        stimulus_range = np.stack((stimulus_range, np.ones_like(stimulus_range)), axis=1).astype(np.float32)
        W = self.basis_predictions(stimulus_range, parameters)

        pseudoWWT = tf.tensordot(W, W, (0, 0))
        self._pseudoWWT = tf.where(tf.math.is_nan(pseudoWWT), tf.zeros_like(pseudoWWT),
                                   pseudoWWT)
        return self._pseudoWWT

class LogGaussianPRF(GaussianPRF):
    """Log-Gaussian tuning curve with configurable parameterization."""

    parameter_labels = ['mu', 'sd', 'amplitude', 'baseline']

    def __init__(self, paradigm=None, data=None, parameters=None,
                 weights=None, omega=None, allow_neg_amplitudes=False, verbosity=logging.INFO,
                 model_stimulus_amplitude=False,
                 parameterisation='mu_sd_natural',
                 **kwargs):
        """Configure whether the model uses (mu, sd) or (mode, FWHM) parameters."""

        if parameterisation == 'mu_sd_natural':
            self.parameter_labels = ['mu', 'sd', 'amplitude', 'baseline']
            self._basis_predictions_without_amplitude = self._basis_predictions_without_amplitude_n
            self._basis_predictions_with_amplitude = self._basis_predictions_with_amplitude_n
        elif parameterisation == 'mode_fwhm_natural':
            self.parameter_labels = ['mode', 'fwhm', 'amplitude', 'baseline']
            self._basis_predictions_without_amplitude = self._basis_predictions_without_amplitude_mode_fwhm
            self._basis_predictions_with_amplitude = self._basis_predictions_with_amplitude_mode_fwhm
        else:
            raise ValueError('Unknown parameterisation! Needs to be in [mu_sd_natural, mode_fwhm_natural]')

        if allow_neg_amplitudes:
            self.transformations = ['softplus', 'softplus', 'identity', 'identity']
        else:
            self.transformations = ['softplus', 'softplus', 'softplus', 'identity']

        super().__init__(paradigm=paradigm, data=data, parameters=parameters,
                         weights=weights, omega=omega, allow_neg_amplitudes=allow_neg_amplitudes,
                          verbosity=verbosity, model_stimulus_amplitude=model_stimulus_amplitude,
                          **kwargs)

    @tf.function
    def _basis_predictions_without_amplitude_n(self, paradigm, parameters):
        """Log-normal tuning (mu/sd) without external amplitude modulation."""
        return lognormalpdf_n(paradigm[..., tf.newaxis, 0],
                    parameters[:, tf.newaxis, :, 0],
                    parameters[:, tf.newaxis, :, 1]) * \
            parameters[:, tf.newaxis, :, 2] + parameters[:, tf.newaxis, :, 3]

    @tf.function
    def _basis_predictions_with_amplitude_n(self, paradigm, parameters):
        """Log-normal tuning (mu/sd) scaled by stimulus amplitude."""
        return lognormalpdf_n(paradigm[..., tf.newaxis, 0],
                    parameters[:, tf.newaxis, :, 0],
                    parameters[:, tf.newaxis, :, 1]) * \
            parameters[:, tf.newaxis, :, 2] * paradigm[..., tf.newaxis, 1] + parameters[:, tf.newaxis, :, 3]

    @tf.function
    def _basis_predictions_without_amplitude_mode_fwhm(self, paradigm, parameters):
        """Mode/FWHM parameterization without amplitude scaling."""
        return lognormal_pdf_mode_fwhm(paradigm[..., tf.newaxis, 0],
                    parameters[:, tf.newaxis, :, 0],
                    parameters[:, tf.newaxis, :, 1]) * \
            parameters[:, tf.newaxis, :, 2] + parameters[:, tf.newaxis, :, 3]

    @tf.function
    def _basis_predictions_with_amplitude_mode_fwhm(self, paradigm, parameters):
        """Mode/FWHM parameterization scaled by stimulus amplitude."""
        return lognormal_pdf_mode_fwhm(paradigm[..., tf.newaxis, 0],
                    parameters[:, tf.newaxis, :, 0],
                    parameters[:, tf.newaxis, :, 1]) * \
            parameters[:, tf.newaxis, :, 2] * paradigm[..., tf.newaxis, 1] + parameters[:, tf.newaxis, :, 3]

class GaussianPRFWithHRF(HRFEncodingModel, GaussianPRF):
    """Combine Gaussian pRF spatial tuning with an explicit HRF convolution."""

    def __init__(self, paradigm=None, data=None, parameters=None,
                 weights=None, omega=None, hrf_model=None,
                 flexible_hrf_parameters=False, allow_neg_amplitudes=False,
                 model_stimulus_amplitude=False, verbosity=logging.INFO, **kwargs):
        GaussianPRF.__init__(self, paradigm=paradigm, data=data, parameters=parameters,
                             weights=weights, omega=omega,
                             allow_neg_amplitudes=allow_neg_amplitudes,
                             model_stimulus_amplitude=model_stimulus_amplitude,
                             verbosity=verbosity, **kwargs)
        HRFEncodingModel.__init__(self, hrf_model=hrf_model,
                                  flexible_hrf_parameters=flexible_hrf_parameters, **kwargs)


class LogGaussianPRFWithHRF(HRFEncodingModel, LogGaussianPRF):
    """Log-Gaussian tuning plus HRF parameters."""

    def __init__(self, paradigm=None, data=None, parameters=None,
                 weights=None, omega=None, hrf_model=None,
                 flexible_hrf_parameters=False, allow_neg_amplitudes=False,
                 model_stimulus_amplitude=False, parameterisation='mu_sd_natural',
                 verbosity=logging.INFO, **kwargs):
        LogGaussianPRF.__init__(self, paradigm=paradigm, data=data, parameters=parameters,
                                weights=weights, omega=omega,
                                allow_neg_amplitudes=allow_neg_amplitudes,
                                model_stimulus_amplitude=model_stimulus_amplitude,
                                parameterisation=parameterisation,
                                verbosity=verbosity, **kwargs)
        HRFEncodingModel.__init__(self, hrf_model=hrf_model,
                                  flexible_hrf_parameters=flexible_hrf_parameters, **kwargs)


class AlphaGaussianPRF(GaussianPRF):
    """Gaussian pRF with additional alpha parameter controlling asymmetry."""

    parameter_labels = ['mu', 'sd', 'alpha', 'amplitude', 'baseline']

    def __init__(self, paradigm=None, data=None, parameters=None,
                 weights=None, omega=None, allow_neg_amplitudes=False, verbosity=logging.INFO,
                 model_stimulus_amplitude=False,
                 **kwargs):
        """Initialize alpha-Gaussian model (no stimulus amplitude option)."""

        if model_stimulus_amplitude:
            raise NotImplementedError("Modeling stimulus amplitude is not implemented for AlphaGaussianPRF")

        # mu and sd must be positive; alpha is free; amplitude constraint depends on allow_neg_amplitudes
        if allow_neg_amplitudes:
            self.transformations = ['softplus', 'softplus', 'identity', 'identity', 'identity']
        else:
            self.transformations = ['softplus', 'softplus', 'identity', 'softplus', 'identity']

        super().__init__(paradigm=paradigm, data=data, parameters=parameters,
                         weights=weights, omega=omega, allow_neg_amplitudes=allow_neg_amplitudes,
                          verbosity=verbosity, model_stimulus_amplitude=model_stimulus_amplitude,
                          **kwargs)

    @tf.function
    def _basis_predictions_without_amplitude(self, paradigm, parameters):
        def alpha_transform(x, alpha, eps=1e-6):
            """ Computes a numerically stable alpha transformation. """
            return tf.where(
                tf.abs(alpha) < eps,
                tf.math.log(x),  # Directly use log(x) when alpha ≈ 0
                (tf.pow(x, alpha) - 1) / alpha
            )

        def f_x(x, mu_x, sigma_mu, alpha):
            """ Computes p_x(x | mu_x, sigma_x) using the given formula. """
            mu_alpha_x = alpha_transform(x, alpha)  # Using your transformation
            mu_alpha_mu = alpha_transform(mu_x, alpha)  # Using your transformation
            exponent = -tf.square(mu_alpha_x - mu_alpha_mu) / (2 * tf.square(sigma_mu))
            return tf.exp(exponent)

        return f_x(paradigm[..., tf.newaxis, 0],
                    parameters[:, tf.newaxis, :, 0],
                    parameters[:, tf.newaxis, :, 1],
                    parameters[:, tf.newaxis, :, 2]) * \
            parameters[:, tf.newaxis, :, 3] + parameters[:, tf.newaxis, :, 4]

class RegressionAlphaGaussianPRF(EncodingRegressionModel, AlphaGaussianPRF):
    """Alpha-Gaussian pRF variant whose parameters depend on regressors."""


class GaussianPRFOnGaussianSignal(GaussianPRF):
    """pRF evaluated on Gaussian stimulus summaries (mean + SD)."""

    stimulus_type = OneDimensionalGaussianStimulus

    def __init__(self, paradigm=None, data=None, parameters=None,
                 weights=None, omega=None, allow_neg_amplitudes=False,
                  stimulus_grid=None, verbosity=logging.INFO,
                 **kwargs):
        """Set up Gaussian stimuli characterized by their mean/SD distributions."""

        if stimulus_grid is None:
            raise Exception('Need stimulus_grid!')

        if paradigm is not None:
            assert('mu' in paradigm.columns), 'Need mean of Gaussian in paradigm'
            assert('sd' in paradigm.columns), 'Need sd of Gaussian in paradigm'

            paradigm = paradigm[['mu', 'sd']]

        self.stimulus_grid = stimulus_grid.astype(np.float32)

        super().__init__(paradigm=paradigm, data=data, parameters=parameters,
                         weights=weights, omega=omega, allow_neg_amplitudes=allow_neg_amplitudes,
                         verbosity=logging.INFO, **kwargs)

    @tf.function
    def _basis_predictions(self, paradigm, parameters):
        """Convolve receptive-field Gaussians with Gaussian stimulus inputs."""
        # n_stim-grid x n-batches x n-timepoints x n_voxels
        rf_field = norm(self.stimulus_grid[:, tf.newaxis, tf.newaxis, tf.newaxis],  #grid to evaluate on
                        parameters[tf.newaxis, :, tf.newaxis, :, 0],
                        parameters[tf.newaxis, :, tf.newaxis, :, 1])

        input_stimulus = norm(self.stimulus_grid[:, tf.newaxis, tf.newaxis, tf.newaxis],  #grid to evaluate on
                        paradigm[tf.newaxis, ..., 0, tf.newaxis],
                        paradigm[tf.newaxis, ..., 1, tf.newaxis])

        return tf.reduce_sum(rf_field * input_stimulus, axis=0)
