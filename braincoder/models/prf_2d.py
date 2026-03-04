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
from .base import EncodingModel, HRFEncodingModel

class GaussianPointPRF2D(EncodingModel):

    def __init__(self, paradigm=None, data=None, parameters=None,
                 weights=None, omega=None, allow_neg_amplitudes=False, verbosity=logging.INFO,
                 model_stimulus_amplitude=False, correlated_response=False,
                 **kwargs):

        self.correlated_response = correlated_response

        if correlated_response:
            self.parameter_labels = ['mu_x', 'mu_y', 'sd_x', 'sd_y', 'rho', 'amplitude', 'baseline']
        else:
            self.parameter_labels = ['mu_x', 'mu_y', 'sd_x', 'sd_y', 'amplitude', 'baseline']

        if allow_neg_amplitudes:
            self._transform_parameters_forward = self._transform_parameters_forward1
            self._transform_parameters_backward = self._transform_parameters_backward1
        else:
            self._transform_parameters_forward = self._transform_parameters_forward2
            self._transform_parameters_backward = self._transform_parameters_backward2

        self.stimulus_type = self._get_stimulus_type(model_stimulus_amplitude=model_stimulus_amplitude)
        self._basis_predictions = self._get_basis_predictions(model_stimulus_amplitude=model_stimulus_amplitude)

        super().__init__(paradigm=paradigm, data=data, parameters=parameters,
                         weights=weights, omega=omega, verbosity=logging.INFO, **kwargs)


    def _get_stimulus_type(self, model_stimulus_amplitude=False):
            return TwoDimensionalStimulus

    def _get_basis_predictions(self, model_stimulus_amplitude=False):
        if model_stimulus_amplitude:
            return self._basis_predictions_with_amplitude
        else:
            return self._basis_predictions_without_amplitude

    def basis_predictions(self, paradigm=None, parameters=None):

        paradigm = self.get_paradigm(paradigm)
        parameters = self._get_parameters(parameters)

        if hasattr(parameters, 'values'):
            parameters = parameters.values

        parameters = np.float32(parameters)

        return self._basis_predictions(self.stimulus._generate_stimulus(paradigm.values), parameters[np.newaxis, ...])[0]

    def get_init_pars(self, data, paradigm, confounds=None):

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

        mus_x = tf.reduce_sum((data_ * self.stimulus._generate_stimulus(paradigm.values[..., 0])), 0) / tf.reduce_sum(data_, 0)
        mus_y = tf.reduce_sum((data_ * self.stimulus._generate_stimulus(paradigm.values[..., 1])), 0) / tf.reduce_sum(data_, 0)
        sds_x = tf.sqrt(tf.reduce_sum(data_ * (self.stimulus._generate_stimulus(paradigm.values[..., 0]) - mus_x)
                                    ** 2, 0) / tf.reduce_sum(data_, 0))
        sds_y = tf.sqrt(tf.reduce_sum(data_ * (self.stimulus._generate_stimulus(paradigm.values[..., 1]) - mus_y)
                                    ** 2, 0) / tf.reduce_sum(data_, 0))
        amplitudes = tf.reduce_max(data_, 0)

        # Optional covariance parameter initialization
        if self.correlated_response:
            covariances = tf.reduce_sum((data_ * (self.stimulus._generate_stimulus(paradigm.values[..., 0]) - mus_x) *
                                        (self.stimulus._generate_stimulus(paradigm.values[..., 1]) - mus_y)), 0) / tf.reduce_sum(data_, 0)
            parameters = tf.concat([mus_x[:, tf.newaxis],
                                    mus_y[:, tf.newaxis],
                                    sds_x[:, tf.newaxis],
                                    sds_y[:, tf.newaxis],
                                    covariances[:, tf.newaxis],
                                    amplitudes[:, tf.newaxis],
                                    baselines[:, tf.newaxis]], 1)
        else:
            parameters = tf.concat([mus_x[:, tf.newaxis],
                                    mus_y[:, tf.newaxis],
                                    sds_x[:, tf.newaxis],
                                    sds_y[:, tf.newaxis],
                                    amplitudes[:, tf.newaxis],
                                    baselines[:, tf.newaxis]], 1)

        return parameters

    @tf.function
    def _basis_predictions_without_amplitude(self, paradigm, parameters):
        # paradigm: n_batches x n_timepoints x n_stimulus_features (e.g., [x, y])
        # parameters: n_batches x n_voxels x n_parameters

        if self.correlated_response:
            return norm2d(paradigm[..., tf.newaxis, 0],
                          paradigm[..., tf.newaxis, 1],
                          parameters[:, tf.newaxis, :, 0],
                          parameters[:, tf.newaxis, :, 1],
                          parameters[:, tf.newaxis, :, 2],
                          parameters[:, tf.newaxis, :, 3],
                          parameters[:, tf.newaxis, :, 4]) * \
                   parameters[:, tf.newaxis, :, 5] + parameters[:, tf.newaxis, :, 6]
        else:
            return norm2d(paradigm[..., tf.newaxis, 0],
                          paradigm[..., tf.newaxis, 1],
                          parameters[:, tf.newaxis, :, 0],
                          parameters[:, tf.newaxis, :, 1],
                          parameters[:, tf.newaxis, :, 2],
                          parameters[:, tf.newaxis, :, 3]) * \
                   parameters[:, tf.newaxis, :, 4] + parameters[:, tf.newaxis, :, 5]

    @tf.function
    def _basis_predictions_with_amplitude(self, paradigm, parameters):
        # paradigm: n_batches x n_timepoints x n_stimulus_features (e.g., [x, y, amplitude])
        # parameters: n_batches x n_voxels x n_parameters

        if self.correlated_response:
            return norm2d(paradigm[..., tf.newaxis, 0],
                          paradigm[..., tf.newaxis, 1],
                          parameters[:, tf.newaxis, :, 0],
                          parameters[:, tf.newaxis, :, 1],
                          parameters[:, tf.newaxis, :, 2],
                          parameters[:, tf.newaxis, :, 3],
                          parameters[:, tf.newaxis, :, 4]) * \
                   parameters[:, tf.newaxis, :, 5] * paradigm[:, :, tf.newaxis, 2] + parameters[:, tf.newaxis, :, 6]
        else:
            return norm2d(paradigm[..., tf.newaxis, 0],
                          paradigm[..., tf.newaxis, 1],
                          parameters[:, tf.newaxis, :, 0],
                          parameters[:, tf.newaxis, :, 1],
                          parameters[:, tf.newaxis, :, 2],
                          parameters[:, tf.newaxis, :, 3]) * \
                   parameters[:, tf.newaxis, :, 4] * paradigm[:, :, tf.newaxis, 2] + parameters[:, tf.newaxis, :, 5]

    def init_pseudoWWT(self, stimulus_range, parameters):

        stimulus_range = stimulus_range.astype(np.float32)
        W = self.basis_predictions(stimulus_range, parameters)

        pseudoWWT = tf.tensordot(W, W, (0, 0))
        self._pseudoWWT = tf.where(tf.math.is_nan(pseudoWWT), tf.zeros_like(pseudoWWT),
                                   pseudoWWT)
        return self._pseudoWWT

    def get_pseudoWWT(self):

        if self.weights is not None:
            return self.weights.T.dot(self.weights).values

        if hasattr(self, '_pseudoWWT'):
            return self._pseudoWWT
        else:
            raise ValueError(
                'First initialize WWT for a specific stimulus range using init_pseudoWWT!')


    def _transform_parameters_forward1(self, parameters):

        if self.correlated_response:
            return tf.concat([parameters[:, 0][:, tf.newaxis],
                              parameters[:, 1][:, tf.newaxis],
                              tf.math.softplus(parameters[:, 2][:, tf.newaxis]),
                              tf.math.softplus(parameters[:, 3][:, tf.newaxis]),
                              tf.math.softplus(parameters[:, 4][:, tf.newaxis]) * 2 - 1,
                              parameters[:, 5][:, tf.newaxis],
                              parameters[:, 6][:, tf.newaxis]], axis=1)
        else:
            return tf.concat([parameters[:, 0][:, tf.newaxis],
                             parameters[:, 1][:, tf.newaxis],
                             tf.math.softplus(parameters[:, 2][:, tf.newaxis]),
                             tf.math.softplus(parameters[:, 3][:, tf.newaxis]),
                             parameters[:, 4][:, tf.newaxis],
                             parameters[:, 5][:, tf.newaxis]], axis=1)

    def _transform_parameters_backward1(self, parameters):

        if self.correlated_response:
            return tf.concat([parameters[:, 0][:, tf.newaxis],
                              parameters[:, 1][:, tf.newaxis],
                                tfp.math.softplus_inverse(parameters[:, 2][:, tf.newaxis]),
                                tfp.math.softplus_inverse(parameters[:, 3][:, tf.newaxis]),
                                tfp.math.softplus_inverse((parameters[:, 4][:, tf.newaxis] + 1) / 2.),
                                parameters[:, 5][:, tf.newaxis],
                                parameters[:, 6][:, tf.newaxis]], axis=1)

        else:
            return tf.concat([parameters[:, 0][:, tf.newaxis],
                             parameters[:, 1][:, tf.newaxis],
                             tfp.math.softplus_inverse(parameters[:, 2][:, tf.newaxis]),
                             tfp.math.softplus_inverse(parameters[:, 3][:, tf.newaxis]),
                             parameters[:, 4][:, tf.newaxis],
                             parameters[:, 5][:, tf.newaxis]], axis=1)


    def _transform_parameters_forward2(self, parameters):

        if self.correlated_response:
            return tf.concat([parameters[:, 0][:, tf.newaxis],
                              parameters[:, 1][:, tf.newaxis],
                              tf.math.softplus(parameters[:, 2][:, tf.newaxis]),
                              tf.math.softplus(parameters[:, 3][:, tf.newaxis]),
                              tf.math.softplus(parameters[:, 4][:, tf.newaxis]) * 2 - 1,
                              tf.math.softplus(parameters[:, 5][:, tf.newaxis]),
                              parameters[:, 6][:, tf.newaxis]], axis=1)
        else:
            return tf.concat([parameters[:, 0][:, tf.newaxis],
                             parameters[:, 1][:, tf.newaxis],
                             tf.math.softplus(parameters[:, 2][:, tf.newaxis]),
                             tf.math.softplus(parameters[:, 3][:, tf.newaxis]),
                             tf.math.softplus(parameters[:, 4][:, tf.newaxis]),
                             parameters[:, 5][:, tf.newaxis]], axis=1)

    def _transform_parameters_backward2(self, parameters):

        if self.correlated_response:
            return tf.concat([parameters[:, 0][:, tf.newaxis],
                              parameters[:, 1][:, tf.newaxis],
                                tfp.math.softplus_inverse(parameters[:, 2][:, tf.newaxis]),
                                tfp.math.softplus_inverse(parameters[:, 3][:, tf.newaxis]),
                                tfp.math.softplus_inverse((parameters[:, 4][:, tf.newaxis] + 1) / 2.),
                                tfp.math.softplus_inverse(parameters[:, 5][:, tf.newaxis]),
                                parameters[:, 6][:, tf.newaxis]], axis=1)

        else:
            return tf.concat([parameters[:, 0][:, tf.newaxis],
                             parameters[:, 1][:, tf.newaxis],
                             tfp.math.softplus_inverse(parameters[:, 2][:, tf.newaxis]),
                             tfp.math.softplus_inverse(parameters[:, 3][:, tf.newaxis]),
                             tfp.math.softplus_inverse(parameters[:, 4][:, tf.newaxis]),
                             parameters[:, 5][:, tf.newaxis]], axis=1)

class GaussianMixturePRF2D(EncodingModel):

    def __init__(self, paradigm=None, data=None, parameters=None,
                 weights=None, omega=None, allow_neg_amplitudes=False, verbosity=logging.INFO,
                 model_stimulus_amplitude=False,
                 same_rfs=False,
                 **kwargs):

        if allow_neg_amplitudes:
            self._transform_parameters_forward = self._transform_parameters_forward1
            self._transform_parameters_backward = self._transform_parameters_backward1
        else:
            self._transform_parameters_forward = self._transform_parameters_forward2
            self._transform_parameters_backward = self._transform_parameters_backward2

        self.stimulus_type = self._get_stimulus_type()

        self.same_rfs = same_rfs

        if same_rfs:
            self.parameter_labels = ['mu', 'sd', 'weight', 'amplitude', 'baseline']
        else:
            self.parameter_labels = ['mu_x', 'mu_y', 'sd_x', 'sd_y', 'weight', 'amplitude', 'baseline']

        super().__init__(paradigm=paradigm, data=data, parameters=parameters,
                         weights=weights, omega=omega, verbosity=logging.INFO, **kwargs)


    def _get_stimulus_type(self):
        return TwoDimensionalStimulus

    def _get_basis_predictions(self, model_stimulus_amplitude=False):
        if model_stimulus_amplitude:
            return self._basis_predictions_with_amplitude
        else:
            return self._basis_predictions_without_amplitude

    def basis_predictions(self, paradigm=None, parameters=None):

        paradigm = self.get_paradigm(paradigm)
        parameters = self._get_parameters(parameters)

        if hasattr(parameters, 'values'):
            parameters = parameters.values

        parameters = np.float32(parameters)

        return self._basis_predictions(self.stimulus._generate_stimulus(paradigm.values), parameters[np.newaxis, ...])[0]

    @tf.function
    def _basis_predictions(self, paradigm, parameters):
        # paradigm: n_batches x n_timepoints x n_stimulus_features
        # parameters:: n_batches x n_voxels x n_parameters

        # norm: n_batches x n_timepoints x n_voxels

        # output: n_batches x n_timepoints x n_voxels

        if self.same_rfs:
            return (parameters[:, tf.newaxis, :, 2] * norm(paradigm[..., tf.newaxis, 0],
                                                          parameters[:, tf.newaxis, :, 0],
                                                          parameters[:, tf.newaxis, :, 1]) + \
                    (1 - parameters[:, tf.newaxis, :, 2]) * norm(paradigm[..., tf.newaxis, 1],
                                                                 parameters[:, tf.newaxis, :, 0],
                                                          parameters[:, tf.newaxis, :, 1])) * \
                     parameters[:, tf.newaxis, :, 3] + parameters[:, tf.newaxis, :, 4]

        else:
            return (parameters[:, tf.newaxis, :, 4] * norm(paradigm[..., tf.newaxis, 0],
                                                    parameters[:, tf.newaxis, :, 0],
                                                    parameters[:, tf.newaxis, :, 2]) + \
                    (1 - parameters[:, tf.newaxis, :, 4]) * norm(paradigm[..., tf.newaxis, 1],
                                                            parameters[:, tf.newaxis, :, 1],
                                                            parameters[:, tf.newaxis, :, 3])) * \
                    parameters[:, tf.newaxis, :, 5] + parameters[:, tf.newaxis, :, 6]

    def init_pseudoWWT(self, stimulus_range, parameters):

        stimulus_range = stimulus_range.astype(np.float32)
        W = self.basis_predictions(stimulus_range, parameters)

        pseudoWWT = tf.tensordot(W, W, (0, 0))
        self._pseudoWWT = tf.where(tf.math.is_nan(pseudoWWT), tf.zeros_like(pseudoWWT),
                                   pseudoWWT)
        return self._pseudoWWT

    def get_pseudoWWT(self):

        if self.weights is not None:
            return self.weights.T.dot(self.weights).values

        if hasattr(self, '_pseudoWWT'):
            return self._pseudoWWT
        else:
            raise ValueError(
                'First initialize WWT for a specific stimulus range using init_pseudoWWT!')

    def get_WWT(self):
        return self.get_pseudoWWT()

    @tf.function
    def _transform_parameters_forward1(self, parameters):

        if self.same_rfs:
            return tf.concat([parameters[:, 0][:, tf.newaxis], # mu
                            tf.math.softplus(parameters[:, 1][:, tf.newaxis]), # sd
                            tf.math.sigmoid(parameters[:, 2][:, tf.newaxis]), # weight
                            tf.math.softplus(parameters[:, 3][:, tf.newaxis]), # amplitude
                            parameters[:, 4][:, tf.newaxis]], # baseline
                            axis=1)

        else:
            return tf.concat([parameters[:, 0][:, tf.newaxis],                      #mu_1
                            parameters[:, 1][:, tf.newaxis],
                            tf.math.softplus(parameters[:, 2][:, tf.newaxis]),    #sd_1
                            tf.math.softplus(parameters[:, 3][:, tf.newaxis]),    #sd_2
                            tf.math.sigmoid(parameters[:, 4][:, tf.newaxis]),
                            parameters[:, 5][:, tf.newaxis],
                            parameters[:, 6][:, tf.newaxis]], axis=1)

    @tf.function
    def _transform_parameters_backward1(self, parameters):
        if self.same_rfs:
            return tf.concat([parameters[:, 0][:, tf.newaxis], # mu
                            tfp.math.softplus_inverse(parameters[:, 1][:, tf.newaxis]), # sd
                            logit(parameters[:, 2][:, tf.newaxis]), # weight
                            tfp.math.softplus_inverse(parameters[:, 3][:, tf.newaxis]), # amplitude
                            parameters[:, 4][:, tf.newaxis]], # baseline
                            axis=1)
        else:
            return tf.concat([parameters[:, 0][:, tf.newaxis],
                            parameters[:, 1][:, tf.newaxis],
                            tfp.math.softplus_inverse(
                                parameters[:, 2][:, tf.newaxis]),
                            tfp.math.softplus_inverse(
                                parameters[:, 3][:, tf.newaxis]),
                            logit(parameters[:, 4][:, tf.newaxis]),
                            parameters[:, 5][:, tf.newaxis],
                            parameters[:, 6][:, tf.newaxis]], axis=1)


    @tf.function
    def _transform_parameters_forward2(self, parameters):
        if self.same_rfs:
            return tf.concat([parameters[:, 0][:, tf.newaxis],                      # mu
                              tf.math.softplus(parameters[:, 1][:, tf.newaxis]),    # sd
                              tf.math.sigmoid(parameters[:, 2][:, tf.newaxis]),     # weight
                              tf.math.softplus(parameters[:, 3][:, tf.newaxis]),    # amplitude
                              parameters[:, 4][:, tf.newaxis]], axis=1)            # baseline
        else:
            return tf.concat([parameters[:, 0][:, tf.newaxis],                      # mu_1
                              parameters[:, 1][:, tf.newaxis],                      # mu_2
                              tf.math.softplus(parameters[:, 2][:, tf.newaxis]),    # sd_1
                              tf.math.softplus(parameters[:, 3][:, tf.newaxis]),    # sd_2
                              tf.math.sigmoid(parameters[:, 4][:, tf.newaxis]),     # weight
                              tf.math.softplus(parameters[:, 5][:, tf.newaxis]),    # amplitude
                              parameters[:, 6][:, tf.newaxis]], axis=1)            # baseline

    @tf.function
    def _transform_parameters_backward2(self, parameters):
        if self.same_rfs:
            return tf.concat([parameters[:, 0][:, tf.newaxis],                      # mu
                              tfp.math.softplus_inverse(parameters[:, 1][:, tf.newaxis]),    # sd
                              logit(parameters[:, 2][:, tf.newaxis]),               # weight
                              tfp.math.softplus_inverse(parameters[:, 3][:, tf.newaxis]),    # amplitude
                              parameters[:, 4][:, tf.newaxis]], axis=1)            # baseline
        else:
            return tf.concat([parameters[:, 0][:, tf.newaxis],                      # mu_1
                              parameters[:, 1][:, tf.newaxis],                      # mu_2
                              tfp.math.softplus_inverse(parameters[:, 2][:, tf.newaxis]),    # sd_1
                              tfp.math.softplus_inverse(parameters[:, 3][:, tf.newaxis]),    # sd_2
                              logit(parameters[:, 4][:, tf.newaxis]),               # weight
                              tfp.math.softplus_inverse(parameters[:, 5][:, tf.newaxis]),    # amplitude
                              parameters[:, 6][:, tf.newaxis]], axis=1)            # baseline

class GaussianPRF2D(EncodingModel):

    parameter_labels = ['x', 'y', 'sd', 'baseline', 'amplitude']
    stimulus_type = ImageStimulus

    def __init__(self, grid_coordinates=None, paradigm=None, data=None, parameters=None,
                 weights=None, omega=None, positive_image_values_only=True, verbosity=logging.INFO, **kwargs):

        self.data = data
        self.parameters = format_parameters(parameters)
        self.weights = weights
        self.omega = omega

        if grid_coordinates is None:
            grid_coordinates = np.array(np.meshgrid(np.linspace(-1, 1, paradigm.shape[1]),
                                                    np.linspace(-1, 1, paradigm.shape[2]),), dtype=np.float32)

            grid_coordinates = np.swapaxes(grid_coordinates, 2, 1)
            grid_coordinates = np.reshape(
                grid_coordinates, (len(grid_coordinates), -1)).T

        self.grid_coordinates = pd.DataFrame(
            grid_coordinates, columns=['x', 'y'])
        self._grid_coordinates = self.grid_coordinates.values

        self.n_x = len(self.grid_coordinates['x'].unique())
        self.n_y = len(self.grid_coordinates['y'].unique())
        self.stimulus = self.stimulus_type(self.grid_coordinates, positive_only=positive_image_values_only)
        self.paradigm = self.stimulus.clean_paradigm(paradigm)

        x_diff = np.diff(np.sort(self.grid_coordinates['x'].unique())).mean()
        y_diff = np.diff(np.sort(self.grid_coordinates['y'].unique())).mean()
        self.pixel_area = x_diff * y_diff

        if omega is not None:
            self.omega_chol = np.linalg.cholesky(omega)


    def get_rf(self, as_frame=False, unpack=False, parameters=None):

        grid_coordinates = self.grid_coordinates.values

        parameters = self._get_parameters(parameters)
        parameters = self.parameters.values[np.newaxis, ...]

        rf = self._get_rf(grid_coordinates, parameters).numpy()[0]

        if as_frame:
            rf = pd.concat([pd.DataFrame(e,
                                         index=pd.MultiIndex.from_frame(self.grid_coordinates))
                            for e in rf],
                           keys=self.parameters.index)

            if unpack:
                rf = rf.unstack('x').sort_index(ascending=False)

        return rf

    @tf.function
    def _basis_predictions(self, paradigm, parameters):
        # paradigm: n_batches x n_timepoints x n_stimulus_features
        # parameters:: n_batches x n_voxels x n_parameters

        # norm: n_batches x n_timepoints x n_voxels

        # output: n_batches x n_timepoints x n_voxels

        rf = self._get_rf(self.grid_coordinates, parameters)
        baseline = parameters[:, tf.newaxis, :, 3]
        result = tf.tensordot(paradigm, rf, (2, 2))[:, :, 0, :] + baseline

        return result

    @tf.function
    def _get_rf(self, grid_coordinates, parameters, normalize=True):

        # n_batches x n_populations x  n_grid_spaces
        x = grid_coordinates[:, 0][tf.newaxis, tf.newaxis, :]
        y = grid_coordinates[:, 1][tf.newaxis, tf.newaxis, :]

        # n_batches x n_populations x n_grid_spaces (broadcast)
        mu_x = parameters[:, :, 0, tf.newaxis]
        mu_y = parameters[:, :, 1, tf.newaxis]
        sd = parameters[:, :, 2, tf.newaxis]
        amplitude = parameters[:, :, 4, tf.newaxis]

        gauss = (tf.exp(-((x-mu_x)**2 + (y-mu_y)**2)/(2*sd**2))) * amplitude

        if normalize:
            norm = sd * tf.sqrt(2 * np.pi) / self.pixel_area
            gauss = gauss / norm

        return gauss

    @tf.function
    def _transform_parameters_forward(self, parameters):
        return tf.concat([parameters[:, 0][:, tf.newaxis],
                          parameters[:, 1][:, tf.newaxis],
                          tf.math.softplus(parameters[:, 2][:, tf.newaxis]),
                          parameters[:, 3][:, tf.newaxis],
                          parameters[:, 4][:, tf.newaxis]], axis=1)

    @tf.function
    def _transform_parameters_backward(self, parameters):
        return tf.concat([parameters[:, 0][:, tf.newaxis],
                          parameters[:, 1][:, tf.newaxis],
                          tfp.math.softplus_inverse(
                              parameters[:, 2][:, tf.newaxis]),
                          parameters[:, 3][:, tf.newaxis],
                          parameters[:, 4][:, tf.newaxis]], axis=1)

    def get_pseudoWWT(self, remove_baseline=True):

        parameters = self._get_parameters().copy()

        if remove_baseline:
            parameters['baseline'] = np.float32(0.0)

        rf = self.get_rf(parameters=parameters.astype(np.float32))

        return rf.dot(rf.T)

    def to_linear_model(self):
        from .linear import LinearModelWithBaseline
        return LinearModelWithBaseline(self.paradigm, self.data, self.parameters[['baseline']], weights=self.get_rf().T)

    def unpack_stimulus(self, stimulus):
        return np.reshape(stimulus, (-1, self.n_x, self.n_y))

class GaussianPRF2DAngle(GaussianPRF2D):

    parameter_labels = ['theta', 'ecc', 'sd', 'baseline', 'amplitude']

    @tf.function
    def _get_rf(self, grid_coordinates, parameters):

        # n_batches x n_populations x  n_grid_spaces
        x = grid_coordinates[:, 0][tf.newaxis, tf.newaxis, :]
        y = grid_coordinates[:, 1][tf.newaxis, tf.newaxis, :]

        # n_batches x n_populations x n_grid_spaces (broadcast)
        theta = parameters[:, :, 0, tf.newaxis]
        ecc = parameters[:, :, 1, tf.newaxis]
        mu_x = tf.math.cos(theta) * ecc
        mu_y = tf.math.sin(theta) * ecc
        sd = parameters[:, :, 2, tf.newaxis]
        amplitude = parameters[:, :, 4, tf.newaxis]

        return (tf.exp(-((x-mu_x)**2 + (y-mu_y)**2)/(2*sd**2))) * amplitude

    @tf.function
    def _transform_parameters_forward(self, parameters):
        return tf.concat([parameters[:, 0][:, tf.newaxis],
                          tf.math.softplus(parameters[:, 1][:, tf.newaxis]),
                          tf.math.softplus(parameters[:, 2][:, tf.newaxis]),
                          parameters[:, 3][:, tf.newaxis],
                          parameters[:, 4][:, tf.newaxis]], axis=1)

    @tf.function
    def _transform_parameters_backward(self, parameters):
        return tf.concat([restrict_radians(parameters[:, 0][:, tf.newaxis]),
                          tfp.math.softplus_inverse(
                              parameters[:, 1][:, tf.newaxis]),
                          tfp.math.softplus_inverse(
                              parameters[:, 2][:, tf.newaxis]),
                          parameters[:, 3][:, tf.newaxis],
                          parameters[:, 4][:, tf.newaxis]], axis=1)

    def to_linear_model(self):
        from .linear import LinearModelWithBaseline
        return LinearModelWithBaseline(self.paradigm, self.data, self.parameters[['baseline']], weights=self.get_rf().T)

    def unpack_stimulus(self, stimulus):
        return np.reshape(stimulus, (-1, self.n_x, self.n_y))

    def to_xy_model(self):
        parameters = self.parameters.copy()
        parameters['x'] = np.cos(parameters['theta']) * parameters['ecc']
        parameters['y'] = np.sin(parameters['theta']) * parameters['ecc']
        parameters = parameters[['x', 'y', 'sd', 'baseline', 'amplitude']]

        return GaussianPRF2D(grid_coordinates=self.grid_coordinates,
                paradigm=self.paradigm, data=self.data, parameters=parameters,
                     weights=self.weights, omega=self.omega)


class GaussianPRF2DWithHRF(HRFEncodingModel, GaussianPRF2D):
    def __init__(self, grid_coordinates=None, paradigm=None, data=None, parameters=None,
                 positive_image_values_only=True,
                 weights=None, hrf_model=None, flexible_hrf_parameters=False, verbosity=logging.INFO, **kwargs):

        GaussianPRF2D.__init__(self, grid_coordinates=grid_coordinates, paradigm=paradigm, data=data,
                               parameters=parameters, weights=weights, verbosity=verbosity,
                               positive_image_values_only=positive_image_values_only, **kwargs)
        HRFEncodingModel.__init__(self, hrf_model=hrf_model, flexible_hrf_parameters=flexible_hrf_parameters, **kwargs)

    def to_linear_model(self):
        from .linear import LinearModelWithBaselineHRF
        return LinearModelWithBaselineHRF(self.paradigm, self.data,
                                          self.parameters[['baseline']], weights=self.get_rf().T,
                                          hrf_model=self.hrf_model)

    @tf.function
    def _transform_parameters_forward(self, parameters):
        if self.flexible_hrf_parameters:
            n_hrf_pars = len(self.hrf_model.parameter_labels)
            encoding_pars = GaussianPRF2D._transform_parameters_forward(self, parameters[:, :-n_hrf_pars])
            hrf_pars = self.hrf_model._transform_parameters_forward(parameters[:, -n_hrf_pars:])
            return tf.concat([encoding_pars, hrf_pars], axis=1)
        else:
            return GaussianPRF2D._transform_parameters_forward(self, parameters)

    @tf.function
    def _transform_parameters_backward(self, parameters):
        if self.flexible_hrf_parameters:
            n_hrf_pars = len(self.hrf_model.parameter_labels)
            encoding_pars = GaussianPRF2D._transform_parameters_backward(self, parameters[:, :-n_hrf_pars])
            hrf_pars = self.hrf_model._transform_parameters_backward(parameters[:, -n_hrf_pars:])
            return tf.concat([encoding_pars, hrf_pars], axis=1)
        else:
            return GaussianPRF2D._transform_parameters_backward(self, parameters)

class GaussianPRF2DAngleWithHRF(HRFEncodingModel, GaussianPRF2DAngle):

    def __init__(self, grid_coordinates=None, paradigm=None, data=None, parameters=None,
                 weights=None, hrf_model=None, verbosity=logging.INFO,
                  positive_image_values_only=True, flexible_hrf_parameters=False, **kwargs):

        GaussianPRF2DAngle.__init__(self, grid_coordinates=grid_coordinates, paradigm=paradigm, data=data, parameters=parameters, weights=weights, verbosity=verbosity,
                        positive_image_values_only=positive_image_values_only, **kwargs)

        HRFEncodingModel.__init__(self, hrf_model=hrf_model, flexible_hrf_parameters=flexible_hrf_parameters, **kwargs)
        self.hrf_model = hrf_model

    def to_linear_model(self):
        from .linear import LinearModelWithBaselineHRF
        return LinearModelWithBaselineHRF(self.paradigm, self.data,
                                          self.parameters[[
                                              'baseline']], weights=self.get_rf().T,
                                          hrf_model=self.hrf_model)

    def to_xy_model(self):

        no_hrf_model = super().to_xy_model()

        return GaussianPRF2DWithHRF(grid_coordinates=self.grid_coordinates,
                paradigm=self.paradigm, data=self.data, parameters=no_hrf_model.parameters,
                     weights=self.weights, omega=self.omega,
                     hrf_model=self.hrf_model)

class DifferenceOfGaussiansPRF2D(GaussianPRF2D):

    # Amplitude is as a fraction of the positive amplitude and is limited to be within [0, 1]
    # srf factor is limited to be above 1
    parameter_labels = ['x', 'y', 'sd', 'baseline',
                        'amplitude', 'srf_amplitude', 'srf_size']

    transformations = ['identity', 'identity', 'softplus', 'identity',
                       'softplus', 'softplus', 'softplus']
    @tf.function
    def _get_rf(self, grid_coordinates, parameters):

        # n_batches x n_populations x n_grid_spaces (broadcast)
        mu_x = parameters[:, :, 0, tf.newaxis]
        mu_y = parameters[:, :, 1, tf.newaxis]
        sd = parameters[:, :, 2, tf.newaxis]
        amplitude = parameters[:, :, 4, tf.newaxis]

        srf_amplitude = parameters[:, :, 5, tf.newaxis]
        srf_size = parameters[:, :, 6, tf.newaxis]

        standard_prf = super()._get_rf(grid_coordinates, parameters)

        srf_pars = tf.concat([mu_x, mu_y, sd*srf_size, tf.zeros_like(mu_x), srf_amplitude*amplitude*srf_size], axis=2)
        sprf = super()._get_rf(grid_coordinates, srf_pars)

        return standard_prf - sprf


class DifferenceOfGaussiansPRF2DWithHRF(HRFEncodingModel, DifferenceOfGaussiansPRF2D):

    def __init__(self, grid_coordinates=None, paradigm=None, data=None, parameters=None,
                 positive_image_values_only=True,
                 weights=None, hrf_model=None, flexible_hrf_parameters=False, verbosity=logging.INFO, **kwargs):

        DifferenceOfGaussiansPRF2D.__init__(self, grid_coordinates=grid_coordinates, paradigm=paradigm, data=data, parameters=parameters, weights=weights, verbosity=verbosity,
                        positive_image_values_only=positive_image_values_only, **kwargs)

        HRFEncodingModel.__init__(self, hrf_model=hrf_model, flexible_hrf_parameters=flexible_hrf_parameters, **kwargs)

    @tf.function
    def _transform_parameters_forward(self, parameters):

        if self.flexible_hrf_parameters:
            n_hrf_pars = len(self.hrf_model.parameter_labels)

            encoding_pars = DifferenceOfGaussiansPRF2D._transform_parameters_forward(self, parameters[:, :-n_hrf_pars])
            hrf_pars = self.hrf_model._transform_parameters_forward(parameters[:, -n_hrf_pars:])

            return tf.concat([encoding_pars, hrf_pars], axis=1)
        else:
            return DifferenceOfGaussiansPRF2D._transform_parameters_forward(self, parameters)

    @tf.function
    def _transform_parameters_backward(self, parameters):

        if self.flexible_hrf_parameters:
            n_hrf_pars = len(self.hrf_model.parameter_labels)

            encoding_pars = DifferenceOfGaussiansPRF2D._transform_parameters_backward(self, parameters[:, :-n_hrf_pars])
            hrf_pars = self.hrf_model._transform_parameters_backward(parameters[:, -n_hrf_pars:])

            return tf.concat([encoding_pars, hrf_pars], axis=1)
        else:
            return DifferenceOfGaussiansPRF2D._transform_parameters_backward(self, parameters)

class DivisiveNormalizationGaussianPRF2D(GaussianPRF2D):
    # Amplitude is as a fraction of the positive amplitude and is limited to be within [0, 1]
    # srf factor is limited to be above 1
    parameter_labels = ['x', 'y', 'sd',
                        'rf_amplitude', 'srf_amplitude', 'srf_size',
                        'neural_baseline', 'surround_baseline']

    @tf.function
    def _transform_parameters_forward(self, parameters):
        return tf.concat([parameters[:, 0][:, tf.newaxis], # x
                          parameters[:, 1][:, tf.newaxis], # y
                          tf.math.softplus(parameters[:, 2][:, tf.newaxis]), # sd
                          parameters[:, 3][:, tf.newaxis], # rf_amplitude
                          tf.math.softplus(parameters[:, 4][:, tf.newaxis]), # srf_amplitude
                          tf.math.softplus(parameters[:, 5][:, tf.newaxis]) + 1, # srf_size
                          tf.math.softplus(parameters[:, 6][:,tf.newaxis]), # neural_baseline
                          tf.math.softplus(parameters[:, 7][:,tf.newaxis]), # surround_baseline
                          ], axis=1)

    @tf.function
    def _transform_parameters_backward(self, parameters):
        return tf.concat([parameters[:, 0][:, tf.newaxis],
                          parameters[:, 1][:, tf.newaxis],
                          tfp.math.softplus_inverse(
                              parameters[:, 2][:, tf.newaxis]),
                          parameters[:, 3][:, tf.newaxis],
                          tfp.math.softplus_inverse(
                              parameters[:, 4][:, tf.newaxis]),
                          tfp.math.softplus_inverse(parameters[:, 5][:, tf.newaxis] - 1),
                          tfp.math.softplus_inverse(parameters[:, 6][:, tf.newaxis]),
                          tfp.math.softplus_inverse(parameters[:, 7][:, tf.newaxis])], axis=1)


    @tf.function
    def _basis_predictions(self, paradigm, parameters):
        # paradigm: n_batches x n_timepoints x n_stimulus_features
        # parameters:: n_batches x n_voxels x n_parameters

        # norm: n_batches x n_timepoints x n_voxels

        # output: n_batches x n_timepoints x n_voxels


        mu_x = parameters[:, :, 0, tf.newaxis]
        mu_y = parameters[:, :, 1, tf.newaxis]
        sd = parameters[:, :, 2, tf.newaxis]
        rf_parameters = tf.concat([mu_x, mu_y, sd, tf.zeros_like(mu_x), tf.ones_like(mu_x)], axis=2)
        rf = self._get_rf(self.grid_coordinates, rf_parameters)

        srf_size = parameters[:, :, 5, tf.newaxis]

        srf_parameters = tf.concat([mu_x, mu_y, sd*srf_size, tf.zeros_like(mu_x), tf.ones_like(mu_x)], axis=2)

        srf = self._get_rf(self.grid_coordinates, srf_parameters)


        # From n_batches x n_voxels to
        # n_batches x n_timespoints x n_populations
        rf_amplitude = parameters[:, :, 3][:, tf.newaxis, :]
        srf_amplitude = parameters[:, :, 4][:, tf.newaxis, :]
        neural_baseline = parameters[:, :, 6][:, tf.newaxis, :]
        surround_baseline = parameters[:, :, 7][:, tf.newaxis, :]

        neural_activation = rf_amplitude * tf.tensordot(paradigm, rf, (2, 2))[:, :, 0, :] + neural_baseline
        normalization = (srf_amplitude * rf_amplitude) * tf.tensordot(paradigm, srf, (2, 2))[:, :, 0, :] + surround_baseline

        normalized_activation = (neural_activation / normalization)

        return normalized_activation

class DivisiveNormalizationGaussianPRF2DWithHRF(HRFEncodingModel, DivisiveNormalizationGaussianPRF2D):

    parameter_labels = ['x', 'y', 'sd',
                        'rf_amplitude', 'srf_amplitude', 'srf_size',
                        'neural_baseline', 'surround_baseline',
                        'bold_baseline']

    def __init__(self, grid_coordinates=None, paradigm=None, data=None, parameters=None,
                 positive_image_values_only=True,
                 weights=None, hrf_model=None, flexible_hrf_parameters=False, verbosity=logging.INFO, **kwargs):

        DivisiveNormalizationGaussianPRF2D.__init__(self, grid_coordinates=grid_coordinates, paradigm=paradigm, data=data, parameters=parameters, weights=weights, verbosity=verbosity,
                        positive_image_values_only=positive_image_values_only, **kwargs)

        HRFEncodingModel.__init__(self, hrf_model=hrf_model, flexible_hrf_parameters=flexible_hrf_parameters, **kwargs)

    @tf.function
    def _transform_parameters_forward(self, parameters):

        if self.flexible_hrf_parameters:
            n_hrf_pars = len(self.hrf_model.parameter_labels)

            encoding_pars = DivisiveNormalizationGaussianPRF2D._transform_parameters_forward(self, parameters[:, :-n_hrf_pars-1])
            bold_baseline = parameters[:, -n_hrf_pars-1][:, tf.newaxis]
            hrf_pars = self.hrf_model._transform_parameters_forward(parameters[:, -n_hrf_pars-1:])

            return tf.concat([encoding_pars, bold_baseline, hrf_pars], axis=1)
        else:
            encoding_pars1 = DivisiveNormalizationGaussianPRF2D._transform_parameters_forward(self, parameters[:, :-1])
            bold_baseline = parameters[:, -1:]
            return tf.concat([encoding_pars1, bold_baseline], axis=1)

    @tf.function
    def _transform_parameters_backward(self, parameters):

        if self.flexible_hrf_parameters:
            n_hrf_pars = len(self.hrf_model.parameter_labels)

            encoding_pars = DivisiveNormalizationGaussianPRF2D._transform_parameters_backward(self, parameters[:, :-n_hrf_pars-1])
            bold_baseline = parameters[:, -n_hrf_pars-1][:, tf.newaxis]
            hrf_pars = self.hrf_model._transform_parameters_backward(parameters[:, -n_hrf_pars:])
            return tf.concat([encoding_pars, bold_baseline, hrf_pars], axis=1)

        else:
            encoding_pars1 =  DivisiveNormalizationGaussianPRF2D._transform_parameters_backward(self, parameters[:, :-1])
            bold_baseline = parameters[:, -1:]
            return tf.concat([encoding_pars1, bold_baseline], axis=1)


    @tf.function
    def _predict(self, paradigm, parameters, weights):


        pre_convolve_parameters = parameters[..., :8]
        pre_convolve = DivisiveNormalizationGaussianPRF2D._predict(self, paradigm, pre_convolve_parameters, weights)

        neural_baseline = parameters[tf.newaxis, :, :, 6]
        surround_baseline = parameters[tf.newaxis, :, :, 7]
        bold_baseline = parameters[tf.newaxis, :, :, 8]

        pre_convolve = pre_convolve - (neural_baseline / surround_baseline)


        kwargs = {}
        # parameters: n_batch x n_units x n_parameters
        if self.flexible_hrf_parameters:
            for ix, label in enumerate(self.hrf_model.parameter_labels):
                kwargs[label] = parameters[:, :, -len(self.hrf_model.parameter_labels) + ix]

        # pred: n_batch x n_timepoints x n_units
        pred_convolved = self.hrf_model.convolve(pre_convolve, **kwargs) + bold_baseline

        return pred_convolved
