import tensorflow as tf
import tensorflow_probability as tfp
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from .utils import norm, format_data, format_paradigm, format_parameters, format_weights


class EncodingModel(object):

    parameter_labels = None

    def __init__(self, paradigm=None, data=None, parameters=None,
                 weights=None, verbosity=logging.INFO):

        self.paradigm = paradigm
        self.data = data
        self.parameters = parameters
        self.weights = weights

    @tf.function
    def _predict(self, paradigm, parameters, weights=None):
        if weights is None:
            return self._basis_predictions(paradigm, parameters)
        else:
            return tf.tensordot(self._basis_predictions(paradigm, parameters), weights, (1, 0))

    def predict(self, paradigm=None, parameters=None, weights=None):

        if paradigm is not None:
            self.paradigm = paradigm

        if parameters is not None:
            self.parameters = parameters

        if weights is not None:
            self.weights = weights

        predictions = self._predict(
            self.paradigm.values, self.parameters.values, self.weights)

        return format_data(predictions)

    def _fit_weights(self, y, paradigm, parameters, l2_cost=0.0):
        return tf.linalg.lstsq(self._basis_predictions(paradigm, parameters),
                               y,
                               l2_regularizer=l2_cost)

    def simulate(self, paradigm=None, parameters=None, weights=None, noise=1.):
        if paradigm is not None:
            self.paradigm = paradigm

        if parameters is not None:
            self.parameters = parameters

        if weights is not None:
            self.weights = weights

        self.data = self._simulate(self.paradigm, self.parameters, self.weights, noise)
        return self.data

    def _simulate(self, paradigm, parameters, weights, noise=1.):

        n_timepoints = paradigm.shape[0]

        if weights is None:
            n_voxels = parameters.shape[0]
        else:
            n_voxels = weights.shape[1]

        noise = tf.random.normal(shape=(n_timepoints, n_voxels),
                                 mean=0.0,
                                 stddev=noise,
                                 dtype=tf.float32)

        return self._predict(paradigm, parameters, weights) + noise

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        if data is None:
            self._data = None
        else:
            self._data = format_data(data)

    @property
    def paradigm(self):
        return self._paradigm

    @paradigm.setter
    def paradigm(self, paradigm):
        self._paradigm = format_paradigm(paradigm)

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, parameters):
        self._parameters = format_parameters(parameters, self.parameter_labels)

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights):
        self._weights = format_weights(weights)


    def to_discrete_model(self, grid, parameters=None, weights=None):
        
        grid = np.array(grid, dtype=np.float32)[:, np.newaxis]
        parameters = format_parameters(parameters)
        weights = format_weights(weights)
        
        if parameters is None:
            parameters = self.parameters
        
        if weights is None:
            weights = self.weights
            
        if weights is not None:
            weights = weights.value        

        discrete_weights = self._predict(grid, parameters.values, weights)
            
        return DiscreteModel(paradigm=self.paradigm, 
                      parameters=np.diag(grid[:, 0]),
                      weights=discrete_weights,
                      data=self.data)

    def get_stimulus_posterior(self, data, grid, omega=None, parameters=None, weights=None, normalize=True):
        
        n_voxels = data.shape[1]
        
        if omega is None:
            self.omega = omega
        
        grid = format_paradigm(grid[:, np.newaxis])
        
        predictions = self.predict(grid, parameters, weights)
        predictions.set_index(pd.MultiIndex.from_frame(grid))
        
        # time x grid x voxel
        residuals = data.values[:, np.newaxis, :] - predictions.values[np.newaxis, :, :]
        
        residual_dist = tfd.MultivariateNormalFullCovariance(
                        tf.zeros(n_voxels),
                        omega, allow_nan_stats=False)
        
        # we use log likelihood to correct for very small numbers
        p = pd.DataFrame(residual_dist.log_prob(residuals).numpy(),
                         index=data.index,
                         columns=pd.MultiIndex.from_frame(grid))
        
        p -= p.min().min()
        
        p = np.exp(p)

        if normalize:
            p = (p.T / p.T.sum()).T
        
        return p
        

class HRFEncodingModel(EncodingModel):

    def __init__(self, paradigm=None, data=None, parameters=None,
                 weights=None, hrf_model=None, verbosity=logging.INFO):

        if hrf_model is None:
            raise ValueError('Please provide HRFModel!')

        self.hrf_model = hrf_model

        super().__init__(paradigm, data, parameters, weights, verbosity)

    @tf.function
    def _predict(self, paradigm, parameters, weights):
        pre_convolve = EncodingModel._predict(
            self, paradigm, parameters, weights)

        return self.hrf_model.convolve(pre_convolve)

    def get_init_pars(self, data, paradigm, confounds=None):

        paradigm_shift = tf.cast(tf.math.round(
            self.hrf_model.delay / self.hrf_model.tr), tf.int32)

        padding = [[paradigm_shift, 0], [0, 0]]

        paradigm = tf.pad(paradigm, padding)[:-paradigm_shift]

        return super().get_init_pars(data, paradigm, confounds)


class GaussianPRF(EncodingModel):

    parameter_labels = ['mu', 'sd', 'amplitude', 'baseline']
    parameter_transforms = [None, 'softplus', None, None]

    def basis_predictions(self, paradigm):
        parameters = np.array(self.parameters)

        if paradigm.ndim == 1:
            paradigm = paradigm[:, np.newaxis]

        return self._basis_predictions(paradigm, parameters)

    def get_init_pars(self, data, paradigm, confounds=None):

        if confounds is not None:
            beta = tf.linalg.lstsq(confounds, data)
            predictions = (confounds @ beta)
            data -= predictions

        if hasattr(data, 'values'):
            data = data.values

        if hasattr(paradigm, 'values'):
            paradigm = paradigm.values

        baselines = tf.reduce_min(data, 0)
        data_ = (data - baselines)

        mus = tf.reduce_sum((data_ * paradigm), 0) / tf.reduce_sum(data_, 0)
        sds = tf.sqrt(tf.reduce_sum(data_ * (paradigm - mus)
                                    ** 2, 0) / tf.reduce_sum(data_, 0))
        amplitudes = tf.reduce_max(data_, 0)

        parameters = tf.concat([mus[:, tf.newaxis],
                                sds[:, tf.newaxis],
                                amplitudes[:, tf.newaxis],
                                baselines[:, tf.newaxis]], 1)

        return parameters

    @tf.function
    def _basis_predictions(self, paradigm, parameters):
        return norm(paradigm, parameters[:, 0], parameters[:, 1]) * parameters[:, 2] + parameters[:, 3]

    @tf.function
    def _transform_parameters_forward(self, parameters):
        return tf.concat([parameters[:, 0][:, tf.newaxis],
                          tf.math.softplus(parameters[:, 1][:, tf.newaxis]),
                          tf.math.softplus(parameters[:, 2][:, tf.newaxis]),
                          parameters[:, 3][:, tf.newaxis]], axis=1)

    @tf.function
    def _transform_parameters_backward(self, parameters):
        return tf.concat([parameters[:, 0][:, tf.newaxis],
                          tfp.math.softplus_inverse(
                              parameters[:, 1][:, tf.newaxis]),
                          tfp.math.softplus_inverse(
                              parameters[:, 2][:, tf.newaxis]),
                          parameters[:, 3][:, tf.newaxis]], axis=1)


class GaussianPRFWithHRF(GaussianPRF, HRFEncodingModel):
    pass

class DiscreteModel(EncodingModel):

    def __init__(self, paradigm=None, data=None, parameters=None,
            weights=None, verbosity=logging.INFO):

        self.parameter_labels = ['stim=={}'.format(p) for p in np.diag(parameters)]
        _parameters = np.zeros_like(parameters) * np.nan
        _parameters[np.diag_indices(len(parameters))] = np.diag(parameters)

        super().__init__(paradigm, data, _parameters, weights, verbosity)



    
    @tf.function
    def _basis_predictions(self, paradigm, parameters):
        
        parameters_ = tf.linalg.diag_part(parameters)
        
        return tf.cast(tf.equal(paradigm, parameters_[tf.newaxis, :]), tf.float32)
