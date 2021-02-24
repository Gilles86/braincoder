import tensorflow as tf
import tensorflow_probability as tfp
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from .utils import norm, format_data, format_paradigm, format_parameters


class EncodingModel(object):

    parameter_labels = None

    def __init__(self, paradigm=None, data=None, parameters=None,
                 weights=None, verbosity=logging.INFO):

        self.paradigm = paradigm
        self.data = data
        self.parameters = parameters
        self.weights = weights

        if (self.weights is None) and (self.parameters is not None) and (self.data is not None) and (len(self.parameters) == self.data.shape[1]):
            self.weights = np.identity(len(self.parameters), dtype=np.float32)

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

    def simulate(self, paradigm, parameters, weights=None, noise=1.):
        self.paradigm = paradigm
        self.parameters = parameters

        self.data = self._simulate(self.paradigm, self.parameters, weights, noise)
        return self.data

    def _simulate(self, paradigm, parameters, weights, noise=1.):

        n_timepoints = paradigm.shape[0]
        n_voxels = parameters.shape[0]

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
