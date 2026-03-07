import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
import keras
from keras import ops
from ..utils import norm, format_data, format_paradigm, format_parameters, format_weights, logit, restrict_radians, lognormalpdf_n, von_mises_pdf, lognormal_pdf_mode_fwhm, norm2d
from ..utils.math import aggressive_softplus, aggressive_softplus_inverse, norm
import scipy.stats as ss
from ..stimuli import Stimulus, OneDimensionalRadialStimulus, OneDimensionalGaussianStimulus, OneDimensionalStimulusWithAmplitude, OneDimensionalRadialStimulusWithAmplitude, ImageStimulus, TwoDimensionalStimulus
from patsy import dmatrix, build_design_matrices
from .base import EncodingModel, HRFEncodingModel

class DiscreteModel(EncodingModel):

    def __init__(self, paradigm=None, data=None, parameters=None,
                 weights=None, verbosity=logging.INFO):

        self.parameter_labels = ['stim=={}'.format(
            p) for p in np.diag(parameters)]
        _parameters = np.zeros_like(parameters) * np.nan
        _parameters[np.diag_indices(len(parameters))] = np.diag(parameters)

        super().__init__(paradigm, data, _parameters, weights, verbosity)

    def _basis_predictions(self, paradigm, parameters):

        parameters_ = ops.diag(parameters[0])

        return ops.cast(ops.equal(paradigm, parameters_[None, :]), 'float32')


class LinearModel(EncodingModel):
    """Identity mapping from paradigm features to voxel responses."""

    parameter_labels = []

    def __init__(self, paradigm=None, data=None, parameters=None,
                 weights=None, omega=None, verbosity=logging.INFO, **kwargs):

        if parameters is not None:
            raise ValueError('LinearModel does not use any parameters!')

        super().__init__(paradigm=paradigm, data=data, parameters=parameters,
                         weights=weights, verbosity=logging.INFO, **kwargs)

        if paradigm is not None:
            self.stimulus = self._get_stimulus(n_dimensions=paradigm.shape[1])
            self.paradigm = self.stimulus.clean_paradigm(paradigm)
        else:
            self.stimulus = self._get_stimulus()
            self.paradigm = None


    def predict(self, paradigm=None, parameters=None, weights=None):

        if parameters is not None:
            raise ValueError('LinearModel does not use any parameters!')

        return super().predict(paradigm, parameters, weights)

    def _basis_predictions(self, paradigm, parameters):
        return paradigm


class LinearModelWithBaseline(EncodingModel):
    """Linear encoding model that adds a voxel-specific baseline parameter."""

    parameter_labels = ['baseline']

    def _predict(self, paradigm, parameters, weights=None):

        basis_predictions = self._basis_predictions(paradigm, None)

        if weights is None:
            return basis_predictions + parameters[..., 0]
        else:
            return ops.tensordot(basis_predictions, weights, axes=[[2], [1]])[:, :, 0, :] + \
                ops.transpose(parameters, axes=[0, 2, 1])

    def _basis_predictions(self, paradigm, parameters):
        return paradigm


class LinearModelWithBaselineHRF(LinearModelWithBaseline, HRFEncodingModel):
    """LinearModelWithBaseline variant that automatically applies an HRF convolution."""

    def __init__(self, paradigm=None, data=None, parameters=None,
                 weights=None, hrf_model=None, verbosity=logging.INFO,
                 **kwargs):

        super().__init__(paradigm=paradigm,
                         data=data,
                         parameters=parameters,
                         weights=weights,
                         verbosity=verbosity,
                         hrf_model=hrf_model,
                         **kwargs)

    def _predict(self, paradigm, parameters, weights):
        pre_convolve = LinearModelWithBaseline._predict(
            self, paradigm, parameters, weights)

        return self.hrf_model.convolve(pre_convolve)

    def _predict_no_hrf(self, paradigm, parameters, weights):
        return LinearModelWithBaseline._predict(self, paradigm, parameters, weights)
