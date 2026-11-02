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
    """One response per discrete stimulus value.

    ``stimulus_values`` lists the values; each becomes one parameter. Use
    :meth:`get_parameters` for the matching parameter table.
    """

    def __init__(self, stimulus_values, verbosity=logging.INFO, **kwargs):
        self.stimulus_values = np.asarray(stimulus_values, dtype=np.float32)
        self.parameter_labels = ['stim=={}'.format(p) for p in self.stimulus_values]
        super().__init__(verbosity=verbosity, **kwargs)

    def get_parameters(self):
        parameters = np.full((len(self.stimulus_values),) * 2, np.nan, dtype=np.float32)
        parameters[np.diag_indices_from(parameters)] = self.stimulus_values
        return pd.DataFrame(parameters, columns=self.parameter_labels)

    def _basis_predictions(self, paradigm, parameters):

        parameters_ = ops.diag(parameters[0])

        return ops.cast(ops.equal(paradigm, parameters_[None, :]), 'float32')


class LinearModel(EncodingModel):
    """Identity mapping from paradigm features to voxel responses."""

    parameter_labels = []

    def predict(self, paradigm, parameters=None, weights=None):

        if parameters is not None:
            raise ValueError('LinearModel does not use any parameters!')

        return super().predict(paradigm, parameters, weights)

    def _basis_predictions(self, paradigm, parameters):
        return paradigm


class LinearModelWithBaseline(EncodingModel):
    """Linear encoding model that adds a voxel-specific baseline parameter."""

    parameter_labels = ['baseline']
    transformations = ['identity']

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

    def __init__(self, hrf_model=None, verbosity=logging.INFO,
                 **kwargs):

        if hrf_model is None:
            raise ValueError('Please provide HRFModel!')

        LinearModelWithBaseline.__init__(self, verbosity=verbosity, **kwargs)
        self.hrf_model = hrf_model
        self.flexible_hrf_parameters = False

    def _predict(self, paradigm, parameters, weights):
        paradigm = ops.convert_to_tensor(paradigm, dtype='float32')
        parameters = ops.convert_to_tensor(parameters, dtype='float32')
        # The baseline is a BOLD offset: convolve the signal, then add it.
        pre_convolve = LinearModelWithBaseline._predict(
            self, paradigm, ops.zeros_like(parameters), weights)

        return self.hrf_model.convolve(pre_convolve) + ops.transpose(parameters, axes=[0, 2, 1])

    def _predict_grid(self, stimulus_range, parameters, weights):
        return self._predict_no_hrf(stimulus_range, parameters, weights)

    def _predict_no_hrf(self, paradigm, parameters, weights):
        return LinearModelWithBaseline._predict(self, paradigm, parameters, weights)
