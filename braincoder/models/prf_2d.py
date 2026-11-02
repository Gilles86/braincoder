import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
import keras
from keras import ops
from ..utils import norm, format_data, format_paradigm, format_parameters, format_weights, logit, restrict_radians, lognormalpdf_n, von_mises_pdf, lognormal_pdf_mode_fwhm, norm2d
from ..utils.math import aggressive_softplus, aggressive_softplus_inverse, norm
from ..utils.backend import softplus_inverse
import scipy.stats as ss
from ..stimuli import Stimulus, OneDimensionalRadialStimulus, OneDimensionalGaussianStimulus, OneDimensionalStimulusWithAmplitude, OneDimensionalRadialStimulusWithAmplitude, ImageStimulus, TwoDimensionalStimulus
from patsy import dmatrix, build_design_matrices
from .base import _reject_state_kwargs, EncodingModel, HRFEncodingModel

class GaussianPointPRF2D(EncodingModel):

    def __init__(self, allow_neg_amplitudes=False, verbosity=logging.INFO,
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

        super().__init__(verbosity=logging.INFO, **kwargs)

    def _get_stimulus_type(self, model_stimulus_amplitude=False):
            return TwoDimensionalStimulus

    def _get_basis_predictions(self, model_stimulus_amplitude=False):
        if model_stimulus_amplitude:
            return self._basis_predictions_with_amplitude
        else:
            return self._basis_predictions_without_amplitude

    def basis_predictions(self, paradigm, parameters):

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
            beta = ops.lstsq(confounds, data)
            predictions = (confounds @ beta)
            data -= predictions

        if hasattr(data, 'values'):
            data = data.values

        baselines = ops.min(data, axis=0)
        data_ = (data - baselines)

        mus_x = ops.sum((data_ * self.stimulus._generate_stimulus(paradigm.values[..., 0])), axis=0) / ops.sum(data_, axis=0)
        mus_y = ops.sum((data_ * self.stimulus._generate_stimulus(paradigm.values[..., 1])), axis=0) / ops.sum(data_, axis=0)
        sds_x = ops.sqrt(ops.sum(data_ * (self.stimulus._generate_stimulus(paradigm.values[..., 0]) - mus_x)
                                    ** 2, axis=0) / ops.sum(data_, axis=0))
        sds_y = ops.sqrt(ops.sum(data_ * (self.stimulus._generate_stimulus(paradigm.values[..., 1]) - mus_y)
                                    ** 2, axis=0) / ops.sum(data_, axis=0))
        amplitudes = ops.max(data_, axis=0)

        if self.correlated_response:
            covariances = ops.sum((data_ * (self.stimulus._generate_stimulus(paradigm.values[..., 0]) - mus_x) *
                                        (self.stimulus._generate_stimulus(paradigm.values[..., 1]) - mus_y)), axis=0) / ops.sum(data_, axis=0)
            parameters = ops.concatenate([mus_x[:, None],
                                    mus_y[:, None],
                                    sds_x[:, None],
                                    sds_y[:, None],
                                    covariances[:, None],
                                    amplitudes[:, None],
                                    baselines[:, None]], axis=1)
        else:
            parameters = ops.concatenate([mus_x[:, None],
                                    mus_y[:, None],
                                    sds_x[:, None],
                                    sds_y[:, None],
                                    amplitudes[:, None],
                                    baselines[:, None]], axis=1)

        return parameters

    def _basis_predictions_without_amplitude(self, paradigm, parameters):
        if self.correlated_response:
            return norm2d(paradigm[..., None, 0],
                          paradigm[..., None, 1],
                          parameters[:, None, :, 0],
                          parameters[:, None, :, 1],
                          parameters[:, None, :, 2],
                          parameters[:, None, :, 3],
                          parameters[:, None, :, 4]) * \
                   parameters[:, None, :, 5] + parameters[:, None, :, 6]
        else:
            return norm2d(paradigm[..., None, 0],
                          paradigm[..., None, 1],
                          parameters[:, None, :, 0],
                          parameters[:, None, :, 1],
                          parameters[:, None, :, 2],
                          parameters[:, None, :, 3]) * \
                   parameters[:, None, :, 4] + parameters[:, None, :, 5]

    def _basis_predictions_with_amplitude(self, paradigm, parameters):
        if self.correlated_response:
            return norm2d(paradigm[..., None, 0],
                          paradigm[..., None, 1],
                          parameters[:, None, :, 0],
                          parameters[:, None, :, 1],
                          parameters[:, None, :, 2],
                          parameters[:, None, :, 3],
                          parameters[:, None, :, 4]) * \
                   parameters[:, None, :, 5] * paradigm[:, :, None, 2] + parameters[:, None, :, 6]
        else:
            return norm2d(paradigm[..., None, 0],
                          paradigm[..., None, 1],
                          parameters[:, None, :, 0],
                          parameters[:, None, :, 1],
                          parameters[:, None, :, 2],
                          parameters[:, None, :, 3]) * \
                   parameters[:, None, :, 4] * paradigm[:, :, None, 2] + parameters[:, None, :, 5]

    def get_pseudoWWT(self, stimulus_range, parameters, subtract_baseline=True):

        stimulus_range = stimulus_range.astype(np.float32)
        if subtract_baseline:
            parameters = self._zero_baseline(parameters)
        W = self.basis_predictions(stimulus_range, parameters)

        pseudoWWT = ops.tensordot(W, W, axes=[[0], [0]])
        return ops.where(ops.isnan(pseudoWWT), ops.zeros_like(pseudoWWT), pseudoWWT)


    def _transform_parameters_forward1(self, parameters):

        if self.correlated_response:
            return ops.concatenate([parameters[:, 0][:, None],
                              parameters[:, 1][:, None],
                              ops.softplus(parameters[:, 2][:, None]),
                              ops.softplus(parameters[:, 3][:, None]),
                              ops.softplus(parameters[:, 4][:, None]) * 2 - 1,
                              parameters[:, 5][:, None],
                              parameters[:, 6][:, None]], axis=1)
        else:
            return ops.concatenate([parameters[:, 0][:, None],
                             parameters[:, 1][:, None],
                             ops.softplus(parameters[:, 2][:, None]),
                             ops.softplus(parameters[:, 3][:, None]),
                             parameters[:, 4][:, None],
                             parameters[:, 5][:, None]], axis=1)

    def _transform_parameters_backward1(self, parameters):

        if self.correlated_response:
            return ops.concatenate([parameters[:, 0][:, None],
                              parameters[:, 1][:, None],
                              softplus_inverse(parameters[:, 2][:, None]),
                              softplus_inverse(parameters[:, 3][:, None]),
                              softplus_inverse((parameters[:, 4][:, None] + 1) / 2.),
                              parameters[:, 5][:, None],
                              parameters[:, 6][:, None]], axis=1)

        else:
            return ops.concatenate([parameters[:, 0][:, None],
                             parameters[:, 1][:, None],
                             softplus_inverse(parameters[:, 2][:, None]),
                             softplus_inverse(parameters[:, 3][:, None]),
                             parameters[:, 4][:, None],
                             parameters[:, 5][:, None]], axis=1)

    def _transform_parameters_forward2(self, parameters):

        if self.correlated_response:
            return ops.concatenate([parameters[:, 0][:, None],
                              parameters[:, 1][:, None],
                              ops.softplus(parameters[:, 2][:, None]),
                              ops.softplus(parameters[:, 3][:, None]),
                              ops.softplus(parameters[:, 4][:, None]) * 2 - 1,
                              ops.softplus(parameters[:, 5][:, None]),
                              parameters[:, 6][:, None]], axis=1)
        else:
            return ops.concatenate([parameters[:, 0][:, None],
                             parameters[:, 1][:, None],
                             ops.softplus(parameters[:, 2][:, None]),
                             ops.softplus(parameters[:, 3][:, None]),
                             ops.softplus(parameters[:, 4][:, None]),
                             parameters[:, 5][:, None]], axis=1)

    def _transform_parameters_backward2(self, parameters):

        if self.correlated_response:
            return ops.concatenate([parameters[:, 0][:, None],
                              parameters[:, 1][:, None],
                              softplus_inverse(parameters[:, 2][:, None]),
                              softplus_inverse(parameters[:, 3][:, None]),
                              softplus_inverse((parameters[:, 4][:, None] + 1) / 2.),
                              softplus_inverse(parameters[:, 5][:, None]),
                              parameters[:, 6][:, None]], axis=1)

        else:
            return ops.concatenate([parameters[:, 0][:, None],
                             parameters[:, 1][:, None],
                             softplus_inverse(parameters[:, 2][:, None]),
                             softplus_inverse(parameters[:, 3][:, None]),
                             softplus_inverse(parameters[:, 4][:, None]),
                             parameters[:, 5][:, None]], axis=1)

class GaussianMixturePRF2D(EncodingModel):

    def __init__(self, allow_neg_amplitudes=False, verbosity=logging.INFO,
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

        super().__init__(verbosity=logging.INFO, **kwargs)

    def _get_stimulus_type(self):
        return TwoDimensionalStimulus

    def _get_basis_predictions(self, model_stimulus_amplitude=False):
        if model_stimulus_amplitude:
            return self._basis_predictions_with_amplitude
        else:
            return self._basis_predictions_without_amplitude

    def basis_predictions(self, paradigm, parameters):

        paradigm = self.get_paradigm(paradigm)
        parameters = self._get_parameters(parameters)

        if hasattr(parameters, 'values'):
            parameters = parameters.values

        parameters = np.float32(parameters)

        return self._basis_predictions(self.stimulus._generate_stimulus(paradigm.values), parameters[np.newaxis, ...])[0]

    def _basis_predictions(self, paradigm, parameters):

        if self.same_rfs:
            return (parameters[:, None, :, 2] * norm(paradigm[..., None, 0],
                                                          parameters[:, None, :, 0],
                                                          parameters[:, None, :, 1]) + \
                    (1 - parameters[:, None, :, 2]) * norm(paradigm[..., None, 1],
                                                                 parameters[:, None, :, 0],
                                                          parameters[:, None, :, 1])) * \
                     parameters[:, None, :, 3] + parameters[:, None, :, 4]

        else:
            return (parameters[:, None, :, 4] * norm(paradigm[..., None, 0],
                                                    parameters[:, None, :, 0],
                                                    parameters[:, None, :, 2]) + \
                    (1 - parameters[:, None, :, 4]) * norm(paradigm[..., None, 1],
                                                            parameters[:, None, :, 1],
                                                            parameters[:, None, :, 3])) * \
                    parameters[:, None, :, 5] + parameters[:, None, :, 6]

    def get_pseudoWWT(self, stimulus_range, parameters, subtract_baseline=True):

        stimulus_range = stimulus_range.astype(np.float32)
        if subtract_baseline:
            parameters = self._zero_baseline(parameters)
        W = self.basis_predictions(stimulus_range, parameters)

        pseudoWWT = ops.tensordot(W, W, axes=[[0], [0]])
        return ops.where(ops.isnan(pseudoWWT), ops.zeros_like(pseudoWWT), pseudoWWT)



    def _transform_parameters_forward1(self, parameters):

        if self.same_rfs:
            return ops.concatenate([parameters[:, 0][:, None],
                            ops.softplus(parameters[:, 1][:, None]),
                            ops.sigmoid(parameters[:, 2][:, None]),
                            ops.softplus(parameters[:, 3][:, None]),
                            parameters[:, 4][:, None]],
                            axis=1)

        else:
            return ops.concatenate([parameters[:, 0][:, None],
                            parameters[:, 1][:, None],
                            ops.softplus(parameters[:, 2][:, None]),
                            ops.softplus(parameters[:, 3][:, None]),
                            ops.sigmoid(parameters[:, 4][:, None]),
                            parameters[:, 5][:, None],
                            parameters[:, 6][:, None]], axis=1)

    def _transform_parameters_backward1(self, parameters):
        if self.same_rfs:
            return ops.concatenate([parameters[:, 0][:, None],
                            softplus_inverse(parameters[:, 1][:, None]),
                            logit(parameters[:, 2][:, None]),
                            softplus_inverse(parameters[:, 3][:, None]),
                            parameters[:, 4][:, None]],
                            axis=1)
        else:
            return ops.concatenate([parameters[:, 0][:, None],
                            parameters[:, 1][:, None],
                            softplus_inverse(parameters[:, 2][:, None]),
                            softplus_inverse(parameters[:, 3][:, None]),
                            logit(parameters[:, 4][:, None]),
                            parameters[:, 5][:, None],
                            parameters[:, 6][:, None]], axis=1)

    def _transform_parameters_forward2(self, parameters):
        if self.same_rfs:
            return ops.concatenate([parameters[:, 0][:, None],
                              ops.softplus(parameters[:, 1][:, None]),
                              ops.sigmoid(parameters[:, 2][:, None]),
                              ops.softplus(parameters[:, 3][:, None]),
                              parameters[:, 4][:, None]], axis=1)
        else:
            return ops.concatenate([parameters[:, 0][:, None],
                              parameters[:, 1][:, None],
                              ops.softplus(parameters[:, 2][:, None]),
                              ops.softplus(parameters[:, 3][:, None]),
                              ops.sigmoid(parameters[:, 4][:, None]),
                              ops.softplus(parameters[:, 5][:, None]),
                              parameters[:, 6][:, None]], axis=1)

    def _transform_parameters_backward2(self, parameters):
        if self.same_rfs:
            return ops.concatenate([parameters[:, 0][:, None],
                              softplus_inverse(parameters[:, 1][:, None]),
                              logit(parameters[:, 2][:, None]),
                              softplus_inverse(parameters[:, 3][:, None]),
                              parameters[:, 4][:, None]], axis=1)
        else:
            return ops.concatenate([parameters[:, 0][:, None],
                              parameters[:, 1][:, None],
                              softplus_inverse(parameters[:, 2][:, None]),
                              softplus_inverse(parameters[:, 3][:, None]),
                              logit(parameters[:, 4][:, None]),
                              softplus_inverse(parameters[:, 5][:, None]),
                              parameters[:, 6][:, None]], axis=1)

class GaussianPRF2D(EncodingModel):
    """2D Gaussian population receptive field on a pixel grid.

    ``rf_normalization`` (ctor kwarg) sets the scale of the Gaussian
    before ``amplitude`` multiplies it, and so what the amplitude means:

    - ``'legacy'`` (default): divided by ``σ·√(2π) / pixel_area``.
    - ``'peak'``: peak 1 (prfpy's ``normalize_RFs=False``); amplitude is
      the drive per stimulated pixel at the pRF centre.
    - ``'area'``: integral 1 over the visual field (``pixel_area /
      (2π σ²)``); ``<stimulus, G>`` is the stimulated fraction of the pRF.

    The three are reparameterisations of each other: with σ fixed,
    amplitudes convert exactly by the ratio of the scale factors.
    """

    parameter_labels = ['x', 'y', 'sd', 'baseline', 'amplitude']
    stimulus_type = ImageStimulus
    RF_NORMALIZATIONS = ('legacy', 'peak', 'area')

    def __init__(self, grid_coordinates=None, positive_image_values_only=True, verbosity=logging.INFO,
                 rf_normalization='legacy', **kwargs):

        if rf_normalization not in self.RF_NORMALIZATIONS:
            raise ValueError(f'rf_normalization must be one of {self.RF_NORMALIZATIONS}, '
                             f'got {rf_normalization!r}')
        _reject_state_kwargs(kwargs, type(self).__name__)
        self.rf_normalization = rf_normalization

        if grid_coordinates is None:
            raise ValueError(f'{type(self).__name__} needs grid_coordinates: (n_pixels, 2) x and y.')

        self.grid_coordinates = pd.DataFrame(
            grid_coordinates, columns=['x', 'y'])
        self._grid_coordinates = self.grid_coordinates.values

        self.n_x = len(self.grid_coordinates['x'].unique())
        self.n_y = len(self.grid_coordinates['y'].unique())
        self.stimulus = self.stimulus_type(self.grid_coordinates, positive_only=positive_image_values_only)

        x_diff = np.diff(np.sort(self.grid_coordinates['x'].unique())).mean()
        y_diff = np.diff(np.sort(self.grid_coordinates['y'].unique())).mean()
        self.pixel_area = x_diff * y_diff

    #: The pseudo-WWT of an image model comes from its receptive fields.
    pseudo_wwt_needs_stimulus_range = False

    def get_rf(self, parameters, as_frame=False, unpack=False):

        grid_coordinates = self.grid_coordinates.values

        index = format_parameters(parameters).index
        parameters = self._get_parameters(parameters).values[np.newaxis, ...]

        rf = self._get_rf(grid_coordinates, parameters)
        rf = ops.convert_to_numpy(rf)[0]

        if as_frame:
            rf = pd.concat([pd.DataFrame(e,
                                         index=pd.MultiIndex.from_frame(self.grid_coordinates))
                            for e in rf],
                           keys=index)

            if unpack:
                rf = rf.unstack('x').sort_index(ascending=False)

        return rf

    def _basis_predictions(self, paradigm, parameters):

        rf = self._get_rf(self.grid_coordinates, parameters)
        baseline = parameters[:, None, :, 3]
        result = ops.tensordot(paradigm, rf, axes=[[2], [2]])[:, :, 0, :] + baseline

        return result

    def _get_rf(self, grid_coordinates, parameters, normalize=True):

        if hasattr(grid_coordinates, 'values'):
            grid_coordinates = grid_coordinates.values

        x = ops.convert_to_tensor(grid_coordinates[:, 0][None, None, :], dtype='float32')
        y = ops.convert_to_tensor(grid_coordinates[:, 1][None, None, :], dtype='float32')
        parameters = ops.convert_to_tensor(parameters, dtype='float32')

        mu_x = parameters[:, :, 0, None]
        mu_y = parameters[:, :, 1, None]
        sd = parameters[:, :, 2, None]
        amplitude = parameters[:, :, 4, None]

        gauss = (ops.exp(-((x-mu_x)**2 + (y-mu_y)**2)/(2*sd**2))) * amplitude

        if normalize:
            gauss = gauss * self._rf_scale(sd)

        return gauss

    def _rf_scale(self, sd):
        """Factor applied to a peak-1 Gaussian of width ``sd`` (see ``rf_normalization``)."""
        if self.rf_normalization == 'peak':
            return ops.ones_like(sd)
        if self.rf_normalization == 'area':
            return float(self.pixel_area) / (2 * np.pi * sd ** 2)
        return float(self.pixel_area) / (sd * np.sqrt(2 * np.pi))

    def _transform_parameters_forward(self, parameters):
        return ops.concatenate([parameters[:, 0][:, None],
                          parameters[:, 1][:, None],
                          ops.softplus(parameters[:, 2][:, None]),
                          parameters[:, 3][:, None],
                          parameters[:, 4][:, None]], axis=1)

    def _transform_parameters_backward(self, parameters):
        return ops.concatenate([parameters[:, 0][:, None],
                          parameters[:, 1][:, None],
                          softplus_inverse(parameters[:, 2][:, None]),
                          parameters[:, 3][:, None],
                          parameters[:, 4][:, None]], axis=1)

    def get_pseudoWWT(self, stimulus_range=None, parameters=None, subtract_baseline=True):
        """Tuning similarity between voxels, from their receptive fields."""

        parameters = self._get_parameters(parameters).copy()

        if subtract_baseline:
            parameters['baseline'] = np.float32(0.0)

        rf = self.get_rf(parameters.astype(np.float32))

        return rf.dot(rf.T)

    def to_linear_model(self, parameters):
        """Equivalent linear model: returns ``(model, parameters, weights)``."""
        from .linear import LinearModelWithBaseline
        parameters = self._get_parameters(parameters)
        return (LinearModelWithBaseline(n_dimensions=len(self.grid_coordinates)),
                parameters[['baseline']],
                pd.DataFrame(self.get_rf(parameters).T, columns=parameters.index))

    def unpack_stimulus(self, stimulus):
        return np.reshape(stimulus, (-1, self.n_x, self.n_y))

class GaussianPRF2DAngle(GaussianPRF2D):

    parameter_labels = ['theta', 'ecc', 'sd', 'baseline', 'amplitude']

    def _get_rf(self, grid_coordinates, parameters):

        if hasattr(grid_coordinates, 'values'):
            grid_coordinates = grid_coordinates.values

        x = ops.convert_to_tensor(grid_coordinates[:, 0][None, None, :], dtype='float32')
        y = ops.convert_to_tensor(grid_coordinates[:, 1][None, None, :], dtype='float32')
        parameters = ops.convert_to_tensor(parameters, dtype='float32')

        theta = parameters[:, :, 0, None]
        ecc = parameters[:, :, 1, None]
        mu_x = ops.cos(theta) * ecc
        mu_y = ops.sin(theta) * ecc
        sd = parameters[:, :, 2, None]
        amplitude = parameters[:, :, 4, None]

        return (ops.exp(-((x-mu_x)**2 + (y-mu_y)**2)/(2*sd**2))) * amplitude

    def _transform_parameters_forward(self, parameters):
        return ops.concatenate([parameters[:, 0][:, None],
                          ops.softplus(parameters[:, 1][:, None]),
                          ops.softplus(parameters[:, 2][:, None]),
                          parameters[:, 3][:, None],
                          parameters[:, 4][:, None]], axis=1)

    def _transform_parameters_backward(self, parameters):
        return ops.concatenate([restrict_radians(parameters[:, 0][:, None]),
                          softplus_inverse(parameters[:, 1][:, None]),
                          softplus_inverse(parameters[:, 2][:, None]),
                          parameters[:, 3][:, None],
                          parameters[:, 4][:, None]], axis=1)

    def to_linear_model(self, parameters):
        """Equivalent linear model: returns ``(model, parameters, weights)``."""
        from .linear import LinearModelWithBaseline
        parameters = self._get_parameters(parameters)
        return (LinearModelWithBaseline(n_dimensions=len(self.grid_coordinates)),
                parameters[['baseline']],
                pd.DataFrame(self.get_rf(parameters).T, columns=parameters.index))

    def unpack_stimulus(self, stimulus):
        return np.reshape(stimulus, (-1, self.n_x, self.n_y))

    def _xy_parameters(self, parameters):
        parameters = self._get_parameters(parameters).copy()
        parameters['x'] = np.cos(parameters['theta']) * parameters['ecc']
        parameters['y'] = np.sin(parameters['theta']) * parameters['ecc']
        return parameters[['x', 'y', 'sd', 'baseline', 'amplitude']]

    def to_xy_model(self, parameters):
        """Equivalent x/y model: returns ``(model, parameters)``."""
        return GaussianPRF2D(grid_coordinates=self.grid_coordinates), self._xy_parameters(parameters)

class GaussianPRF2DWithHRF(HRFEncodingModel, GaussianPRF2D):
    def __init__(self, grid_coordinates=None, positive_image_values_only=True,
                 hrf_model=None, flexible_hrf_parameters=False, verbosity=logging.INFO, **kwargs):

        GaussianPRF2D.__init__(self, grid_coordinates=grid_coordinates, verbosity=verbosity,
                               positive_image_values_only=positive_image_values_only, **kwargs)
        HRFEncodingModel.__init__(self, hrf_model=hrf_model, flexible_hrf_parameters=flexible_hrf_parameters, **kwargs)

    def to_linear_model(self, parameters):
        """Equivalent linear model: returns ``(model, parameters, weights)``."""
        from .linear import LinearModelWithBaselineHRF
        parameters = self._get_parameters(parameters)
        return (LinearModelWithBaselineHRF(hrf_model=self.hrf_model, n_dimensions=len(self.grid_coordinates)),
                parameters[['baseline']],
                pd.DataFrame(self.get_rf(parameters).T, columns=parameters.index))

    def _transform_parameters_forward(self, parameters):
        if self.flexible_hrf_parameters:
            n_hrf_pars = len(self.hrf_model.parameter_labels)
            encoding_pars = GaussianPRF2D._transform_parameters_forward(self, parameters[:, :-n_hrf_pars])
            hrf_pars = self.hrf_model._transform_parameters_forward(parameters[:, -n_hrf_pars:])
            return ops.concatenate([encoding_pars, hrf_pars], axis=1)
        else:
            return GaussianPRF2D._transform_parameters_forward(self, parameters)

    def _transform_parameters_backward(self, parameters):
        if self.flexible_hrf_parameters:
            n_hrf_pars = len(self.hrf_model.parameter_labels)
            encoding_pars = GaussianPRF2D._transform_parameters_backward(self, parameters[:, :-n_hrf_pars])
            hrf_pars = self.hrf_model._transform_parameters_backward(parameters[:, -n_hrf_pars:])
            return ops.concatenate([encoding_pars, hrf_pars], axis=1)
        else:
            return GaussianPRF2D._transform_parameters_backward(self, parameters)

class GaussianPRF2DAngleWithHRF(HRFEncodingModel, GaussianPRF2DAngle):

    def __init__(self, grid_coordinates=None, hrf_model=None, verbosity=logging.INFO,
                  positive_image_values_only=True, flexible_hrf_parameters=False, **kwargs):

        GaussianPRF2DAngle.__init__(self, grid_coordinates=grid_coordinates, verbosity=verbosity,
                        positive_image_values_only=positive_image_values_only, **kwargs)

        HRFEncodingModel.__init__(self, hrf_model=hrf_model, flexible_hrf_parameters=flexible_hrf_parameters, **kwargs)
        self.hrf_model = hrf_model

    def to_linear_model(self, parameters):
        """Equivalent linear model: returns ``(model, parameters, weights)``."""
        from .linear import LinearModelWithBaselineHRF
        parameters = self._get_parameters(parameters)
        return (LinearModelWithBaselineHRF(hrf_model=self.hrf_model, n_dimensions=len(self.grid_coordinates)),
                parameters[['baseline']],
                pd.DataFrame(self.get_rf(parameters).T, columns=parameters.index))

    def to_xy_model(self, parameters):
        """Equivalent x/y model: returns ``(model, parameters)``."""
        return (GaussianPRF2DWithHRF(grid_coordinates=self.grid_coordinates, hrf_model=self.hrf_model),
                self._xy_parameters(parameters))

class DifferenceOfGaussiansPRF2D(GaussianPRF2D):

    parameter_labels = ['x', 'y', 'sd', 'baseline',
                        'amplitude', 'srf_amplitude', 'srf_size']

    transformations = ['identity', 'identity', 'softplus', 'identity',
                       'softplus', 'softplus', 'softplus']

    # The parent Gaussian transforms cover only the first 5 columns.
    def _transform_parameters_forward(self, parameters):
        gauss_pars = GaussianPRF2D._transform_parameters_forward(self, parameters[:, :5])
        srf_amplitude = ops.softplus(parameters[:, 5][:, None])
        srf_size = ops.softplus(parameters[:, 6][:, None]) + 1.0   # surround wider than center
        return ops.concatenate([gauss_pars, srf_amplitude, srf_size], axis=1)

    def _transform_parameters_backward(self, parameters):
        gauss_pars = GaussianPRF2D._transform_parameters_backward(self, parameters[:, :5])
        srf_amplitude = softplus_inverse(parameters[:, 5][:, None])
        srf_size = softplus_inverse(parameters[:, 6][:, None] - 1.0)
        return ops.concatenate([gauss_pars, srf_amplitude, srf_size], axis=1)

    def _get_rf(self, grid_coordinates, parameters):

        mu_x = parameters[:, :, 0, None]
        mu_y = parameters[:, :, 1, None]
        sd = parameters[:, :, 2, None]
        amplitude = parameters[:, :, 4, None]

        srf_amplitude = parameters[:, :, 5, None]
        srf_size = parameters[:, :, 6, None]

        standard_prf = super()._get_rf(grid_coordinates, parameters)

        srf_pars = ops.concatenate([mu_x, mu_y, sd*srf_size, ops.zeros_like(mu_x), srf_amplitude*amplitude*srf_size], axis=2)
        sprf = super()._get_rf(grid_coordinates, srf_pars)

        return standard_prf - sprf

class DifferenceOfGaussiansPRF2DWithHRF(HRFEncodingModel, DifferenceOfGaussiansPRF2D):

    def __init__(self, grid_coordinates=None, positive_image_values_only=True,
                 hrf_model=None, flexible_hrf_parameters=False, verbosity=logging.INFO, **kwargs):

        DifferenceOfGaussiansPRF2D.__init__(self, grid_coordinates=grid_coordinates, verbosity=verbosity,
                        positive_image_values_only=positive_image_values_only, **kwargs)

        HRFEncodingModel.__init__(self, hrf_model=hrf_model, flexible_hrf_parameters=flexible_hrf_parameters, **kwargs)

    def _transform_parameters_forward(self, parameters):

        if self.flexible_hrf_parameters:
            n_hrf_pars = len(self.hrf_model.parameter_labels)

            encoding_pars = DifferenceOfGaussiansPRF2D._transform_parameters_forward(self, parameters[:, :-n_hrf_pars])
            hrf_pars = self.hrf_model._transform_parameters_forward(parameters[:, -n_hrf_pars:])

            return ops.concatenate([encoding_pars, hrf_pars], axis=1)
        else:
            return DifferenceOfGaussiansPRF2D._transform_parameters_forward(self, parameters)

    def _transform_parameters_backward(self, parameters):

        if self.flexible_hrf_parameters:
            n_hrf_pars = len(self.hrf_model.parameter_labels)

            encoding_pars = DifferenceOfGaussiansPRF2D._transform_parameters_backward(self, parameters[:, :-n_hrf_pars])
            hrf_pars = self.hrf_model._transform_parameters_backward(parameters[:, -n_hrf_pars:])

            return ops.concatenate([encoding_pars, hrf_pars], axis=1)
        else:
            return DifferenceOfGaussiansPRF2D._transform_parameters_backward(self, parameters)

class DivisiveNormalizationGaussianPRF2D(GaussianPRF2D):
    parameter_labels = ['x', 'y', 'sd',
                        'rf_amplitude', 'srf_amplitude', 'srf_size',
                        'neural_baseline', 'surround_baseline']

    def _transform_parameters_forward(self, parameters):
        return ops.concatenate([parameters[:, 0][:, None],
                          parameters[:, 1][:, None],
                          ops.softplus(parameters[:, 2][:, None]),
                          parameters[:, 3][:, None],
                          ops.softplus(parameters[:, 4][:, None]),
                          ops.softplus(parameters[:, 5][:, None]) + 1,
                          ops.softplus(parameters[:, 6][:, None]),
                          ops.softplus(parameters[:, 7][:, None]),
                          ], axis=1)

    def _transform_parameters_backward(self, parameters):
        return ops.concatenate([parameters[:, 0][:, None],
                          parameters[:, 1][:, None],
                          softplus_inverse(parameters[:, 2][:, None]),
                          parameters[:, 3][:, None],
                          softplus_inverse(parameters[:, 4][:, None]),
                          softplus_inverse(parameters[:, 5][:, None] - 1),
                          softplus_inverse(parameters[:, 6][:, None]),
                          softplus_inverse(parameters[:, 7][:, None])], axis=1)

    def _basis_predictions(self, paradigm, parameters):

        mu_x = parameters[:, :, 0, None]
        mu_y = parameters[:, :, 1, None]
        sd = parameters[:, :, 2, None]
        rf_parameters = ops.concatenate([mu_x, mu_y, sd, ops.zeros_like(mu_x), ops.ones_like(mu_x)], axis=2)
        rf = self._get_rf(self.grid_coordinates, rf_parameters)

        srf_size = parameters[:, :, 5, None]

        srf_parameters = ops.concatenate([mu_x, mu_y, sd*srf_size, ops.zeros_like(mu_x), ops.ones_like(mu_x)], axis=2)

        srf = self._get_rf(self.grid_coordinates, srf_parameters)

        rf_amplitude = parameters[:, :, 3][:, None, :]
        srf_amplitude = parameters[:, :, 4][:, None, :]
        neural_baseline = parameters[:, :, 6][:, None, :]
        surround_baseline = parameters[:, :, 7][:, None, :]

        neural_activation = rf_amplitude * ops.tensordot(paradigm, rf, axes=[[2], [2]])[:, :, 0, :] + neural_baseline
        normalization = (srf_amplitude * ops.abs(rf_amplitude)) * ops.tensordot(paradigm, srf, axes=[[2], [2]])[:, :, 0, :] + surround_baseline

        normalized_activation = (neural_activation / normalization)

        return normalized_activation

class DivisiveNormalizationGaussianPRF2DWithHRF(HRFEncodingModel, DivisiveNormalizationGaussianPRF2D):

    parameter_labels = ['x', 'y', 'sd',
                        'rf_amplitude', 'srf_amplitude', 'srf_size',
                        'neural_baseline', 'surround_baseline',
                        'bold_baseline']

    def __init__(self, grid_coordinates=None, positive_image_values_only=True,
                 hrf_model=None, flexible_hrf_parameters=False, verbosity=logging.INFO, **kwargs):

        DivisiveNormalizationGaussianPRF2D.__init__(self, grid_coordinates=grid_coordinates, verbosity=verbosity,
                        positive_image_values_only=positive_image_values_only, **kwargs)

        HRFEncodingModel.__init__(self, hrf_model=hrf_model, flexible_hrf_parameters=flexible_hrf_parameters, **kwargs)

    def _transform_parameters_forward(self, parameters):

        if self.flexible_hrf_parameters:
            n_hrf_pars = len(self.hrf_model.parameter_labels)

            encoding_pars = DivisiveNormalizationGaussianPRF2D._transform_parameters_forward(self, parameters[:, :-n_hrf_pars-1])
            bold_baseline = parameters[:, -n_hrf_pars-1][:, None]
            hrf_pars = self.hrf_model._transform_parameters_forward(parameters[:, -n_hrf_pars:])

            return ops.concatenate([encoding_pars, bold_baseline, hrf_pars], axis=1)
        else:
            encoding_pars1 = DivisiveNormalizationGaussianPRF2D._transform_parameters_forward(self, parameters[:, :-1])
            bold_baseline = parameters[:, -1:]
            return ops.concatenate([encoding_pars1, bold_baseline], axis=1)

    def _transform_parameters_backward(self, parameters):

        if self.flexible_hrf_parameters:
            n_hrf_pars = len(self.hrf_model.parameter_labels)

            encoding_pars = DivisiveNormalizationGaussianPRF2D._transform_parameters_backward(self, parameters[:, :-n_hrf_pars-1])
            bold_baseline = parameters[:, -n_hrf_pars-1][:, None]
            hrf_pars = self.hrf_model._transform_parameters_backward(parameters[:, -n_hrf_pars:])
            return ops.concatenate([encoding_pars, bold_baseline, hrf_pars], axis=1)

        else:
            encoding_pars1 = DivisiveNormalizationGaussianPRF2D._transform_parameters_backward(self, parameters[:, :-1])
            bold_baseline = parameters[:, -1:]
            return ops.concatenate([encoding_pars1, bold_baseline], axis=1)

    def _predict_no_hrf(self, paradigm, parameters, weights):
        parameters = ops.convert_to_tensor(parameters, dtype='float32')
        pre_convolve = DivisiveNormalizationGaussianPRF2D._predict(self, paradigm, parameters[..., :8], weights)
        return pre_convolve - parameters[None, :, :, 6] / parameters[None, :, :, 7] + parameters[None, :, :, 8]

    def _predict(self, paradigm, parameters, weights):

        parameters = ops.convert_to_tensor(parameters, dtype='float32')
        pre_convolve_parameters = parameters[..., :8]
        pre_convolve = DivisiveNormalizationGaussianPRF2D._predict(self, paradigm, pre_convolve_parameters, weights)

        neural_baseline = parameters[None, :, :, 6]
        surround_baseline = parameters[None, :, :, 7]
        bold_baseline = parameters[None, :, :, 8]

        pre_convolve = pre_convolve - (neural_baseline / surround_baseline)

        kwargs = {}
        if self.flexible_hrf_parameters:
            for ix, label in enumerate(self.hrf_model.parameter_labels):
                kwargs[label] = parameters[:, :, -len(self.hrf_model.parameter_labels) + ix]

        pred_convolved = self.hrf_model.convolve(pre_convolve, **kwargs) + bold_baseline

        return pred_convolved
