import tensorflow as tf
import tensorflow_probability as tfp
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
from .utils import norm, format_data, format_paradigm, format_parameters, format_weights, logit, restrict_radians, lognormalpdf_n, von_mises_pdf, lognormal_pdf_mode_fwhm, norm2d
from tensorflow_probability import distributions as tfd
from braincoder.utils.math import aggressive_softplus, aggressive_softplus_inverse, norm
import pandas as pd
import scipy.stats as ss
from .stimuli import Stimulus, OneDimensionalRadialStimulus, OneDimensionalGaussianStimulus, OneDimensionalStimulusWithAmplitude, OneDimensionalRadialStimulusWithAmplitude, ImageStimulus, TwoDimensionalStimulus
from patsy import dmatrix, build_design_matrices

class EncodingModel(object):

    parameter_labels = None
    stimulus_type = Stimulus

    def __init__(self, paradigm=None, data=None, parameters=None,
                 weights=None, omega=None, verbosity=logging.INFO):

        if paradigm is not None:
            self.stimulus = self._get_stimulus(n_dimensions=paradigm.shape[1])
            self.paradigm = self.stimulus.clean_paradigm(paradigm)
        else:
            self.stimulus = self._get_stimulus()
            self.paradigm = None

        self.data = data
        self.parameters = format_parameters(parameters, parameter_labels=self.parameter_labels)

        if (self.parameter_labels is not None) and (self.parameters is not None):
            self.parameters = self.parameters[self.parameter_labels]

        self.weights = weights
        self.omega = omega

        if omega is not None:
            self.omega_chol = np.linalg.cholesky(omega)

    def get_parameter_labels(self):
        return self.parameter_labels

    @tf.function
    def _predict(self, paradigm, parameters, weights=None):

        # paradigm: n_batch x n_timepoints x n_stimulus_features
        # parameters: n_batch x n_units x n_parameters
        # weights: n_batch x n_basis_functions x n_units

        # returns: n_batch x n_timepoints x n_units
        if weights is None:
            return self._basis_predictions(paradigm, parameters)
        else:
            return tf.tensordot(self._basis_predictions(paradigm, parameters), weights, (2, 1))[:, :, 0, :]

    def _get_stimulus_type(self, **kwargs):
        return self.stimulus_type

    def _get_stimulus(self, **kwargs):
        return self.stimulus_type(**kwargs)

    def predict(self, paradigm=None, parameters=None, weights=None):

        weights, weights_ = self._get_weights(weights)
        
        paradigm = self.get_paradigm(paradigm)
        paradigm_ = self._get_paradigm(paradigm)[np.newaxis, ...]

        parameters = self._get_parameters(parameters)
        
        parameters_ = parameters.values[np.newaxis, ...] if parameters is not None else None

        predictions = self._predict(paradigm_, parameters_, weights_)[0]

        if weights is None:
            return pd.DataFrame(predictions.numpy(), index=paradigm.index, columns=parameters.index)
        else:
            return pd.DataFrame(predictions.numpy(), index=paradigm.index, columns=weights.columns)

    def simulate(self, paradigm=None, parameters=None, weights=None, noise=1.):

        weights, weights_ = self._get_weights(weights)
        paradigm = self.get_paradigm(paradigm)
        paradigm_ = self._get_paradigm(paradigm)

        if parameters is None:
            parameters = self.parameters
        else:
            parameters = format_parameters(parameters)

        parameters = self._get_parameters(parameters)

        if np.isscalar(noise):
            simulated_data = self._simulate(
                self.stimulus._generate_stimulus(paradigm_)[np.newaxis, ...],
                parameters.values[np.newaxis, ...],
                weights_, noise).numpy()[0]
        else:
            assert(noise.ndim == 2), 'noise should be either a scalar or a square covariance matrix'
            pred = self.predict(paradigm, parameters, weights)
            noise = ss.multivariate_normal(np.zeros(pred.shape[1]), cov=noise).rvs(len(pred)).astype(np.float32)
            simulated_data = pred + noise

        if weights is None:
            return pd.DataFrame(simulated_data, index=paradigm.index, columns=parameters.index)
        else:
            return pd.DataFrame(simulated_data, index=paradigm.index, columns=weights.columns)

    def _simulate(self, paradigm, parameters, weights, noise=1.):

        n_batches = paradigm.shape[0]
        n_timepoints = paradigm.shape[1]

        if weights is None:
            n_voxels = parameters.shape[1]
        else:
            n_voxels = weights.shape[2]

        noise = tf.random.normal(shape=(n_batches, n_timepoints, n_voxels),
                                 mean=0.0,
                                 stddev=noise,
                                 dtype=tf.float32)

        return self._predict(paradigm, parameters, weights) + noise

    def _gradient(self, stimuli, parameters):
        stimuli = tf.convert_to_tensor(stimuli)
        
        with tf.GradientTape() as tape:
            tape.watch(stimuli)
            predictions = self._predict(stimuli, parameters)
        
        # Compute the Jacobian, expected to result in [1, n, m, 1, n, 1]
        jacobians = tape.jacobian(predictions, stimuli)
        
        # Correct handling of the Jacobian to transform [1, n, m, 1, n, 1] to [1, n, m]
        # Sum over redundant dimensions, specifically the input's batch and spatial dimensions (since we want derivative w.r.t. each input independently)
        gradients = tf.reduce_sum(jacobians, axis=[-2, -1])

        return tf.squeeze(gradients, axis=-1)

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

    def likelihood(self, stimuli, data=None, parameters=None, weights=None, omega=None, dof=None, logp=False, normalize=True):

        if data is None:
            data = self.data
        else:
            data = format_data(data)

        if parameters is None:
            parameters = self.parameters
        else:
            parameters = format_parameters(parameters)

        if weights is not None:
            weights = weights if not hasattr(weights, 'values') else weights.values

        if not isinstance(stimuli, pd.DataFrame):
            stimuli = pd.DataFrame(stimuli)
            stimuli.index.name = 'stimulus'
            stimuli.columns.name = 'stimulus dimension'

        for name, value in zip(['data', 'parameters'], [data, parameters]):
            if value is None:
                raise Exception('Please set {}'.format(name))

        omega_chol = np.linalg.cholesky(omega)

        # stimuli: n_batches x n_timepoints x n_stimulus_features
        # data: n_batches x n_timepoints x n_units
        # parameters: n_batches x n_subpops x n_parmeters
        # weights: n_batches x n_subpops x n_units
        # omega: n_units x n_units

        # n_batches * n_timepoints x n_stimulus_features
        likelihood = self._likelihood(stimuli.values[np.newaxis, ...],
                                      data.values[np.newaxis, ...],
                                      parameters.values[np.newaxis, ...],
                                      weights[np.newaxis, ...] if weights else None,
                                      omega_chol,
                                      dof,
                                      logp,
                                      normalize).numpy()

        likelihood = pd.DataFrame(
            likelihood, index=data.index, columns=stimuli.index)

        return likelihood

    def get_stimulus_pdf(self, data, stimulus_range, parameters=None, weights=None, omega=None, dof=None, normalize=True):

        if hasattr(data, 'values'):
            time_index = data.index
            data = data.values
        else:
            time_index = pd.Index(np.arange(len(data)), name='frame')

        parameters = self._get_parameters(parameters)

        if hasattr(stimulus_range, 'values'):
            stimulus_range = stimulus_range.values

        if hasattr(parameters, 'values'):
            parameters = parameters.values

        if omega is None:
            omega = self.omega

        weights, weights_ = self._get_weights(weights)

        # stimuli: n_batches x n_timepoints x n_stimulus_features
        # data: n_batches x n_timepoints x n_units
        # parameters: n_batches x n_subpops x n_parmeters
        # weights: n_batches x n_subpops x n_units
        # omega: n_units x n_units

        stimulus_range = self.stimulus._clean_paradigm(stimulus_range)

        if stimulus_range.ndim == 1:
            stimulus_range = stimulus_range[:, np.newaxis, np.newaxis]
        elif stimulus_range.ndim == 2:
            stimulus_range = stimulus_range[:, np.newaxis, :]
        else:
            raise Exception('Stimulus range needs to be either 1D or 2D')

        # n_batches * n_timepoints x n_stimulus_features
        ll = self._likelihood(stimulus_range,
                              data[np.newaxis, :, :],
                              parameters[np.newaxis, :, :] if parameters is not None else None,
                              weights_,
                              omega,
                              dof,
                              logp=True,

                              normalize=False).numpy()

        
        if stimulus_range.shape[-1] == 1:
            ll = pd.DataFrame(ll.T, index=time_index, columns=pd.Index(
                stimulus_range[:, 0, 0], name='stimulus'))
        else:
            index = pd.MultiIndex.from_frame(pd.DataFrame(stimulus_range[:, 0, :],
                                             columns=self.stimulus.dimension_labels))
            ll = pd.DataFrame(ll.T, index=time_index, columns=index)

        # Normalize, working from log likelihoods (otherwise we get numerical issues)
        ll = np.exp(ll.apply(lambda d: d-d.max(), 1))
        # ll = ll.apply(lambda d: d/d.sum(), axis=1)

        # ll = np.exp(ll)

        if normalize:
            ll /= np.trapz(ll, ll.columns)[:, np.newaxis]

        return ll

    def apply_mask(self, mask):

        if self.data is not None:
            self.data = self.data.loc[:, mask]

        if self.weights is None:
            if self.parameters is not None:
                self.parameters = self.parameters.loc[mask]
        else:
            self.weights = self.weights.loc[:, mask]

    def get_WWT(self):
        return self.weights.T.dot(self.weights)

    def get_residual_dist(self, n_voxels, omega_chol, dof):

        if dof is None:
            residual_dist = tfd.MultivariateNormalTriL(
                tf.zeros(n_voxels),
                scale_tril=omega_chol, allow_nan_stats=False)
        else:
            residual_dist = tfd.MultivariateStudentTLinearOperator(
                dof,
                tf.zeros(n_voxels),
                tf.linalg.LinearOperatorLowerTriangular(omega_chol), allow_nan_stats=False)

        return residual_dist

    @tf.function
    def _likelihood(self, stimuli, data, parameters, weights, omega_chol, dof, logp=False, normalize=False):

        # stimuli: n_batches x n_timepoints x n_stimulus_features
        # data: n_batches x n_timepoints x n_units
        # parameters: n_batches x n_subpops x n_parmeters
        # weights: n_batches x n_subpops x n_units
        # omega: n_units x n_units

        # n_batches * n_timepoints x n_stimulus_features
        prediction = self._predict(stimuli, parameters, weights)

        return self._likelihood_timeseries(data, prediction, omega_chol, dof, logp, normalize)

    @tf.function
    def _likelihood_timeseries(self, data, prediction, omega_chol, dof, logp=False, normalize=False):
        # n_timepoints x n_stimuli x n_units
        n_units = data.shape[2]

        residuals = data - prediction
        residual_dist = self.get_residual_dist(n_units, omega_chol, dof)

        # we use log likelihood to correct for very small numbers
        p = residual_dist.log_prob(residuals)

        if logp:
            return p

        if normalize:
            p = p - tf.reduce_max(p, 1)[:, tf.newaxis]
            p = tf.exp(p)
            p = p / tf.reduce_sum(p, 1)[:, tf.newaxis]
        else:
            p = tf.exp(p)

        return p

    def _get_weights(self, weights=None):

        if (weights is None) and (self.weights is not None):
            weights = self.weights

        weights = format_weights(weights)

        if weights is None:
            weights_ = weights
        else:
            weights_ = weights.values[np.newaxis, ...]

        return weights, weights_

    def get_fisher_information(self, stimuli, omega=None, dof=None, weights=None, parameters=None, n=1000,
                               analytical=True):

        if analytical and (dof is not None):
            raise ValueError('Cannot use analytical Fisher information with t-distribution!')

        if omega is None:
            omega = self.omega

        if omega is None:
            raise ValueError("Need noise covariance matrix omega!")

        weights, weights_ = self._get_weights(weights)

        if parameters is None:
            if self.parameters is None:
                raise Exception('Need to set parameters')
            else:
                parameters = self.parameters


        parameters_ = parameters.values[np.newaxis, ...].astype(np.float32)

        if stimuli.ndim == 1:
            stimuli = stimuli[:, np.newaxis]


        L = tf.linalg.cholesky(omega)

        if analytical:
            stimuli_ = tf.Variable(stimuli[np.newaxis, ...], name='stimuli')
            gradient = self._gradient(stimuli_, parameters_)[0] # number of stimuli x number of voxels

            y = []

            for i in range(gradient.shape[0]):
                y.append(tf.linalg.triangular_solve(L, gradient[i, :, tf.newaxis], lower=True))

            y = tf.concat(y, axis=1)

            fisher_info = tf.reduce_sum(y ** 2, axis=0)

        else:
            stimuli_ = tf.repeat(stimuli[np.newaxis, ...], n, axis=0)
            stimuli_ = tf.Variable(stimuli_, name='stimuli')

            dist = self.get_residual_dist(omega.shape[0], L, dof)
            pred = self._predict(stimuli_, parameters_, weights_)
            noise = dist.sample(n)

            # Batches (noise) x stimuli x n_voxels
            data = pred + noise[:, tf.newaxis, :]

            with tf.GradientTape() as tape:
                ll = self._likelihood(stimuli_, data, parameters_, weights_, L, dof, logp=True, normalize=False)

            dy_dx = tape.gradient(ll, stimuli_)

            fisher_info = tf.reduce_mean(dy_dx**2, 0)[..., 0]

        if stimuli.shape[1] == 1:
            return pd.Series(fisher_info.numpy(), index=pd.Index(stimuli[:, 0], name='stimulus'), name='Fisher information')
        else:
            return pd.Series(fisher_info.numpy(), index=pd.MultiIndex.from_frame(pd.DataFrame(stimuli)), name='Fisher information')

    def _get_parameters(self, parameters=None):

        if (parameters is None) and (self.parameters is not None):
            parameters = self.parameters

        parameters = format_parameters(parameters)

        if parameters is not None:
            parameters = parameters[self.parameter_labels]

        return parameters

    def get_paradigm(self, paradigm):
            
        if paradigm is None:
            if self.paradigm is not None:
                return self.paradigm
            else:
                raise ValueError('Please provide paradigm!')

        paradigm = self.stimulus.clean_paradigm(paradigm)

        return paradigm

    def _get_paradigm(self, paradigm):
            
        if paradigm is None:
            paradigm = self.get_paradigm(paradigm)

        paradigm = self.stimulus._clean_paradigm(paradigm)

        return paradigm


class EncodingRegressionModel(EncodingModel):

    def __init__(self, paradigm=None, data=None, parameters=None,
                regressors={}, weights=None, omega=None,
                baseline_parameter_values=None,
                 verbosity=logging.INFO, **kwargs):

        self.regressors = regressors

        if paradigm is not None:
            self.stimulus = self._get_stimulus()

            for paradigm_label in self.stimulus.dimension_labels:
                if paradigm_label not in paradigm:
                    raise ValueError('Paradigm is missing required dimension: ' + paradigm_label + \
                                    '\nNote that `EncodingRegressionModel` requires a paradigm named stimulus dimensions!')

            base_paradigm = paradigm[self.stimulus.dimension_labels]

        else:
            raise ValueError('Please provide paradigm!')

        self.base_parameter_labels = self.parameter_labels
        self.set_paradigm(paradigm, regressors)

        super().__init__(paradigm=base_paradigm, data=data, parameters=parameters,
                         weights=weights, omega=omega, verbosity=logging.INFO, **kwargs)
        

        # If baseline_parameter_values is not provided, use empty dictionary
        baseline_parameter_values = baseline_parameter_values or {}

        # If parameter is not in baseline_parameter_values, set it to 0
        self.baseline_parameter_values = {
            param: baseline_parameter_values.get(param, 0.0)
            for param in self.base_parameter_labels
        }

        self._basis_basis_predictions = self._basis_predictions
        self._basis_predictions = self._basis_predictions_regressors

        self._base_transform_parameters_forward = self._transform_parameters_forward
        self._base_transform_parameters_backward = self._transform_parameters_backward

        self._transform_parameters_forward = lambda x: x
        self._transform_parameters_backward = lambda x: x


    def _get_regressor_parameter_labels(self, design_matrices):
        regressor_parameters = []

        for parameter in self.base_parameter_labels:
            regressor_parameters += zip([parameter+'_unbounded'] * design_matrices[parameter].shape[1],
                                        design_matrices[parameter].design_info.column_names)

        return pd.MultiIndex.from_tuples(regressor_parameters, names=['parameter', 'regressor'])

    def _get_base_parameters(self, design_matrices, regressor_parameters):

        parameters = []

        ix = 0

        for parameter in self.base_parameter_labels:
            end_ix = ix + design_matrices[parameter].shape[1]

            parameters.append(tf.reduce_sum(np.asarray(design_matrices[parameter], dtype=np.float32)[:, np.newaxis, :] * \
                                      regressor_parameters[:, :, ix:end_ix], axis=2) + self.baseline_parameter_values[parameter])

            ix = end_ix

        parameters = tf.stack(parameters, axis=2)
        parameters = tf.map_fn(self._base_transform_parameters_forward, parameters)

        return parameters

    def build_design_matrices(self, paradigm, regressors=None):

        design_matrices = {}

        if not hasattr(self, 'design_matrices'):
            for parameter in self.base_parameter_labels:
                if parameter in regressors:
                    design_matrices[parameter] = dmatrix(self.regressors[parameter], paradigm)
                else:
                    design_matrices[parameter] = dmatrix('1', paradigm)
        else:
            assert regressors is None, 'Regressors should be set when the model is initialized.'

            for parameter in self.base_parameter_labels:
                design_info = self.design_matrices[parameter].design_info
                design_matrices[parameter] = build_design_matrices([design_info], paradigm)[0]

        return design_matrices


    def set_paradigm(self, paradigm, regressors=None):

        if not hasattr(self, 'paradigm'):

            if regressors is None:
                regressors = {}

            if regressors is None:
                raise Exception('For EncodignRegressionModel, the regressors should be set when the model is initialized.')
                # regressors = self.regressors

            self.paradigm = paradigm

            self.design_matrices = self.build_design_matrices(paradigm, regressors)
            self.parameter_labels = self._get_regressor_parameter_labels(self.design_matrices)

        else:
            if regressors is not None:
                raise Exception('For EncodingRegressionModel, the regressors should be set when the model is initialized.')

            self.design_matrices = self.build_design_matrices(paradigm)
            self.paradigm = paradigm

        self.base_paradigm = paradigm[self.stimulus.dimension_labels]

    def _basis_predictions_regressors(self, paradigm, parameters):
        base_parameters = self._get_base_parameters(self.design_matrices, parameters)
        result = self._basis_basis_predictions(self.base_paradigm.values[:, np.newaxis, :], base_parameters)
        return tf.reshape(result, [1, result.shape[0], -1])

    def _get_paradigm(self, paradigm):

        # if not paradigm.equals(self.paradigm):
        #     raise Exception('For EncodignRegressionModel, the paradigm should be set when the model is initialized OR using set_paradigm().')

        paradigm = self.stimulus._clean_paradigm(paradigm)

        return paradigm

    def get_conditionspecific_parameters(self, conditions, parameters):

        design_matrices = self.build_design_matrices(conditions)

        if hasattr(parameters, 'values'):
            parameters_ = parameters.values
        else:
            parameters_ = np.array(parameters)

        parameters_ = parameters_[np.newaxis, ...]

        transformed_parameters = self._get_base_parameters(design_matrices, parameters_).numpy()

        transformed_parameters = np.reshape(transformed_parameters, (-1, transformed_parameters.shape[-1]))

        transformed_parameters = pd.DataFrame(transformed_parameters,
                                            index=pd.MultiIndex.from_product([conditions.index, parameters.index]),
                                            columns=self.base_parameter_labels)

        return transformed_parameters



class HRFEncodingModel(object):

    def __init__(self, paradigm=None, data=None, parameters=None,
                    weights=None, omega=None, hrf_model=None, verbosity=logging.INFO,
                    flexible_hrf_parameters=False, **kwargs):

        if hrf_model is None:
            raise ValueError('Please provide HRFModel!')

        self.hrf_model = hrf_model

        if flexible_hrf_parameters != hrf_model.unique_hrfs:
            hrf_model.set_unique_hrfs(flexible_hrf_parameters)

        if flexible_hrf_parameters:
            self.flexible_hrf_parameters = True
            self.parameter_labels = self.get_parameter_labels() + self.hrf_model.parameter_labels
        else:
            self.flexible_hrf_parameters = False

        self.parameters = self._get_parameters(parameters)


    @tf.function
    def _predict(self, paradigm, parameters, weights):

        standardized_parameters = tf.identity(parameters)

        n_batches, n_voxels = parameters.shape[0], parameters.shape[1]

         # We define that baseline and amplitude are applied to the HRF-convolved signal
         # If we don't do that, we can't easily recover these parameters using OLS...
        if 'baseline' in self.parameter_labels:
            baseline_idx = self.parameter_labels.index('baseline')
            indices_baseline = tf.constant([[i, j, baseline_idx] for i in range(n_batches) for j in range(n_voxels)], dtype=tf.int32)
            updates_baseline = tf.zeros((n_batches * n_voxels,), dtype=parameters.dtype)
            standardized_parameters = tf.tensor_scatter_nd_update(standardized_parameters, indices_baseline, updates_baseline)

        if 'amplitude' in self.parameter_labels:
            amplitude_idx = self.parameter_labels.index('amplitude')
            indices_amplitude = tf.constant([[i, j, amplitude_idx] for i in range(n_batches) for j in range(n_voxels)], dtype=tf.int32)
            updates_amplitude = tf.ones((n_batches * n_voxels,), dtype=parameters.dtype)
            standardized_parameters = tf.tensor_scatter_nd_update(standardized_parameters, indices_amplitude, updates_amplitude)

        pre_convolve = EncodingModel._predict(
            self, paradigm, standardized_parameters, weights)

        kwargs = {}
        # parameters: n_batch x n_units x n_parameters
        if self.flexible_hrf_parameters:
            for ix, label in enumerate(self.hrf_model.parameter_labels):
                kwargs[label] = parameters[:, :, -len(self.hrf_model.parameter_labels) + ix]

        # pred: n_batch x n_timepoints x n_units
        pred_convolved = self.hrf_model.convolve(pre_convolve, **kwargs)

        if 'amplitude' in self.parameter_labels:
            pred_convolved *= parameters[:, :, amplitude_idx][:, tf.newaxis, :]
        
        if 'baseline' in self.parameter_labels:
            pred_convolved += parameters[:, :, baseline_idx][:, tf.newaxis, :]

        return pred_convolved

    @tf.function
    def _predict_no_hrf(self, paradigm, parameters, weights):
        return EncodingModel._predict(self, paradigm, parameters, weights)

class GaussianPRF(EncodingModel):

    parameter_labels = ['mu', 'sd', 'amplitude', 'baseline']

    def __init__(self, paradigm=None, data=None, parameters=None,
                 weights=None, omega=None, allow_neg_amplitudes=False, verbosity=logging.INFO,
                 model_stimulus_amplitude=False,
                 **kwargs):

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
        if model_stimulus_amplitude:
            return OneDimensionalStimulusWithAmplitude
        else:
            return Stimulus

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
        # paradigm: n_batches x n_timepoints x n_stimulus_features
        # parameters:: n_batches x n_voxels x n_parameters

        # norm: n_batches x n_timepoints x n_voxels

        # output: n_batches x n_timepoints x n_voxels
        return norm(paradigm[..., tf.newaxis, 0],
                    parameters[:, tf.newaxis, :, 0],
                    parameters[:, tf.newaxis, :, 1]) * \
            parameters[:, tf.newaxis, :, 2] * paradigm[:, :, tf.newaxis, 1] + parameters[:, tf.newaxis, :, 3]


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
        return tf.concat([parameters[:, 0][:, tf.newaxis],
                          aggressive_softplus(parameters[:, 1][:, tf.newaxis]), 
                          parameters[:, 2][:, tf.newaxis],
                          parameters[:, 3][:, tf.newaxis]], axis=1)

    @tf.function
    def _transform_parameters_backward1(self, parameters):
        return tf.concat([parameters[:, 0][:, tf.newaxis],
                          aggressive_softplus_inverse(
                              parameters[:, 1][:, tf.newaxis]),
                          parameters[:, 2][:, tf.newaxis],
                          parameters[:, 3][:, tf.newaxis]], axis=1)

    @tf.function
    def _transform_parameters_forward2(self, parameters):
        return tf.concat([parameters[:, 0][:, tf.newaxis],
                          aggressive_softplus(parameters[:, 1][:, tf.newaxis]),
                          aggressive_softplus(parameters[:, 2][:, tf.newaxis]),
                          parameters[:, 3][:, tf.newaxis]], axis=1)

    @tf.function
    def _transform_parameters_backward2(self, parameters):
        return tf.concat([parameters[:, 0][:, tf.newaxis],
                          aggressive_softplus_inverse(
                              parameters[:, 1][:, tf.newaxis]),
                            aggressive_softplus_inverse(
                                parameters[:, 2][:, tf.newaxis]),
                          parameters[:, 3][:, tf.newaxis]], axis=1)

class RegressionGaussianPRF(EncodingRegressionModel, GaussianPRF):
    pass
class VonMisesPRF(GaussianPRF):

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
        if model_stimulus_amplitude:
            return OneDimensionalRadialStimulusWithAmplitude
        else:
            return OneDimensionalRadialStimulus

    @tf.function
    def _basis_predictions_without_amplitude(self, paradigm, parameters):
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
        # paradigm: n_batches x n_timepoints x n_stimulus_features
        # parameters:: n_batches x n_voxels x n_parameters

        # norm: n_batches x n_timepoints x n_voxels

        # output: n_batches x n_timepoints x n_voxels
        return von_mises_pdf(paradigm[..., tf.newaxis, 0],
                    parameters[:, tf.newaxis, :, 0],
                    parameters[:, tf.newaxis, :, 1]) * \
            parameters[:, tf.newaxis, :, 2] * paradigm[..., tf.newaxis, 1] + parameters[:, tf.newaxis, :, 3]

    def init_pseudoWWT(self, stimulus_range, parameters):

        if stimulus_range.ndim == 2:
            stimulus_range = stimulus_range[:, [0]]

        stimulus_range = np.stack((stimulus_range, np.ones_like(stimulus_range)), axis=1).astype(np.float32)
        W = self.basis_predictions(stimulus_range, parameters)

        pseudoWWT = tf.tensordot(W, W, (0, 0))
        self._pseudoWWT = tf.where(tf.math.is_nan(pseudoWWT), tf.zeros_like(pseudoWWT),
                                   pseudoWWT)
        return self._pseudoWWT

class LogGaussianPRF(GaussianPRF):

    parameter_labels = ['mu', 'sd', 'amplitude', 'baseline']

    def __init__(self, paradigm=None, data=None, parameters=None,
                 weights=None, omega=None, allow_neg_amplitudes=False, verbosity=logging.INFO,
                 model_stimulus_amplitude=False,
                 parameterisation='mu_sd_natural',
                 **kwargs):

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


        super().__init__(paradigm=paradigm, data=data, parameters=parameters,
                         weights=weights, omega=omega, allow_neg_amplitudes=allow_neg_amplitudes,
                          verbosity=verbosity, model_stimulus_amplitude=model_stimulus_amplitude,
                          **kwargs)

    @tf.function
    def _transform_parameters_forward1(self, parameters):
        return tf.concat([tf.math.softplus(parameters[:, 0][:, tf.newaxis]),
                          tf.math.softplus(parameters[:, 1][:, tf.newaxis]),
                          parameters[:, 2][:, tf.newaxis],
                          parameters[:, 3][:, tf.newaxis]], axis=1)

    @tf.function
    def _transform_parameters_backward1(self, parameters):
        return tf.concat([tfp.math.softplus_inverse(parameters[:, 0][:, tf.newaxis]),
                          tfp.math.softplus_inverse(
                              parameters[:, 1][:, tf.newaxis]),
                          parameters[:, 2][:, tf.newaxis],
                          parameters[:, 3][:, tf.newaxis]], axis=1)
    @tf.function
    def _transform_parameters_forward2(self, parameters):
        return tf.concat([tf.math.softplus(parameters[:, 0][:, tf.newaxis]),
                          tf.math.softplus(parameters[:, 1][:, tf.newaxis]),
                          tf.math.softplus(parameters[:, 2][:, tf.newaxis]),
                          parameters[:, 3][:, tf.newaxis]], axis=1)

    @tf.function
    def _transform_parameters_backward2(self, parameters):
        return tf.concat([tfp.math.softplus_inverse(parameters[:, 0][:, tf.newaxis]),
                          tfp.math.softplus_inverse(
                              parameters[:, 1][:, tf.newaxis]),
                          tfp.math.softplus_inverse(parameters[:, 2][:, tf.newaxis]),
                          parameters[:, 3][:, tf.newaxis]], axis=1)
    @tf.function
    def _basis_predictions_without_amplitude_n(self, paradigm, parameters):
        return lognormalpdf_n(paradigm[..., tf.newaxis, 0],
                    parameters[:, tf.newaxis, :, 0],
                    parameters[:, tf.newaxis, :, 1]) * \
            parameters[:, tf.newaxis, :, 2] + parameters[:, tf.newaxis, :, 3]

    @tf.function
    def _basis_predictions_with_amplitude_n(self, paradigm, parameters):
        return lognormalpdf_n(paradigm[..., tf.newaxis, 0],
                    parameters[:, tf.newaxis, :, 0],
                    parameters[:, tf.newaxis, :, 1]) * \
            parameters[:, tf.newaxis, :, 2] * paradigm[..., tf.newaxis, 1] + parameters[:, tf.newaxis, :, 3]

    @tf.function
    def _basis_predictions_without_amplitude_mode_fwhm(self, paradigm, parameters):
        return lognormal_pdf_mode_fwhm(paradigm[..., tf.newaxis, 0],
                    parameters[:, tf.newaxis, :, 0],
                    parameters[:, tf.newaxis, :, 1]) * \
            parameters[:, tf.newaxis, :, 2] + parameters[:, tf.newaxis, :, 3]

    @tf.function
    def _basis_predictions_with_amplitude_mode_fwhm(self, paradigm, parameters):
        return lognormal_pdf_mode_fwhm(paradigm[..., tf.newaxis, 0],
                    parameters[:, tf.newaxis, :, 0],
                    parameters[:, tf.newaxis, :, 1]) * \
            parameters[:, tf.newaxis, :, 2] * paradigm[..., tf.newaxis, 1] + parameters[:, tf.newaxis, :, 3]

class GaussianPRFWithHRF(GaussianPRF, HRFEncodingModel):
    pass

class LogGaussianPRFWithHRF(LogGaussianPRF, HRFEncodingModel):
    pass

class GaussianPRFOnGaussianSignal(GaussianPRF):

    stimulus_type = OneDimensionalGaussianStimulus

    def __init__(self, paradigm=None, data=None, parameters=None,
                 weights=None, omega=None, allow_neg_amplitudes=False,
                  stimulus_grid=None, verbosity=logging.INFO,
                 **kwargs):

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
        # n_stim-grid x n-batches x n-timepoints x n_voxels
        rf_field = norm(self.stimulus_grid[:, tf.newaxis, tf.newaxis, tf.newaxis],  #grid to evaluate on
                        parameters[tf.newaxis, :, tf.newaxis, :, 0],
                        parameters[tf.newaxis, :, tf.newaxis, :, 1])

        input_stimulus = norm(self.stimulus_grid[:, tf.newaxis, tf.newaxis, tf.newaxis],  #grid to evaluate on
                        paradigm[tf.newaxis, ..., 0, tf.newaxis],
                        paradigm[tf.newaxis, ..., 1, tf.newaxis])
        
        return tf.reduce_sum(rf_field * input_stimulus, axis=0)

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

        GaussianPRF2DAngle.__init__(self, grid_coordinates=grid_coordinates, paradigm=paradigm, data=data, parameters=parameters, weights=weights, verbosity=verbosity,
                        positive_image_values_only=positive_image_values_only, **kwargs)

        HRFEncodingModel.__init__(self, hrf_model=hrf_model, flexible_hrf_parameters=flexible_hrf_parameters, **kwargs)

    def to_linear_model(self):
        return LinearModelWithBaselineHRF(self.paradigm, self.data,
                                          self.parameters[[
                                              'baseline']], weights=self.get_rf().T,
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
    @tf.function
    def _transform_parameters_forward(self, parameters):
        return tf.concat([parameters[:, 0][:, tf.newaxis],
                          parameters[:, 1][:, tf.newaxis],
                          tf.math.softplus(parameters[:, 2][:, tf.newaxis]),
                          parameters[:, 3][:, tf.newaxis],
                          tf.math.softplus(parameters[:, 4][:, tf.newaxis]),
                          tf.math.softplus(parameters[:, 5][:, tf.newaxis]),
                          tf.math.softplus(parameters[:, 6][:, tf.newaxis]) + 1], axis=1)

    @tf.function
    def _transform_parameters_backward(self, parameters):
        return tf.concat([parameters[:, 0][:, tf.newaxis],
                          parameters[:, 1][:, tf.newaxis],
                          tfp.math.softplus_inverse(
                              parameters[:, 2][:, tf.newaxis]),
                          parameters[:, 3][:, tf.newaxis],
                          tfp.math.softplus_inverse(
                              parameters[:, 4][:, tf.newaxis]),
                          tfp.math.softplus_inverse(
                              parameters[:, 5][:, tf.newaxis]),
                          tfp.math.softplus_inverse(parameters[:, 6][:, tf.newaxis] - 1)], axis=1)

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
                          tf.math.softplus(parameters[:, 6][:, tf.newaxis]),
                          tf.math.softplus(parameters[:, 7][:, tf.newaxis])], axis=1)


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

        pre_convolve = pre_convolve - (neural_baseline + surround_baseline)


        kwargs = {}
        # parameters: n_batch x n_units x n_parameters
        if self.flexible_hrf_parameters:
            for ix, label in enumerate(self.hrf_model.parameter_labels):
                kwargs[label] = parameters[:, :, -len(self.hrf_model.parameter_labels) + ix]

        # pred: n_batch x n_timepoints x n_units
        pred_convolved = self.hrf_model.convolve(pre_convolve, **kwargs) + bold_baseline

        return pred_convolved


class DiscreteModel(EncodingModel):

    def __init__(self, paradigm=None, data=None, parameters=None,
                 weights=None, verbosity=logging.INFO):

        self.parameter_labels = ['stim=={}'.format(
            p) for p in np.diag(parameters)]
        _parameters = np.zeros_like(parameters) * np.nan
        _parameters[np.diag_indices(len(parameters))] = np.diag(parameters)

        super().__init__(paradigm, data, _parameters, weights, verbosity)

    @tf.function
    def _basis_predictions(self, paradigm, parameters):

        parameters_ = tf.linalg.diag_part(parameters)

        return tf.cast(tf.equal(paradigm, parameters_[tf.newaxis, :]), tf.float32)


class LinearModel(EncodingModel):
    
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

    @tf.function
    def _basis_predictions(self, paradigm, parameters):
        return paradigm


class LinearModelWithBaseline(EncodingModel):

    parameter_labels = ['baseline']

    @tf.function
    def _predict(self, paradigm, parameters, weights=None):

        basis_predictions = self._basis_predictions(paradigm, None)

        if weights is None:
            return basis_predictions + parameters[..., 0]
        else:
            return tf.tensordot(basis_predictions, weights, (2, 1))[:, :, 0, :] + \
                tf.transpose(parameters, [0, 2, 1])

    @tf.function
    def _basis_predictions(self, paradigm, parameters):
        return paradigm


class LinearModelWithBaselineHRF(LinearModelWithBaseline, HRFEncodingModel):

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

    @tf.function
    def _predict(self, paradigm, parameters, weights):
        pre_convolve = LinearModelWithBaseline._predict(
            self, paradigm, parameters, weights)

        return self.hrf_model.convolve(pre_convolve)

    @tf.function
    def _predict_no_hrf(self, paradigm, parameters, weights):
        return LinearModelWithBaseline._predict(self, paradigm, parameters, weights)
