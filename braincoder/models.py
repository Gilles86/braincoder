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
    """Abstract base class for encoding models.

    Handles paradigm/parameter formatting, TensorFlow prediction graphs,
    and utilities such as simulation, gradients, and noise injection. Most
    concrete models only need to implement ``_basis_predictions`` (and
    optionally ``_predict``) to become drop-in replacements across the
    fitting/decoding stack.
    """

    parameter_labels = None
    stimulus_type = Stimulus

    def __init__(self, paradigm=None, data=None, parameters=None,
                 weights=None, omega=None, verbosity=logging.INFO):
        """Normalize paradigm/parameter inputs and set shared attributes."""

        if paradigm is not None:

            if paradigm.ndim == 1:
                paradigm = paradigm[:, np.newaxis]

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
        """Return the ordered list of parameter labels used by the model."""
        return self.parameter_labels

    @tf.function
    def _predict(self, paradigm, parameters, weights=None):
        """Low-level TF prediction graph used by ``predict``/``simulate``."""

        # paradigm: n_batch x n_timepoints x n_stimulus_features
        # parameters: n_batch x n_units x n_parameters
        # weights: n_batch x n_basis_functions x n_units

        # returns: n_batch x n_timepoints x n_units
        if weights is None:
            return self._basis_predictions(paradigm, parameters)
        else:
            return tf.tensordot(self._basis_predictions(paradigm, parameters), weights, (2, 1))[:, :, 0, :]

    def _get_stimulus_type(self, **kwargs):
        """Return the ``Stimulus`` subclass used to clean/generate paradigms."""
        return self.stimulus_type
    
    def _get_stimulus(self, **kwargs):
        """Instantiate the configured ``Stimulus`` type."""
        return self.stimulus_type(**kwargs)
    
    def predict(self, paradigm=None, parameters=None, weights=None):
        """Return pandas predictions for the provided paradigm/parameters."""

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

    def simulate(self, paradigm=None, parameters=None, weights=None, noise=1.,
                dof=None,
                n_repeats=1):
        """Generate synthetic data by adding Gaussian/Student noise to predictions."""

        weights, weights_ = self._get_weights(weights)
        paradigm = self.get_paradigm(paradigm)
        paradigm_ = self._get_paradigm(paradigm)

        if parameters is None:
            parameters = self.parameters
        else:
            parameters = format_parameters(parameters)

        parameters = self._get_parameters(parameters)

        stimulus = self.stimulus._generate_stimulus(paradigm_) 

        stimulus = np.repeat(stimulus[np.newaxis, ...], n_repeats, axis=0)

        # if np.isscalar(noise):
        simulated_data = self._simulate(
            stimulus,
            parameters.values[np.newaxis, ...],
            weights_, noise, dof).numpy()


        # Collapse the first two dimensions
        simulated_data = np.reshape(simulated_data, (n_repeats*paradigm.shape[0], simulated_data.shape[2]))

        if n_repeats == 1:
            index = pd.Index(paradigm.index, name='stimulus')
        else:
            # index = pd.MultiIndex.from_product([paradigm.index, np.arange(n_repeats)], names=['stimulus', 'repeat'])
            index = pd.MultiIndex.from_product([np.arange(n_repeats), paradigm.index], names=['repeat', 'stimulus'])

        if weights is None:
            return pd.DataFrame(simulated_data, index=index, columns=parameters.index)
        else:
            return pd.DataFrame(simulated_data, index=index, columns=weights.columns)

    def _simulate(self, paradigm, parameters, weights, noise=1., dof=None):
        """TensorFlow implementation of ``simulate`` supporting noise sampling."""

        n_batches = paradigm.shape[0]
        n_timepoints = paradigm.shape[1]

        if weights is None:
            n_voxels = parameters.shape[1]
        else:
            n_voxels = weights.shape[2]

        if dof is None:
            if tf.experimental.numpy.isscalar(noise):
                noise = tf.random.normal(shape=(n_batches, n_timepoints, n_voxels),
                                        mean=0.0,
                                        stddev=noise,
                                        dtype=tf.float32)
            else:
                noise = noise.astype(np.float32)
                mvn = tfd.MultivariateNormalTriL(tf.zeros(n_voxels, dtype=np.float32),  tf.linalg.cholesky(noise))
                noise = mvn.sample((n_batches, n_timepoints))
        else:
            if tf.experimental.numpy.isscalar(noise):
                dist = tfd.StudentT(df=dof, loc=0.0, scale=noise)
                noise = dist.sample((n_batches, n_timepoints, n_voxels))
            else:
                noise = noise.astype(np.float32)
                mvn = tfd.MultivariateStudentTLinearOperator(df=dof, loc=tf.zeros(n_voxels, dtype=np.float32), scale=tf.linalg.LinearOperatorLowerTriangular(noise))
                noise = mvn.sample((n_batches, n_timepoints))

        print(noise.shape)
        return self._predict(paradigm, parameters, weights) + noise

    def _gradient(self, stimuli, parameters):
        """Compute d(predictions)/d(stimuli) using TF Jacobians."""
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
        """Formatted data matrix (pandas DataFrame)."""
        return self._data

    @data.setter
    def data(self, data):
        """Setter that ensures incoming data is converted to the expected format."""
        if data is None:
            self._data = None
        else:
            self._data = format_data(data)

    @property
    def weights(self):
        """Basis weights used for discrete/basis-function models (DataFrame)."""
        return self._weights

    @weights.setter
    def weights(self, weights):
        """Setter that casts/validates provided weights."""
        self._weights = format_weights(weights)

    def to_discrete_model(self, grid, parameters=None, weights=None):
        """Return a ``DiscreteModel`` evaluated on ``grid`` stimulus coordinates."""

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
        """Log-likelihood of observing ``data`` given stimuli and model parameters."""

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

    def get_stimulus_pdf(self, data, stimulus_range, parameters=None, weights=None, omega=None, dof=None, normalize=True,
                         include_multidimensional_stimulus_index=False):
        """Evaluate posterior over stimuli for each time point given data."""

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
            if include_multidimensional_stimulus_index:
                index = pd.MultiIndex.from_frame(pd.DataFrame(stimulus_range[:, 0, :],
                                                columns=self.stimulus.dimension_labels))
            else:
                index = None

            ll = pd.DataFrame(ll.T, index=time_index, columns=index)

        # Normalize, working from log likelihoods (otherwise we get numerical issues)
        ll = np.exp(ll.apply(lambda d: d-d.max(), 1))
        # ll = ll.apply(lambda d: d/d.sum(), axis=1)

        # ll = np.exp(ll)

        if normalize:
            ll /= np.trapz(ll, ll.columns)[:, np.newaxis]

        return ll

    def apply_mask(self, mask):
        """Subset voxels/weights/parameters according to ``mask`` boolean array."""

        if self.data is not None:
            self.data = self.data.loc[:, mask]

        if self.weights is None:
            if self.parameters is not None:
                self.parameters = self.parameters.loc[mask]
        else:
            self.weights = self.weights.loc[:, mask]

    def get_WWT(self):
        """Return WᵀW — either from stored weights or the cached pseudo matrix."""
        return self.weights.T.dot(self.weights)

    def get_residual_dist(self, n_voxels, omega_chol, dof):
        """Create the residual distribution (Gaussian or Student-t)."""

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
        """TensorFlow helper that computes likelihoods for batches of stimuli."""

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
        """Evaluate log-probabilities for each residual timeseries."""
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
        """Return parameters formatted as DataFrame matching ``parameter_labels``."""

        if (parameters is None) and (self.parameters is not None):
            parameters = self.parameters

        parameters = format_parameters(parameters)

        if parameters is not None:
            parameters = parameters[self.parameter_labels]

        return parameters

    def get_paradigm(self, paradigm):
        """Return the cleaned paradigm DataFrame, falling back to stored one."""
            
        if paradigm is None:
            if self.paradigm is not None:
                return self.paradigm
            else:
                raise ValueError('Please provide paradigm!')

        paradigm = self.stimulus.clean_paradigm(paradigm)

        return paradigm

    def _get_paradigm(self, paradigm):
        """Tensor-ready paradigm (np.array) used inside TF functions."""
            
        if paradigm is None:
            paradigm = self.get_paradigm(paradigm)

        paradigm = self.stimulus._clean_paradigm(paradigm)

        return paradigm


class EncodingRegressionModel(EncodingModel):
    """Encoding model whose parameters are linear combinations of regressors."""

    def __init__(self, paradigm=None, data=None, parameters=None,
                regressors={}, weights=None, omega=None,
                baseline_parameter_values=None,
                 verbosity=logging.INFO, **kwargs):
        """Build Patsy design matrices and tie parameter values to regressors."""

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
        """MultiIndex of (parameter, regressor) pairs used for coefficients."""
        regressor_parameters = []

        for parameter in self.base_parameter_labels:
            regressor_parameters += zip([parameter+'_unbounded'] * design_matrices[parameter].shape[1],
                                        design_matrices[parameter].design_info.column_names)

        return pd.MultiIndex.from_tuples(regressor_parameters, names=['parameter', 'regressor'])

    def _get_base_parameters(self, design_matrices, regressor_parameters):
        """Transform regressor weights back into native parameter space."""

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
        """Create Patsy design matrices for each parameter."""

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
        """Update stored paradigm and rebuild design matrices/regressor labels."""

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
        """Apply regressors to recover base parameters before prediction."""
        base_parameters = self._get_base_parameters(self.design_matrices, parameters)
        result = self._basis_basis_predictions(self.base_paradigm.values[:, np.newaxis, :], base_parameters)
        return tf.reshape(result, [1, result.shape[0], -1])

    def _get_paradigm(self, paradigm):
        """Override to ensure cleaned paradigm is used (regressors already bound)."""

        # if not paradigm.equals(self.paradigm):
        #     raise Exception('For EncodignRegressionModel, the paradigm should be set when the model is initialized OR using set_paradigm().')

        paradigm = self.stimulus._clean_paradigm(paradigm)

        return paradigm

    def get_conditionspecific_parameters(self, conditions, parameters):
        """Evaluate parameter values for specific condition rows."""

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

    def get_stimulus_pdf(self, data, stimulus_range, parameters=None, weights=None, omega=None, dof=None, normalize=True,
                            include_multidimensional_stimulus_index=False):


        # print("Note that non-stimulus dimensions (e.g., the regressors) are part of the likelihood calculation!")

        self.set_paradigm(stimulus_range)

        pred = self.predict(stimulus_range, parameters=parameters, weights=weights)

        # n_predictions x n_timepoints x n_units
        residuals = data.values[np.newaxis, :, :] - pred.values[:, np.newaxis, :]

        omega_chol = np.linalg.cholesky(omega)
        n_units = data.shape[1]

        residual_dist = self.get_residual_dist(n_units, omega_chol, dof)
        # we use log likelihood to correct for very small numbers
        ll = residual_dist.log_prob(residuals).numpy()
        ll = pd.DataFrame(ll, index=pd.MultiIndex.from_frame(stimulus_range), columns=data.index).T
        
        if normalize:
            # Subtract max for numerical stability before exponentiating
            ll = ll.sub(ll.max(axis=1), axis=0)
        
        ll = np.exp(ll)

        return ll
        


class HRFEncodingModel(object):
    """Mixin that equips an encoding model with HRF convolution support.

    Wraps another :class:`EncodingModel` to (optionally) append HRF parameters,
    zero/one-out baseline and amplitude before convolution, and then re-apply
    those parameters after the HRF is applied.  Accepts any :class:`HRFModel`
    implementation and can share or individualize HRFs per voxel.
    """

    def __init__(self, paradigm=None, data=None, parameters=None,
                    weights=None, omega=None, hrf_model=None, verbosity=logging.INFO,
                    flexible_hrf_parameters=False, **kwargs):
        """Wire an ``HRFModel`` into an existing encoding model."""

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
        """Convolve base predictions with HRF, reapplying amplitude/baseline."""

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
        """Bypass HRF convolution; useful for diagnostics/debugging."""
        return EncodingModel._predict(self, paradigm, parameters, weights)

class GaussianPRF(EncodingModel):
    """One-dimensional population receptive field with Gaussian tuning."""

    parameter_labels = ['mu', 'sd', 'amplitude', 'baseline']

    def __init__(self, paradigm=None, data=None, parameters=None,
                 weights=None, omega=None, allow_neg_amplitudes=False, verbosity=logging.INFO,
                 model_stimulus_amplitude=False,
                 **kwargs):
        """Configure Gaussian pRF with optional stimulus amplitude modeling."""

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

class GaussianPRFWithHRF(GaussianPRF, HRFEncodingModel):
    """Combine Gaussian pRF spatial tuning with an explicit HRF convolution."""


class LogGaussianPRFWithHRF(LogGaussianPRF, HRFEncodingModel):
    """Log-Gaussian tuning plus HRF parameters."""


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

        super().__init__(paradigm=paradigm, data=data, parameters=parameters,
                         weights=weights, omega=omega, allow_neg_amplitudes=allow_neg_amplitudes,
                          verbosity=verbosity, model_stimulus_amplitude=model_stimulus_amplitude,
                          **kwargs)

    @tf.function
    def _transform_parameters_forward1(self, parameters):
        return tf.concat([tf.math.softplus(parameters[:, 0][:, tf.newaxis]),
                          tf.math.softplus(parameters[:, 1][:, tf.newaxis]),
                          parameters[:, 2][:, tf.newaxis],
                          parameters[:, 3][:, tf.newaxis],
                          parameters[:, 4][:, tf.newaxis]], axis=1)

    @tf.function
    def _transform_parameters_backward1(self, parameters):
        return tf.concat([tfp.math.softplus_inverse(parameters[:, 0][:, tf.newaxis]),
                          tfp.math.softplus_inverse(
                              parameters[:, 1][:, tf.newaxis]),
                          parameters[:, 2][:, tf.newaxis],
                          parameters[:, 3][:, tf.newaxis],
                          parameters[:, 4][:, tf.newaxis]], axis=1)
    @tf.function
    def _transform_parameters_forward2(self, parameters):
        return tf.concat([tf.math.softplus(parameters[:, 0][:, tf.newaxis]),
                          tf.math.softplus(parameters[:, 1][:, tf.newaxis]),
                          parameters[:, 2][:, tf.newaxis],
                          tf.math.softplus(parameters[:, 3][:, tf.newaxis]),
                          parameters[:, 4][:, tf.newaxis]], axis=1)
    
    @tf.function
    def _transform_parameters_backward2(self, parameters):
        return tf.concat([tfp.math.softplus_inverse(parameters[:, 0][:, tf.newaxis]),
                          tfp.math.softplus_inverse(
                              parameters[:, 1][:, tf.newaxis]),
                          parameters[:, 2][:, tf.newaxis],
                          tfp.math.softplus_inverse(parameters[:, 3][:, tf.newaxis]),
                          parameters[:, 4][:, tf.newaxis]], axis=1)

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

        GaussianPRF2D.__init__(self, grid_coordinates=grid_coordinates, paradigm=paradigm, data=data,
                               parameters=parameters, weights=weights, verbosity=verbosity,
                               positive_image_values_only=positive_image_values_only, **kwargs)
        HRFEncodingModel.__init__(self, hrf_model=hrf_model, flexible_hrf_parameters=flexible_hrf_parameters, **kwargs)

    def to_linear_model(self):
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


class AttentionFieldPRF2D(GaussianPRF2D):
    """Attention-Field-aware 2D Gaussian PRF.

    Implements a Reynolds & Heeger 2009 / Sumiya AF+ -style joint model in
    which each voxel's stimulus-drive PRF (a Gaussian centered at (x, y)
    with size ``sd``) is multiplied at every grid location by an
    attention-field modulation that depends on the *current condition*.

    For ``n_conditions`` ring positions ``ring_positions`` (each (rx, ry))
    and a per-time-point one-hot ``condition_indicator`` (n_timepoints x
    n_conditions) selecting which ring position is the high-probability
    (HP) attended/suppressed location, the modulation field on the
    stimulus grid is

        mod_c(g) = 1 + sign · ( g_HP · A_{H_c}(g)
                              + g_LP · Σ_{ℓ ≠ H_c} A_ℓ(g) )

    where each ``A_ℓ`` is a Gaussian on the grid centered at ring position
    ℓ with shared σ_AF (peak-normalized to 1, so ``g_HP/g_LP`` directly
    parameterize the peak modulation contribution at each ring location).

    The forward pass per timepoint t with condition c(t) is

        prediction_t,v = ∫ paradigm_t(g) · S_v(g) · mod_{c(t)}(g) dg

    With ``mode='attraction'`` the modulation has positive sign (Sumiya
    AF+ analog: voxels are pulled toward the attended locus).
    With ``mode='suppression'`` the sign is flipped, modeling the retsupp
    history-prior result where voxels are pushed *away* from the HP
    distractor location.

    Parameters
    ----------
    grid_coordinates, paradigm, hrf_model, ...
        Same as :class:`GaussianPRF2DWithHRF` (this class is intended to
        be combined with HRF via :class:`AttentionFieldPRF2DWithHRF`).
    condition_indicator : array-like, shape (n_timepoints, n_conditions)
        One-hot encoding of which ring position is the HP at each TR.
        Rows that are all zero (e.g. baseline blocks) are treated as
        having no attention modulation (mod = 1).
    ring_positions : array-like, shape (n_conditions, 2)
        Cartesian (x, y) positions of the four distractor ring locations.
    mode : {'suppression', 'attraction'}
        Sign of the modulation (default 'suppression', matching retsupp).

    Per-voxel parameters
    --------------------
    ``x``, ``y``, ``sd``, ``baseline``, ``amplitude``  — standard PRF.

    Shared (across all voxels) parameters
    -------------------------------------
    ``sigma_AF`` : positive
        Width of every attention-field Gaussian (shared across all four
        ring positions; the user can set ``free_per_condition_g=False``
        and use ``ParameterFitter(..., shared_pars=['sigma_AF', 'g_HP',
        'g_LP'])`` to enforce sharing during fitting).
    ``g_HP`` : positive
        Modulation amplitude at the HP location.
    ``g_LP`` : positive
        Modulation amplitude at each of the 3 LP (non-HP) ring positions.

    Notes
    -----
    All four ring positions are present in every condition, only their
    *amplitudes* differ between HP (``g_HP``) and LP (``g_LP``). This is
    the same model form as the existing ``af_model.fit_four_af_competing``
    in retsupp, but evaluated jointly at the BOLD signal level rather
    than post-hoc on conditionwise PRF parameters.
    """

    parameter_labels = ['x', 'y', 'sd', 'baseline', 'amplitude',
                        'sigma_AF', 'g_HP', 'g_LP']

    def __init__(self, grid_coordinates=None, paradigm=None, data=None,
                 parameters=None, condition_indicator=None,
                 ring_positions=None, mode='suppression',
                 weights=None, omega=None,
                 positive_image_values_only=True,
                 verbosity=logging.INFO, **kwargs):
        if condition_indicator is None:
            raise ValueError(
                "AttentionFieldPRF2D requires a `condition_indicator` "
                "array of shape (n_timepoints, n_conditions).")
        if ring_positions is None:
            raise ValueError(
                "AttentionFieldPRF2D requires `ring_positions` of shape "
                "(n_conditions, 2).")

        if mode not in ('suppression', 'attraction', 'signed'):
            raise ValueError(
                f"mode must be 'suppression', 'attraction', or 'signed', "
                f"got {mode!r}")
        self.mode = mode
        # 'signed' lets the optimizer pick attraction or suppression freely
        # via the SIGN of g_HP / g_LP (gains are no longer softplus-bounded).
        # We absorb the overall sign into the gains, so set _sign = +1.
        self._sign = -1.0 if mode == 'suppression' else +1.0
        self._signed_gains = (mode == 'signed')

        self.condition_indicator = np.asarray(condition_indicator, dtype=np.float32)
        self.ring_positions = np.asarray(ring_positions, dtype=np.float32)
        self.n_conditions = self.ring_positions.shape[0]

        # Per condition c, ring index ℓ:  is_hp_per_cond_ring[c, ℓ] == 1 iff ℓ == c
        # (we assume the c-th condition's HP is the c-th ring position; rearrange
        # rows of `condition_indicator` accordingly when constructing it).
        self._is_hp = np.eye(self.n_conditions, dtype=np.float32)

        super().__init__(
            grid_coordinates=grid_coordinates, paradigm=paradigm, data=data,
            parameters=parameters, weights=weights, omega=omega,
            positive_image_values_only=positive_image_values_only,
            verbosity=verbosity, **kwargs)

        # Cache as TF constants for speed.
        self._tf_condition_indicator = tf.constant(self.condition_indicator,
                                                   dtype=tf.float32)
        self._tf_ring_positions = tf.constant(self.ring_positions, dtype=tf.float32)
        self._tf_is_hp = tf.constant(self._is_hp, dtype=tf.float32)
        self._tf_sign = tf.constant(self._sign, dtype=tf.float32)

    @tf.function
    def _attention_modulation(self, parameters):
        """Compute the per-condition modulation field on the stimulus grid.

        Returns
        -------
        mod : tf.Tensor, shape (n_batches, n_voxels, n_conditions, n_grid)
            Modulation factor at each grid location for each condition.
            Even though this depends on `g_HP`, `g_LP`, `sigma_AF` only —
            which are shared across voxels in our intended fitting setup
            — we keep it per-voxel-per-batch to support fully-flexible
            fits where these can vary per voxel.
        """
        # Grid: (n_grid, 2).
        gx = self._grid_coordinates[:, 0][tf.newaxis, tf.newaxis, tf.newaxis, :]  # (1,1,1,G)
        gy = self._grid_coordinates[:, 1][tf.newaxis, tf.newaxis, tf.newaxis, :]

        sigma_AF = parameters[:, :, 5, tf.newaxis, tf.newaxis]   # (B,V,1,1)
        g_HP = parameters[:, :, 6, tf.newaxis, tf.newaxis]
        g_LP = parameters[:, :, 7, tf.newaxis, tf.newaxis]

        # Ring positions: (n_conditions, 2)  ->  (1,1,n_C,1)
        rx = self._tf_ring_positions[:, 0][tf.newaxis, tf.newaxis, :, tf.newaxis]
        ry = self._tf_ring_positions[:, 1][tf.newaxis, tf.newaxis, :, tf.newaxis]

        # Per-ring Gaussian on the grid (peak-normalized to 1).
        # Shape: (1, 1, n_conditions, n_grid)  after broadcasting with sigma_AF.
        # Note: A_ℓ depends on sigma_AF (B,V,1,1) so result is (B,V,n_C,n_grid).
        diff_sq = (gx - rx) ** 2 + (gy - ry) ** 2
        A = tf.exp(-diff_sq / (2.0 * sigma_AF ** 2))  # (B,V,n_C_ring,n_grid)

        # For each condition C in n_conditions:
        #   mod_C(g) = 1 + sign * ( g_HP * A_{H_C}(g) + g_LP * Σ_{ℓ ≠ H_C} A_ℓ(g) )
        # is_hp[C, ℓ]: 1 if ℓ is HP for condition C, 0 otherwise.
        # Build a per-condition weight: w[C, ℓ] = g_HP if HP else g_LP.
        # Then mod_C(g) = 1 + sign * Σ_ℓ w[C, ℓ] · A_ℓ(g).
        # Shape: (B,V,n_C_cond,n_C_ring)
        is_hp = self._tf_is_hp[tf.newaxis, tf.newaxis, :, :]
        w = is_hp * g_HP + (1.0 - is_hp) * g_LP

        # Σ_ℓ w[C, ℓ] · A_ℓ(g): einsum over ring index.
        # A: (B,V,L,G), w: (B,V,C,L)  ->  (B,V,C,G)
        modulation_sum = tf.einsum('bvcl,bvlg->bvcg', w, A)
        mod = 1.0 + self._tf_sign * modulation_sum  # (B,V,n_C,n_G)
        # Clamp to be non-negative (predictions of suppressed bar should
        # not flip sign; pure attraction can never go below 0 anyway).
        mod = tf.maximum(mod, 0.0)
        return mod

    @tf.function
    def _basis_predictions(self, paradigm, parameters):
        # paradigm: n_batches x n_timepoints x n_stimulus_features (n_grid)
        # parameters: n_batches x n_voxels x n_parameters

        # Per-voxel SD-pRF on the grid: (B, V, G).
        rf = self._get_rf(self.grid_coordinates, parameters)

        # Per-condition modulation field on the grid: (B, V, n_C, G).
        mod = self._attention_modulation(parameters)

        # Effective per-condition RF: (B, V, n_C, G).
        # rf: (B, V, G).
        eff_rf_per_cond = rf[:, :, tf.newaxis, :] * mod

        # Per-condition predictions on paradigm.
        # paradigm: (B, T, G); eff_rf_per_cond: (B, V, n_C, G).
        # partial[B, T, V, C] = Σ_g paradigm[B, T, g] · eff_rf_per_cond[B, V, C, g]
        partial = tf.einsum('btg,bvcg->btvc', paradigm, eff_rf_per_cond)

        # Per-time-point selection of condition: (T, n_C).
        # result[B, T, V] = Σ_c condition_indicator[T, c] · partial[B, T, V, c]
        ci = self._tf_condition_indicator       # (T, n_C)
        result = tf.einsum('tc,btvc->btv', ci, partial)

        baseline = parameters[:, tf.newaxis, :, 3]
        result = result + baseline

        return result

    @tf.function
    def _transform_parameters_forward(self, parameters):
        # Standard PRF: x, y, softplus(sd), baseline, amplitude.
        # AF: softplus(sigma_AF), then for g_HP / g_LP either softplus
        # (positive-only, attraction or suppression mode) or identity
        # (signed mode — gains may be negative).
        if self._signed_gains:
            g_hp = parameters[:, 6][:, tf.newaxis]
            g_lp = parameters[:, 7][:, tf.newaxis]
        else:
            g_hp = tf.math.softplus(parameters[:, 6][:, tf.newaxis])
            g_lp = tf.math.softplus(parameters[:, 7][:, tf.newaxis])
        return tf.concat([
            parameters[:, 0][:, tf.newaxis],                              # x
            parameters[:, 1][:, tf.newaxis],                              # y
            tf.math.softplus(parameters[:, 2][:, tf.newaxis]),            # sd
            parameters[:, 3][:, tf.newaxis],                              # baseline
            parameters[:, 4][:, tf.newaxis],                              # amplitude
            tf.math.softplus(parameters[:, 5][:, tf.newaxis]),            # sigma_AF
            g_hp,                                                         # g_HP
            g_lp,                                                         # g_LP
        ], axis=1)

    @tf.function
    def _transform_parameters_backward(self, parameters):
        if self._signed_gains:
            g_hp_unb = parameters[:, 6][:, tf.newaxis]
            g_lp_unb = parameters[:, 7][:, tf.newaxis]
        else:
            g_hp_unb = tfp.math.softplus_inverse(
                parameters[:, 6][:, tf.newaxis])
            g_lp_unb = tfp.math.softplus_inverse(
                parameters[:, 7][:, tf.newaxis])
        return tf.concat([
            parameters[:, 0][:, tf.newaxis],
            parameters[:, 1][:, tf.newaxis],
            tfp.math.softplus_inverse(parameters[:, 2][:, tf.newaxis]),
            parameters[:, 3][:, tf.newaxis],
            parameters[:, 4][:, tf.newaxis],
            tfp.math.softplus_inverse(parameters[:, 5][:, tf.newaxis]),
            g_hp_unb,
            g_lp_unb,
        ], axis=1)


class AttentionFieldPRF2DWithHRF(HRFEncodingModel, AttentionFieldPRF2D):
    """HRF-convolved version of :class:`AttentionFieldPRF2D`.

    Use this for fitting to BOLD time-courses. The set of free parameters
    is::

        ['x', 'y', 'sd', 'baseline', 'amplitude',
         'sigma_AF', 'g_HP', 'g_LP'] (+ HRF parameters if flexible)

    During joint AF + PRF fitting, pass
    ``shared_pars=['sigma_AF', 'g_HP', 'g_LP']`` to the
    :class:`braincoder.optimize.ParameterFitter`. ``x``, ``y``, ``sd``,
    ``baseline``, ``amplitude`` remain per-voxel.
    """

    def __init__(self, grid_coordinates=None, paradigm=None, data=None,
                 parameters=None, condition_indicator=None,
                 ring_positions=None, mode='suppression',
                 positive_image_values_only=True,
                 weights=None, hrf_model=None,
                 flexible_hrf_parameters=False,
                 verbosity=logging.INFO, **kwargs):

        AttentionFieldPRF2D.__init__(
            self, grid_coordinates=grid_coordinates, paradigm=paradigm,
            data=data, parameters=parameters,
            condition_indicator=condition_indicator,
            ring_positions=ring_positions, mode=mode,
            weights=weights, verbosity=verbosity,
            positive_image_values_only=positive_image_values_only, **kwargs)

        HRFEncodingModel.__init__(self, hrf_model=hrf_model,
                                  flexible_hrf_parameters=flexible_hrf_parameters,
                                  **kwargs)

    @tf.function
    def _transform_parameters_forward(self, parameters):
        if self.flexible_hrf_parameters:
            n_hrf_pars = len(self.hrf_model.parameter_labels)
            encoding_pars = AttentionFieldPRF2D._transform_parameters_forward(
                self, parameters[:, :-n_hrf_pars])
            hrf_pars = self.hrf_model._transform_parameters_forward(
                parameters[:, -n_hrf_pars:])
            return tf.concat([encoding_pars, hrf_pars], axis=1)
        else:
            return AttentionFieldPRF2D._transform_parameters_forward(self, parameters)

    @tf.function
    def _transform_parameters_backward(self, parameters):
        if self.flexible_hrf_parameters:
            n_hrf_pars = len(self.hrf_model.parameter_labels)
            encoding_pars = AttentionFieldPRF2D._transform_parameters_backward(
                self, parameters[:, :-n_hrf_pars])
            hrf_pars = self.hrf_model._transform_parameters_backward(
                parameters[:, -n_hrf_pars:])
            return tf.concat([encoding_pars, hrf_pars], axis=1)
        else:
            return AttentionFieldPRF2D._transform_parameters_backward(self, parameters)


class DoGAttentionFieldPRF2D(DifferenceOfGaussiansPRF2D):
    """Attention-Field-aware Difference-of-Gaussians 2D PRF.

    Same forward-modulation logic as :class:`AttentionFieldPRF2D`, but the
    per-voxel stimulus-drive receptive field is a Difference-of-Gaussians
    (centre + surround), not a single Gaussian. This is the apples-to-apples
    counterpart to the conditionwise DoG fits in
    ``derivatives/prf_conditionfit/model4`` — the receptive field shape
    matches model 4 exactly, with the AF parameters layered on top.

    Per-voxel parameters (7)
    ------------------------
    ``x``, ``y``, ``sd``, ``baseline``, ``amplitude``,
    ``srf_amplitude``, ``srf_size`` — same as
    :class:`DifferenceOfGaussiansPRF2D`.

    Shared (across all voxels) parameters (3)
    -----------------------------------------
    ``sigma_AF``, ``g_HP``, ``g_LP`` — same as
    :class:`AttentionFieldPRF2D`.

    Total: 10 parameters per voxel (7 per-voxel + 3 shared).

    See :class:`AttentionFieldPRF2D` for the full description of the AF
    modulation scheme and the ``mode``/``ring_positions``/
    ``condition_indicator`` arguments.
    """

    parameter_labels = ['x', 'y', 'sd', 'baseline', 'amplitude',
                        'srf_amplitude', 'srf_size',
                        'sigma_AF', 'g_HP', 'g_LP']

    def __init__(self, grid_coordinates=None, paradigm=None, data=None,
                 parameters=None, condition_indicator=None,
                 ring_positions=None, mode='suppression',
                 weights=None, omega=None,
                 positive_image_values_only=True,
                 verbosity=logging.INFO, **kwargs):
        if condition_indicator is None:
            raise ValueError(
                "DoGAttentionFieldPRF2D requires a `condition_indicator` "
                "array of shape (n_timepoints, n_conditions).")
        if ring_positions is None:
            raise ValueError(
                "DoGAttentionFieldPRF2D requires `ring_positions` of shape "
                "(n_conditions, 2).")

        if mode not in ('suppression', 'attraction', 'signed'):
            raise ValueError(
                f"mode must be 'suppression', 'attraction', or 'signed', "
                f"got {mode!r}")
        self.mode = mode
        self._sign = -1.0 if mode == 'suppression' else +1.0
        self._signed_gains = (mode == 'signed')

        self.condition_indicator = np.asarray(condition_indicator,
                                              dtype=np.float32)
        self.ring_positions = np.asarray(ring_positions, dtype=np.float32)
        self.n_conditions = self.ring_positions.shape[0]

        # is_hp[c, ℓ] == 1 iff ring index ℓ is the HP for condition c
        # (we assume condition c's HP is the c-th ring position).
        self._is_hp = np.eye(self.n_conditions, dtype=np.float32)

        # We inherit from DifferenceOfGaussiansPRF2D (not AttentionFieldPRF2D)
        # so the inherited ``_get_rf`` is the DoG kernel. The AF prep above
        # mirrors AttentionFieldPRF2D.__init__; the rest of the setup
        # (grid coordinates, paradigm cleaning, ...) is delegated to the
        # DoG parent ctor, which ultimately calls GaussianPRF2D.__init__.
        DifferenceOfGaussiansPRF2D.__init__(
            self,
            grid_coordinates=grid_coordinates, paradigm=paradigm, data=data,
            parameters=parameters, weights=weights, omega=omega,
            positive_image_values_only=positive_image_values_only,
            verbosity=verbosity, **kwargs)

        # Cache as TF constants for speed.
        self._tf_condition_indicator = tf.constant(self.condition_indicator,
                                                   dtype=tf.float32)
        self._tf_ring_positions = tf.constant(self.ring_positions,
                                              dtype=tf.float32)
        self._tf_is_hp = tf.constant(self._is_hp, dtype=tf.float32)
        self._tf_sign = tf.constant(self._sign, dtype=tf.float32)

    @tf.function
    def _attention_modulation(self, parameters):
        """Per-condition AF modulation field on the stimulus grid.

        Identical formula to :meth:`AttentionFieldPRF2D._attention_modulation`,
        but the AF parameter indices shift by 2 because the DoG voxel
        kernel inserts ``srf_amplitude`` and ``srf_size`` ahead of them.

        Returns
        -------
        mod : tf.Tensor, shape (n_batches, n_voxels, n_conditions, n_grid)
        """
        # Grid: (n_grid, 2).
        gx = self._grid_coordinates[:, 0][tf.newaxis, tf.newaxis, tf.newaxis, :]
        gy = self._grid_coordinates[:, 1][tf.newaxis, tf.newaxis, tf.newaxis, :]

        # Indices shift by 2 vs AttentionFieldPRF2D: 7=sigma_AF, 8=g_HP, 9=g_LP.
        sigma_AF = parameters[:, :, 7, tf.newaxis, tf.newaxis]   # (B,V,1,1)
        g_HP = parameters[:, :, 8, tf.newaxis, tf.newaxis]
        g_LP = parameters[:, :, 9, tf.newaxis, tf.newaxis]

        rx = self._tf_ring_positions[:, 0][tf.newaxis, tf.newaxis, :, tf.newaxis]
        ry = self._tf_ring_positions[:, 1][tf.newaxis, tf.newaxis, :, tf.newaxis]

        diff_sq = (gx - rx) ** 2 + (gy - ry) ** 2
        A = tf.exp(-diff_sq / (2.0 * sigma_AF ** 2))   # (B,V,n_C_ring,n_grid)

        is_hp = self._tf_is_hp[tf.newaxis, tf.newaxis, :, :]
        w = is_hp * g_HP + (1.0 - is_hp) * g_LP

        modulation_sum = tf.einsum('bvcl,bvlg->bvcg', w, A)
        mod = 1.0 + self._tf_sign * modulation_sum     # (B,V,n_C,n_G)
        mod = tf.maximum(mod, 0.0)
        return mod

    @tf.function
    def _basis_predictions(self, paradigm, parameters):
        # paradigm: (B, T, G)
        # parameters: (B, V, n_parameters=10)

        # Per-voxel DoG receptive field on the grid: (B, V, G).
        # self._get_rf is inherited from DifferenceOfGaussiansPRF2D.
        # That method only reads parameters[:, :, 0..6] (x, y, sd,
        # baseline, amplitude, srf_amplitude, srf_size) so the extra AF
        # params at the tail are harmless.
        rf = self._get_rf(self.grid_coordinates, parameters)

        # Per-condition AF modulation field on the grid: (B, V, n_C, G).
        mod = self._attention_modulation(parameters)

        # Effective per-condition RF: (B, V, n_C, G).
        eff_rf_per_cond = rf[:, :, tf.newaxis, :] * mod

        # Per-condition predictions on the paradigm: (B, T, V, n_C).
        partial = tf.einsum('btg,bvcg->btvc', paradigm, eff_rf_per_cond)

        # Select condition per timepoint via condition_indicator.
        ci = self._tf_condition_indicator       # (T, n_C)
        result = tf.einsum('tc,btvc->btv', ci, partial)

        baseline = parameters[:, tf.newaxis, :, 3]
        result = result + baseline

        return result

    @tf.function
    def _transform_parameters_forward(self, parameters):
        # First 7: DoG transforms (identity, identity, softplus, identity,
        # identity, softplus, softplus). Then AF: softplus(sigma_AF) and
        # gain transforms (softplus or identity per signed_gains).
        if self._signed_gains:
            g_hp = parameters[:, 8][:, tf.newaxis]
            g_lp = parameters[:, 9][:, tf.newaxis]
        else:
            g_hp = tf.math.softplus(parameters[:, 8][:, tf.newaxis])
            g_lp = tf.math.softplus(parameters[:, 9][:, tf.newaxis])
        return tf.concat([
            parameters[:, 0][:, tf.newaxis],                              # x
            parameters[:, 1][:, tf.newaxis],                              # y
            tf.math.softplus(parameters[:, 2][:, tf.newaxis]),            # sd
            parameters[:, 3][:, tf.newaxis],                              # baseline
            parameters[:, 4][:, tf.newaxis],                              # amplitude
            tf.math.softplus(parameters[:, 5][:, tf.newaxis]),            # srf_amplitude
            tf.math.softplus(parameters[:, 6][:, tf.newaxis]),            # srf_size
            tf.math.softplus(parameters[:, 7][:, tf.newaxis]),            # sigma_AF
            g_hp,                                                         # g_HP
            g_lp,                                                         # g_LP
        ], axis=1)

    @tf.function
    def _transform_parameters_backward(self, parameters):
        if self._signed_gains:
            g_hp_unb = parameters[:, 8][:, tf.newaxis]
            g_lp_unb = parameters[:, 9][:, tf.newaxis]
        else:
            g_hp_unb = tfp.math.softplus_inverse(
                parameters[:, 8][:, tf.newaxis])
            g_lp_unb = tfp.math.softplus_inverse(
                parameters[:, 9][:, tf.newaxis])
        return tf.concat([
            parameters[:, 0][:, tf.newaxis],
            parameters[:, 1][:, tf.newaxis],
            tfp.math.softplus_inverse(parameters[:, 2][:, tf.newaxis]),
            parameters[:, 3][:, tf.newaxis],
            parameters[:, 4][:, tf.newaxis],
            tfp.math.softplus_inverse(parameters[:, 5][:, tf.newaxis]),
            tfp.math.softplus_inverse(parameters[:, 6][:, tf.newaxis]),
            tfp.math.softplus_inverse(parameters[:, 7][:, tf.newaxis]),
            g_hp_unb,
            g_lp_unb,
        ], axis=1)


class DoGAttentionFieldPRF2DWithHRF(HRFEncodingModel, DoGAttentionFieldPRF2D):
    """HRF-convolved version of :class:`DoGAttentionFieldPRF2D`.

    Free parameters::

        ['x', 'y', 'sd', 'baseline', 'amplitude',
         'srf_amplitude', 'srf_size',
         'sigma_AF', 'g_HP', 'g_LP'] (+ HRF parameters if flexible)

    During joint AF + DoG-PRF fitting, pass
    ``shared_pars=['sigma_AF', 'g_HP', 'g_LP']`` to the
    :class:`braincoder.optimize.ParameterFitter`. The 7 per-voxel DoG
    parameters remain per-voxel.
    """

    def __init__(self, grid_coordinates=None, paradigm=None, data=None,
                 parameters=None, condition_indicator=None,
                 ring_positions=None, mode='suppression',
                 positive_image_values_only=True,
                 weights=None, hrf_model=None,
                 flexible_hrf_parameters=False,
                 verbosity=logging.INFO, **kwargs):

        DoGAttentionFieldPRF2D.__init__(
            self, grid_coordinates=grid_coordinates, paradigm=paradigm,
            data=data, parameters=parameters,
            condition_indicator=condition_indicator,
            ring_positions=ring_positions, mode=mode,
            weights=weights, verbosity=verbosity,
            positive_image_values_only=positive_image_values_only, **kwargs)

        HRFEncodingModel.__init__(self, hrf_model=hrf_model,
                                  flexible_hrf_parameters=flexible_hrf_parameters,
                                  **kwargs)

    @tf.function
    def _transform_parameters_forward(self, parameters):
        if self.flexible_hrf_parameters:
            n_hrf_pars = len(self.hrf_model.parameter_labels)
            encoding_pars = DoGAttentionFieldPRF2D._transform_parameters_forward(
                self, parameters[:, :-n_hrf_pars])
            hrf_pars = self.hrf_model._transform_parameters_forward(
                parameters[:, -n_hrf_pars:])
            return tf.concat([encoding_pars, hrf_pars], axis=1)
        else:
            return DoGAttentionFieldPRF2D._transform_parameters_forward(
                self, parameters)

    @tf.function
    def _transform_parameters_backward(self, parameters):
        if self.flexible_hrf_parameters:
            n_hrf_pars = len(self.hrf_model.parameter_labels)
            encoding_pars = DoGAttentionFieldPRF2D._transform_parameters_backward(
                self, parameters[:, :-n_hrf_pars])
            hrf_pars = self.hrf_model._transform_parameters_backward(
                parameters[:, -n_hrf_pars:])
            return tf.concat([encoding_pars, hrf_pars], axis=1)
        else:
            return DoGAttentionFieldPRF2D._transform_parameters_backward(
                self, parameters)


class DoGDynamicAttentionFieldPRF2D_v2(DoGAttentionFieldPRF2D):
    """Dynamic Attention-Field-aware DoG-PRF (v2: shared σ, split dyn gain).

    DoG-voxel-kernel counterpart to
    :class:`DynamicAttentionFieldPRF2D_v2`. The per-voxel stimulus-drive
    receptive field is a Difference-of-Gaussians (centre + surround), and
    the AF modulation is the same as v2: a single shared ``sigma_AF`` for
    both the sustained and dynamic Gaussians, with the per-TR phasic gain
    split into ``g_HP_dyn`` and ``g_LP_dyn``.

    Per-voxel parameters (7)
    ------------------------
    ``x``, ``y``, ``sd``, ``baseline``, ``amplitude``,
    ``srf_amplitude``, ``srf_size`` — same as
    :class:`DifferenceOfGaussiansPRF2D`.

    Shared (across all voxels) parameters (5)
    -----------------------------------------
    ``sigma_AF``, ``g_HP``, ``g_LP``, ``g_HP_dyn``, ``g_LP_dyn``.

    Total: 12 parameters per voxel (7 per-voxel + 5 shared).

    See :class:`DynamicAttentionFieldPRF2D_v2` for the modulation
    formula. Indices below shift by +2 versus the Gaussian v2 because
    ``srf_amplitude`` and ``srf_size`` sit at positions 5, 6 between
    ``amplitude`` and ``sigma_AF``.
    """

    parameter_labels = ['x', 'y', 'sd', 'baseline', 'amplitude',
                        'srf_amplitude', 'srf_size',
                        'sigma_AF', 'g_HP', 'g_LP',
                        'g_HP_dyn', 'g_LP_dyn']

    def __init__(self, grid_coordinates=None, paradigm=None, data=None,
                 parameters=None, condition_indicator=None,
                 dynamic_indicator=None,
                 ring_positions=None, mode='suppression',
                 weights=None, omega=None,
                 positive_image_values_only=True,
                 verbosity=logging.INFO, **kwargs):
        if dynamic_indicator is None:
            raise ValueError(
                "DoGDynamicAttentionFieldPRF2D_v2 requires a "
                "`dynamic_indicator` array of shape "
                "(n_timepoints, n_ring_positions).")

        super().__init__(
            grid_coordinates=grid_coordinates, paradigm=paradigm, data=data,
            parameters=parameters, condition_indicator=condition_indicator,
            ring_positions=ring_positions, mode=mode,
            weights=weights, omega=omega,
            positive_image_values_only=positive_image_values_only,
            verbosity=verbosity, **kwargs)

        self.dynamic_indicator = np.asarray(dynamic_indicator,
                                            dtype=np.float32)
        if self.dynamic_indicator.shape[1] != self.n_conditions:
            raise ValueError(
                f"dynamic_indicator has {self.dynamic_indicator.shape[1]} "
                f"channels but ring_positions has {self.n_conditions}; "
                "channels must align with ring_positions.")
        self._tf_dynamic_indicator = tf.constant(self.dynamic_indicator,
                                                 dtype=tf.float32)

    @tf.function
    def _attention_modulation_dynamic_v2(self, parameters):
        """Per-TR dynamic-AF modulation field on the stimulus grid (v2, DoG).

        Identical formula to
        :meth:`DynamicAttentionFieldPRF2D_v2._attention_modulation_dynamic_v2`,
        but parameter indices shift by +2:
            sigma_AF -> 7,  g_HP_dyn -> 10,  g_LP_dyn -> 11.
        """
        # Take shared parameters from the first batch / first voxel.
        sigma_AF = parameters[0, 0, 7]                     # scalar
        g_HP_dyn = parameters[0, 0, 10]                    # scalar
        g_LP_dyn = parameters[0, 0, 11]                    # scalar

        # Grid: (n_grid, 2).
        gx = self._grid_coordinates[:, 0][tf.newaxis, :]   # (1, G)
        gy = self._grid_coordinates[:, 1][tf.newaxis, :]

        # Ring positions: (n_C, 2)  ->  (n_C, 1).
        rx = self._tf_ring_positions[:, 0][:, tf.newaxis]
        ry = self._tf_ring_positions[:, 1][:, tf.newaxis]

        # Per-ring AF Gaussian (peak-normalized to 1): (n_C, G).
        # Uses sigma_AF, NOT a separate sigma_dyn.
        diff_sq = (gx - rx) ** 2 + (gy - ry) ** 2
        A = tf.exp(-diff_sq / (2.0 * sigma_AF ** 2))       # (n_C, G)

        # Per-TR per-ring "is HP" mask: (T, n_C).
        is_hp_per_tr = self._tf_condition_indicator        # (T, n_C)

        # d_ℓ(t): (T, n_C). Per-TR per-ring distractor on-fraction.
        d = self._tf_dynamic_indicator                     # (T, n_C)

        # Split into HP-dyn and LP-dyn weights per (t, ℓ).
        w_hp = d * is_hp_per_tr                            # (T, n_C)
        w_lp = d * (1.0 - is_hp_per_tr)                    # (T, n_C)

        # Σ_ℓ w_hp[t, ℓ] · A_ℓ(g) and Σ_ℓ w_lp[t, ℓ] · A_ℓ(g).
        field_hp = tf.einsum('tl,lg->tg', w_hp, A)
        field_lp = tf.einsum('tl,lg->tg', w_lp, A)

        return g_HP_dyn * field_hp + g_LP_dyn * field_lp   # (T, G)

    @tf.function
    def _basis_predictions(self, paradigm, parameters):
        # paradigm: (B, T, G)
        # parameters: (B, V, n_parameters=12)

        # Per-voxel DoG receptive field on the grid: (B, V, G).
        # Inherited from DifferenceOfGaussiansPRF2D — reads parameters
        # 0..6 (x, y, sd, baseline, amplitude, srf_amplitude, srf_size).
        rf = self._get_rf(self.grid_coordinates, parameters)

        # Sustained per-condition AF modulation field on the grid:
        # (B, V, n_C, G). Parent (DoGAttentionFieldPRF2D) reads
        # sigma_AF/g_HP/g_LP at indices 7, 8, 9.
        mod_sustained = self._attention_modulation(parameters)

        # Effective per-condition RF (sustained part): (B, V, n_C, G).
        eff_rf_per_cond = rf[:, :, tf.newaxis, :] * mod_sustained

        # Sustained partial: (B, T, V) via condition_indicator selection.
        partial = tf.einsum('btg,bvcg->btvc', paradigm, eff_rf_per_cond)
        ci = self._tf_condition_indicator       # (T, n_C)
        sustained = tf.einsum('tc,btvc->btv', ci, partial)

        # Dynamic per-TR modulation: (T, G), HP/LP split, shared sigma_AF.
        mod_dyn = self._attention_modulation_dynamic_v2(parameters)

        # Dynamic partial: (B, T, V).
        sign = self._tf_sign
        eff_paradigm_dyn = paradigm * mod_dyn[tf.newaxis, :, :]   # (B, T, G)
        dynamic = sign * tf.einsum('btg,bvg->btv', eff_paradigm_dyn, rf)

        result = sustained + dynamic

        # Add baseline (parent _basis_predictions added it but we don't
        # call it).
        baseline = parameters[:, tf.newaxis, :, 3]
        result = result + baseline

        return result

    @tf.function
    def _transform_parameters_forward(self, parameters):
        # First 10: DoG + sustained-AF transforms (delegate to parent).
        # Then add sign-aware g_HP_dyn / g_LP_dyn (no sigma_dyn in v2).
        base = DoGAttentionFieldPRF2D._transform_parameters_forward(
            self, parameters[:, :10])
        if self._signed_gains:
            g_hp_dyn = parameters[:, 10][:, tf.newaxis]
            g_lp_dyn = parameters[:, 11][:, tf.newaxis]
        else:
            g_hp_dyn = tf.math.softplus(parameters[:, 10][:, tf.newaxis])
            g_lp_dyn = tf.math.softplus(parameters[:, 11][:, tf.newaxis])
        return tf.concat([base, g_hp_dyn, g_lp_dyn], axis=1)

    @tf.function
    def _transform_parameters_backward(self, parameters):
        base = DoGAttentionFieldPRF2D._transform_parameters_backward(
            self, parameters[:, :10])
        if self._signed_gains:
            g_hp_dyn_unb = parameters[:, 10][:, tf.newaxis]
            g_lp_dyn_unb = parameters[:, 11][:, tf.newaxis]
        else:
            g_hp_dyn_unb = tfp.math.softplus_inverse(
                parameters[:, 10][:, tf.newaxis])
            g_lp_dyn_unb = tfp.math.softplus_inverse(
                parameters[:, 11][:, tf.newaxis])
        return tf.concat([base, g_hp_dyn_unb, g_lp_dyn_unb], axis=1)


class DoGDynamicAttentionFieldPRF2DWithHRF_v2(HRFEncodingModel,
                                              DoGDynamicAttentionFieldPRF2D_v2):
    """HRF-convolved version of :class:`DoGDynamicAttentionFieldPRF2D_v2`.

    Free parameters::

        ['x', 'y', 'sd', 'baseline', 'amplitude',
         'srf_amplitude', 'srf_size',
         'sigma_AF', 'g_HP', 'g_LP', 'g_HP_dyn', 'g_LP_dyn']
        (+ HRF parameters if flexible)

    During joint AF + DoG-PRF fitting, pass

        shared_pars=['sigma_AF', 'g_HP', 'g_LP', 'g_HP_dyn', 'g_LP_dyn']

    to the :class:`braincoder.optimize.ParameterFitter`. The 7 per-voxel
    DoG parameters remain per-voxel.
    """

    def __init__(self, grid_coordinates=None, paradigm=None, data=None,
                 parameters=None, condition_indicator=None,
                 dynamic_indicator=None,
                 ring_positions=None, mode='suppression',
                 positive_image_values_only=True,
                 weights=None, hrf_model=None,
                 flexible_hrf_parameters=False,
                 verbosity=logging.INFO, **kwargs):

        DoGDynamicAttentionFieldPRF2D_v2.__init__(
            self, grid_coordinates=grid_coordinates, paradigm=paradigm,
            data=data, parameters=parameters,
            condition_indicator=condition_indicator,
            dynamic_indicator=dynamic_indicator,
            ring_positions=ring_positions, mode=mode,
            weights=weights, verbosity=verbosity,
            positive_image_values_only=positive_image_values_only, **kwargs)

        HRFEncodingModel.__init__(self, hrf_model=hrf_model,
                                  flexible_hrf_parameters=flexible_hrf_parameters,
                                  **kwargs)

    @tf.function
    def _transform_parameters_forward(self, parameters):
        if self.flexible_hrf_parameters:
            n_hrf_pars = len(self.hrf_model.parameter_labels)
            encoding_pars = DoGDynamicAttentionFieldPRF2D_v2._transform_parameters_forward(
                self, parameters[:, :-n_hrf_pars])
            hrf_pars = self.hrf_model._transform_parameters_forward(
                parameters[:, -n_hrf_pars:])
            return tf.concat([encoding_pars, hrf_pars], axis=1)
        else:
            return DoGDynamicAttentionFieldPRF2D_v2._transform_parameters_forward(
                self, parameters)

    @tf.function
    def _transform_parameters_backward(self, parameters):
        if self.flexible_hrf_parameters:
            n_hrf_pars = len(self.hrf_model.parameter_labels)
            encoding_pars = DoGDynamicAttentionFieldPRF2D_v2._transform_parameters_backward(
                self, parameters[:, :-n_hrf_pars])
            hrf_pars = self.hrf_model._transform_parameters_backward(
                parameters[:, -n_hrf_pars:])
            return tf.concat([encoding_pars, hrf_pars], axis=1)
        else:
            return DoGDynamicAttentionFieldPRF2D_v2._transform_parameters_backward(
                self, parameters)


class DoGDynamicAttentionFieldPRF2D_v3(DoGAttentionFieldPRF2D):
    """Dynamic Attention-Field-aware DoG-PRF (v3: separate σ_dyn + split gain).

    DoG-voxel-kernel counterpart to
    :class:`DynamicAttentionFieldPRF2D_v3`. The per-voxel stimulus-drive
    receptive field is a Difference-of-Gaussians, and the AF modulation
    is the same as v3: an INDEPENDENT ``sigma_dyn`` for the dynamic
    Gaussian (separate from sustained ``sigma_AF``) plus the HP/LP split
    on the per-TR phasic gain.

    Per-voxel parameters (7)
    ------------------------
    ``x``, ``y``, ``sd``, ``baseline``, ``amplitude``,
    ``srf_amplitude``, ``srf_size``.

    Shared (across all voxels) parameters (6)
    -----------------------------------------
    ``sigma_AF``, ``g_HP``, ``g_LP``, ``sigma_dyn``,
    ``g_HP_dyn``, ``g_LP_dyn``.

    Total: 13 parameters per voxel (7 per-voxel + 6 shared).

    See :class:`DynamicAttentionFieldPRF2D_v3` for the modulation
    formula. Indices below shift by +2 versus the Gaussian v3 because
    ``srf_amplitude`` and ``srf_size`` sit at positions 5, 6 between
    ``amplitude`` and ``sigma_AF``.
    """

    parameter_labels = ['x', 'y', 'sd', 'baseline', 'amplitude',
                        'srf_amplitude', 'srf_size',
                        'sigma_AF', 'g_HP', 'g_LP',
                        'sigma_dyn', 'g_HP_dyn', 'g_LP_dyn']

    def __init__(self, grid_coordinates=None, paradigm=None, data=None,
                 parameters=None, condition_indicator=None,
                 dynamic_indicator=None,
                 ring_positions=None, mode='suppression',
                 weights=None, omega=None,
                 positive_image_values_only=True,
                 verbosity=logging.INFO, **kwargs):
        if dynamic_indicator is None:
            raise ValueError(
                "DoGDynamicAttentionFieldPRF2D_v3 requires a "
                "`dynamic_indicator` array of shape "
                "(n_timepoints, n_ring_positions).")

        super().__init__(
            grid_coordinates=grid_coordinates, paradigm=paradigm, data=data,
            parameters=parameters, condition_indicator=condition_indicator,
            ring_positions=ring_positions, mode=mode,
            weights=weights, omega=omega,
            positive_image_values_only=positive_image_values_only,
            verbosity=verbosity, **kwargs)

        self.dynamic_indicator = np.asarray(dynamic_indicator,
                                            dtype=np.float32)
        if self.dynamic_indicator.shape[1] != self.n_conditions:
            raise ValueError(
                f"dynamic_indicator has {self.dynamic_indicator.shape[1]} "
                f"channels but ring_positions has {self.n_conditions}; "
                "channels must align with ring_positions.")
        self._tf_dynamic_indicator = tf.constant(self.dynamic_indicator,
                                                 dtype=tf.float32)

    @tf.function
    def _attention_modulation_dynamic_v3(self, parameters):
        """Per-TR dynamic-AF modulation field on the stimulus grid (v3, DoG).

        Identical formula to
        :meth:`DynamicAttentionFieldPRF2D_v3._attention_modulation_dynamic_v3`,
        but parameter indices shift by +2:
            sigma_dyn -> 10,  g_HP_dyn -> 11,  g_LP_dyn -> 12.
        """
        # Take shared parameters from the first batch / first voxel.
        sigma_dyn = parameters[0, 0, 10]                   # scalar
        g_HP_dyn = parameters[0, 0, 11]                    # scalar
        g_LP_dyn = parameters[0, 0, 12]                    # scalar

        # Grid: (n_grid, 2).
        gx = self._grid_coordinates[:, 0][tf.newaxis, :]   # (1, G)
        gy = self._grid_coordinates[:, 1][tf.newaxis, :]

        # Ring positions: (n_C, 2)  ->  (n_C, 1).
        rx = self._tf_ring_positions[:, 0][:, tf.newaxis]
        ry = self._tf_ring_positions[:, 1][:, tf.newaxis]

        # Per-ring DYNAMIC AF Gaussian (peak-normalized to 1): (n_C, G).
        # Uses sigma_dyn — separate from the sustained sigma_AF.
        diff_sq = (gx - rx) ** 2 + (gy - ry) ** 2
        A_dyn = tf.exp(-diff_sq / (2.0 * sigma_dyn ** 2))  # (n_C, G)

        # Per-TR per-ring "is HP" mask: (T, n_C).
        is_hp_per_tr = self._tf_condition_indicator        # (T, n_C)

        # d_ℓ(t): (T, n_C). Per-TR per-ring distractor on-fraction.
        d = self._tf_dynamic_indicator                     # (T, n_C)

        # Split into HP-dyn and LP-dyn weights per (t, ℓ).
        w_hp = d * is_hp_per_tr                            # (T, n_C)
        w_lp = d * (1.0 - is_hp_per_tr)                    # (T, n_C)

        # Σ_ℓ w_hp[t, ℓ] · A_ℓ^{dyn}(g) and Σ_ℓ w_lp[t, ℓ] · A_ℓ^{dyn}(g).
        field_hp = tf.einsum('tl,lg->tg', w_hp, A_dyn)
        field_lp = tf.einsum('tl,lg->tg', w_lp, A_dyn)

        return g_HP_dyn * field_hp + g_LP_dyn * field_lp   # (T, G)

    @tf.function
    def _basis_predictions(self, paradigm, parameters):
        # paradigm: (B, T, G)
        # parameters: (B, V, n_parameters=13)

        # Per-voxel DoG receptive field: (B, V, G).
        rf = self._get_rf(self.grid_coordinates, parameters)

        # Sustained per-condition AF modulation: (B, V, n_C, G). Uses
        # sigma_AF (index 7) via the parent DoGAttentionFieldPRF2D.
        mod_sustained = self._attention_modulation(parameters)

        # Effective per-condition RF (sustained part): (B, V, n_C, G).
        eff_rf_per_cond = rf[:, :, tf.newaxis, :] * mod_sustained

        # Sustained partial: (B, T, V) via condition_indicator selection.
        partial = tf.einsum('btg,bvcg->btvc', paradigm, eff_rf_per_cond)
        ci = self._tf_condition_indicator       # (T, n_C)
        sustained = tf.einsum('tc,btvc->btv', ci, partial)

        # Dynamic per-TR modulation: (T, G), HP/LP split, with σ_dyn.
        mod_dyn = self._attention_modulation_dynamic_v3(parameters)

        # Dynamic partial: (B, T, V).
        sign = self._tf_sign
        eff_paradigm_dyn = paradigm * mod_dyn[tf.newaxis, :, :]   # (B, T, G)
        dynamic = sign * tf.einsum('btg,bvg->btv', eff_paradigm_dyn, rf)

        result = sustained + dynamic

        baseline = parameters[:, tf.newaxis, :, 3]
        result = result + baseline

        return result

    @tf.function
    def _transform_parameters_forward(self, parameters):
        # First 10: DoG + sustained-AF transforms (delegate to parent).
        # Then softplus(sigma_dyn) and sign-aware g_HP_dyn / g_LP_dyn.
        base = DoGAttentionFieldPRF2D._transform_parameters_forward(
            self, parameters[:, :10])
        sigma_dyn = tf.math.softplus(parameters[:, 10][:, tf.newaxis])
        if self._signed_gains:
            g_hp_dyn = parameters[:, 11][:, tf.newaxis]
            g_lp_dyn = parameters[:, 12][:, tf.newaxis]
        else:
            g_hp_dyn = tf.math.softplus(parameters[:, 11][:, tf.newaxis])
            g_lp_dyn = tf.math.softplus(parameters[:, 12][:, tf.newaxis])
        return tf.concat([base, sigma_dyn, g_hp_dyn, g_lp_dyn], axis=1)

    @tf.function
    def _transform_parameters_backward(self, parameters):
        base = DoGAttentionFieldPRF2D._transform_parameters_backward(
            self, parameters[:, :10])
        sigma_dyn_unb = tfp.math.softplus_inverse(
            parameters[:, 10][:, tf.newaxis])
        if self._signed_gains:
            g_hp_dyn_unb = parameters[:, 11][:, tf.newaxis]
            g_lp_dyn_unb = parameters[:, 12][:, tf.newaxis]
        else:
            g_hp_dyn_unb = tfp.math.softplus_inverse(
                parameters[:, 11][:, tf.newaxis])
            g_lp_dyn_unb = tfp.math.softplus_inverse(
                parameters[:, 12][:, tf.newaxis])
        return tf.concat(
            [base, sigma_dyn_unb, g_hp_dyn_unb, g_lp_dyn_unb], axis=1)


class DoGDynamicAttentionFieldPRF2DWithHRF_v3(HRFEncodingModel,
                                              DoGDynamicAttentionFieldPRF2D_v3):
    """HRF-convolved version of :class:`DoGDynamicAttentionFieldPRF2D_v3`.

    Free parameters::

        ['x', 'y', 'sd', 'baseline', 'amplitude',
         'srf_amplitude', 'srf_size',
         'sigma_AF', 'g_HP', 'g_LP',
         'sigma_dyn', 'g_HP_dyn', 'g_LP_dyn']
        (+ HRF parameters if flexible)

    During joint AF + DoG-PRF fitting, pass

        shared_pars=['sigma_AF', 'g_HP', 'g_LP',
                     'sigma_dyn', 'g_HP_dyn', 'g_LP_dyn']

    to the :class:`braincoder.optimize.ParameterFitter`.
    """

    def __init__(self, grid_coordinates=None, paradigm=None, data=None,
                 parameters=None, condition_indicator=None,
                 dynamic_indicator=None,
                 ring_positions=None, mode='suppression',
                 positive_image_values_only=True,
                 weights=None, hrf_model=None,
                 flexible_hrf_parameters=False,
                 verbosity=logging.INFO, **kwargs):

        DoGDynamicAttentionFieldPRF2D_v3.__init__(
            self, grid_coordinates=grid_coordinates, paradigm=paradigm,
            data=data, parameters=parameters,
            condition_indicator=condition_indicator,
            dynamic_indicator=dynamic_indicator,
            ring_positions=ring_positions, mode=mode,
            weights=weights, verbosity=verbosity,
            positive_image_values_only=positive_image_values_only, **kwargs)

        HRFEncodingModel.__init__(self, hrf_model=hrf_model,
                                  flexible_hrf_parameters=flexible_hrf_parameters,
                                  **kwargs)

    @tf.function
    def _transform_parameters_forward(self, parameters):
        if self.flexible_hrf_parameters:
            n_hrf_pars = len(self.hrf_model.parameter_labels)
            encoding_pars = DoGDynamicAttentionFieldPRF2D_v3._transform_parameters_forward(
                self, parameters[:, :-n_hrf_pars])
            hrf_pars = self.hrf_model._transform_parameters_forward(
                parameters[:, -n_hrf_pars:])
            return tf.concat([encoding_pars, hrf_pars], axis=1)
        else:
            return DoGDynamicAttentionFieldPRF2D_v3._transform_parameters_forward(
                self, parameters)

    @tf.function
    def _transform_parameters_backward(self, parameters):
        if self.flexible_hrf_parameters:
            n_hrf_pars = len(self.hrf_model.parameter_labels)
            encoding_pars = DoGDynamicAttentionFieldPRF2D_v3._transform_parameters_backward(
                self, parameters[:, :-n_hrf_pars])
            hrf_pars = self.hrf_model._transform_parameters_backward(
                parameters[:, -n_hrf_pars:])
            return tf.concat([encoding_pars, hrf_pars], axis=1)
        else:
            return DoGDynamicAttentionFieldPRF2D_v3._transform_parameters_backward(
                self, parameters)


class DynamicAttentionFieldPRF2D(AttentionFieldPRF2D):
    """Dynamic Attention-Field-aware 2D Gaussian PRF.

    Extends :class:`AttentionFieldPRF2D` with a *per-TR* dynamic
    distractor-pulse term on top of the existing per-run sustained
    HP/LP modulation. The full modulation field is

        M(g, t) = 1 + sign · [ g_HP · A_{H_run(t)}(g)
                              + g_LP · Σ_{ℓ ≠ H_run(t)} A_ℓ(g)
                              + g_dyn · Σ_ℓ d_ℓ(t) · A_ℓ_dyn(g) ]

    where:
    - ``d_ℓ(t)`` ∈ [0, 1] is the per-TR fraction of the TR during which
      a distractor was on at ring location ℓ (provided as
      ``dynamic_indicator``, shape ``(n_timepoints, n_ring_positions)``).
    - ``A_ℓ_dyn(g)`` is a unit-peak Gaussian centered at ring position
      ℓ with shared width ``sigma_dyn`` (independent of ``sigma_AF``).
    - ``g_dyn`` is a shared scalar gain. Sign is governed by ``mode``
      exactly like ``g_HP``/``g_LP``: positive (softplus) in
      ``'attraction'``/``'suppression'``, free-sign in ``'signed'``.

    The sustained term is identical to the parent class; the dynamic
    term is purely additive in the field, so it integrates cleanly into
    the same paradigm-multiply-and-convolve forward pass.

    Per-voxel parameters
    --------------------
    ``x``, ``y``, ``sd``, ``baseline``, ``amplitude``  — standard PRF.

    Shared (across all voxels) parameters
    -------------------------------------
    ``sigma_AF``, ``g_HP``, ``g_LP`` — sustained AF (as in parent).
    ``sigma_dyn`` : positive
        Width of every dynamic-AF Gaussian.
    ``g_dyn`` : signed (mode='signed') or positive (otherwise)
        Modulation amplitude of the per-trial distractor pulse.
    """

    parameter_labels = ['x', 'y', 'sd', 'baseline', 'amplitude',
                        'sigma_AF', 'g_HP', 'g_LP',
                        'sigma_dyn', 'g_dyn']

    def __init__(self, grid_coordinates=None, paradigm=None, data=None,
                 parameters=None, condition_indicator=None,
                 dynamic_indicator=None,
                 ring_positions=None, mode='suppression',
                 weights=None, omega=None,
                 positive_image_values_only=True,
                 verbosity=logging.INFO, **kwargs):
        if dynamic_indicator is None:
            raise ValueError(
                "DynamicAttentionFieldPRF2D requires a `dynamic_indicator` "
                "array of shape (n_timepoints, n_ring_positions).")

        super().__init__(
            grid_coordinates=grid_coordinates, paradigm=paradigm, data=data,
            parameters=parameters, condition_indicator=condition_indicator,
            ring_positions=ring_positions, mode=mode,
            weights=weights, omega=omega,
            positive_image_values_only=positive_image_values_only,
            verbosity=verbosity, **kwargs)

        self.dynamic_indicator = np.asarray(dynamic_indicator,
                                            dtype=np.float32)
        if self.dynamic_indicator.shape[1] != self.n_conditions:
            raise ValueError(
                f"dynamic_indicator has {self.dynamic_indicator.shape[1]} "
                f"channels but ring_positions has {self.n_conditions}; "
                "channels must align with ring_positions.")
        self._tf_dynamic_indicator = tf.constant(self.dynamic_indicator,
                                                 dtype=tf.float32)

    @tf.function
    def _attention_modulation_dynamic(self, parameters):
        """Per-TR dynamic-AF modulation field on the stimulus grid.

        Returns
        -------
        mod_dyn : tf.Tensor, shape (n_timepoints, n_grid)
            The per-TR sum  g_dyn · Σ_ℓ d_ℓ(t) · A_ℓ_dyn(g),
            BEFORE the sign and the +1 baseline are applied.

        Notes
        -----
        ``sigma_dyn`` and ``g_dyn`` are shared across voxels in our
        intended fitting setup, so we evaluate this with parameters
        from the first batch / first voxel only — yielding a (T, G)
        tensor that we can broadcast cheaply into the predict pass.
        """
        # Take shared parameters from the first batch / first voxel.
        sigma_dyn = parameters[0, 0, 8]                    # scalar
        g_dyn = parameters[0, 0, 9]                        # scalar

        # Grid: (n_grid, 2).
        gx = self._grid_coordinates[:, 0][tf.newaxis, :]   # (1, G)
        gy = self._grid_coordinates[:, 1][tf.newaxis, :]

        # Ring positions: (n_C, 2)  ->  (n_C, 1).
        rx = self._tf_ring_positions[:, 0][:, tf.newaxis]
        ry = self._tf_ring_positions[:, 1][:, tf.newaxis]

        # Per-ring dynamic Gaussian (peak-normalized to 1): (n_C, G).
        diff_sq = (gx - rx) ** 2 + (gy - ry) ** 2
        A_dyn = tf.exp(-diff_sq / (2.0 * sigma_dyn ** 2))   # (n_C, G)

        # d_ℓ(t): (T, n_C); A_dyn: (n_C, G)  ->  (T, G).
        # Σ_ℓ d_ℓ(t) · A_ℓ_dyn(g)
        per_tr_field = tf.einsum('tl,lg->tg',
                                 self._tf_dynamic_indicator, A_dyn)
        return g_dyn * per_tr_field  # (T, G)

    @tf.function
    def _basis_predictions(self, paradigm, parameters):
        # paradigm: (B, T, G)
        # parameters: (B, V, n_parameters)

        # Per-voxel SD-pRF on the grid: (B, V, G).
        rf = self._get_rf(self.grid_coordinates, parameters)

        # Sustained per-condition modulation field on the grid:
        # (B, V, n_C, G).
        mod_sustained = self._attention_modulation(parameters)

        # Effective per-condition RF (sustained part): (B, V, n_C, G).
        eff_rf_per_cond = rf[:, :, tf.newaxis, :] * mod_sustained

        # Sustained partial: (B, T, V) via condition_indicator selection.
        # partial[B, T, V, C] = Σ_g paradigm[B, T, g] · eff_rf[B, V, C, g]
        partial = tf.einsum('btg,bvcg->btvc', paradigm, eff_rf_per_cond)
        ci = self._tf_condition_indicator       # (T, n_C)
        sustained = tf.einsum('tc,btvc->btv', ci, partial)

        # Dynamic per-TR modulation: (T, G). Cheap because shared across
        # voxels — we don't materialize a (B, V, T, G) tensor.
        mod_dyn = self._attention_modulation_dynamic(parameters)

        # Dynamic partial: (B, T, V).
        # paradigm: (B, T, G); mod_dyn: (T, G); rf: (B, V, G).
        # We want sign · Σ_g paradigm[B, T, g] · mod_dyn[T, g] · rf[B, V, g].
        # mod_dyn is the *additive* dynamic modulation BEFORE sign and +1
        # baseline; the +1 baseline is already implicit in the sustained
        # term (which uses the full M_C(g) including the +1).
        sign = self._tf_sign
        eff_paradigm_dyn = paradigm * mod_dyn[tf.newaxis, :, :]   # (B, T, G)
        dynamic = sign * tf.einsum('btg,bvg->btv', eff_paradigm_dyn, rf)

        result = sustained + dynamic

        # Note: baseline was already added inside the parent's
        # _basis_predictions logic — but we don't call it. So add it now.
        baseline = parameters[:, tf.newaxis, :, 3]
        result = result + baseline

        return result

    @tf.function
    def _transform_parameters_forward(self, parameters):
        # Re-use parent's transform for the first 8 params, then add
        # softplus(sigma_dyn) and a sign-aware g_dyn.
        base = AttentionFieldPRF2D._transform_parameters_forward(
            self, parameters[:, :8])
        if self._signed_gains:
            g_dyn = parameters[:, 9][:, tf.newaxis]
        else:
            g_dyn = tf.math.softplus(parameters[:, 9][:, tf.newaxis])
        sigma_dyn = tf.math.softplus(parameters[:, 8][:, tf.newaxis])
        return tf.concat([base, sigma_dyn, g_dyn], axis=1)

    @tf.function
    def _transform_parameters_backward(self, parameters):
        base = AttentionFieldPRF2D._transform_parameters_backward(
            self, parameters[:, :8])
        if self._signed_gains:
            g_dyn_unb = parameters[:, 9][:, tf.newaxis]
        else:
            g_dyn_unb = tfp.math.softplus_inverse(
                parameters[:, 9][:, tf.newaxis])
        sigma_dyn_unb = tfp.math.softplus_inverse(
            parameters[:, 8][:, tf.newaxis])
        return tf.concat([base, sigma_dyn_unb, g_dyn_unb], axis=1)


class DynamicAttentionFieldPRF2DWithHRF(HRFEncodingModel,
                                        DynamicAttentionFieldPRF2D):
    """HRF-convolved version of :class:`DynamicAttentionFieldPRF2D`.

    The set of free parameters is::

        ['x', 'y', 'sd', 'baseline', 'amplitude',
         'sigma_AF', 'g_HP', 'g_LP', 'sigma_dyn', 'g_dyn']
        (+ HRF parameters if flexible)

    During joint AF + PRF fitting, pass
    ``shared_pars=['sigma_AF', 'g_HP', 'g_LP', 'sigma_dyn', 'g_dyn']``
    to the :class:`braincoder.optimize.ParameterFitter`.
    """

    def __init__(self, grid_coordinates=None, paradigm=None, data=None,
                 parameters=None, condition_indicator=None,
                 dynamic_indicator=None,
                 ring_positions=None, mode='suppression',
                 positive_image_values_only=True,
                 weights=None, hrf_model=None,
                 flexible_hrf_parameters=False,
                 verbosity=logging.INFO, **kwargs):

        DynamicAttentionFieldPRF2D.__init__(
            self, grid_coordinates=grid_coordinates, paradigm=paradigm,
            data=data, parameters=parameters,
            condition_indicator=condition_indicator,
            dynamic_indicator=dynamic_indicator,
            ring_positions=ring_positions, mode=mode,
            weights=weights, verbosity=verbosity,
            positive_image_values_only=positive_image_values_only, **kwargs)

        HRFEncodingModel.__init__(self, hrf_model=hrf_model,
                                  flexible_hrf_parameters=flexible_hrf_parameters,
                                  **kwargs)

    @tf.function
    def _transform_parameters_forward(self, parameters):
        if self.flexible_hrf_parameters:
            n_hrf_pars = len(self.hrf_model.parameter_labels)
            encoding_pars = DynamicAttentionFieldPRF2D._transform_parameters_forward(
                self, parameters[:, :-n_hrf_pars])
            hrf_pars = self.hrf_model._transform_parameters_forward(
                parameters[:, -n_hrf_pars:])
            return tf.concat([encoding_pars, hrf_pars], axis=1)
        else:
            return DynamicAttentionFieldPRF2D._transform_parameters_forward(
                self, parameters)

    @tf.function
    def _transform_parameters_backward(self, parameters):
        if self.flexible_hrf_parameters:
            n_hrf_pars = len(self.hrf_model.parameter_labels)
            encoding_pars = DynamicAttentionFieldPRF2D._transform_parameters_backward(
                self, parameters[:, :-n_hrf_pars])
            hrf_pars = self.hrf_model._transform_parameters_backward(
                parameters[:, -n_hrf_pars:])
            return tf.concat([encoding_pars, hrf_pars], axis=1)
        else:
            return DynamicAttentionFieldPRF2D._transform_parameters_backward(
                self, parameters)


class DynamicAttentionFieldPRF2D_v2(AttentionFieldPRF2D):
    """Dynamic Attention-Field-aware 2D Gaussian PRF (v2: shared σ, split gain).

    Refactor of :class:`DynamicAttentionFieldPRF2D` motivated by the
    σ_AF/σ_dyn degeneracy and the desire to test sustained vs phasic
    HP-vs-LP modulation independently.

    Differences from v1:
    - The dynamic-AF Gaussian uses the SAME ``sigma_AF`` as the sustained
      term (biologically the AF size shouldn't depend on whether the
      modulation is sustained or phasic), so ``sigma_dyn`` is dropped.
    - The single ``g_dyn`` is split into two gains:
      * ``g_HP_dyn`` — applied when a distractor is on at the run's HP
        location.
      * ``g_LP_dyn`` — applied when a distractor is on at any of the 3
        LP (non-HP) ring locations.

    The full forward modulation is

        M(g, t) = 1 + sign · [
              g_HP     · A_{H_run(t)}(g)
            + g_LP     · Σ_{ℓ ≠ H_run(t)} A_ℓ(g)
            + g_HP_dyn · d_{H_run(t)}(t) · A_{H_run(t)}(g)
            + g_LP_dyn · Σ_{ℓ ≠ H_run(t)} d_ℓ(t) · A_ℓ(g)
        ]

    where ``A_ℓ`` is a unit-peak Gaussian on the grid centered at ring
    position ℓ with shared width ``sigma_AF`` (SAME σ as the sustained
    term — this is the key v2 change), and ``d_ℓ(t) ∈ [0,1]`` is the
    per-TR fraction-of-distractor-on at ring ℓ (provided as
    ``dynamic_indicator``).

    Per-voxel parameters
    --------------------
    ``x``, ``y``, ``sd``, ``baseline``, ``amplitude``  — standard PRF.

    Shared (across all voxels) parameters
    -------------------------------------
    ``sigma_AF`` : positive
        Width of every attention-field Gaussian — sustained AND dynamic.
    ``g_HP``, ``g_LP`` : sustained gains (same as v1).
    ``g_HP_dyn``, ``g_LP_dyn`` : per-TR phasic gains, applied to the
        per-ring dynamic indicator. Sign convention follows ``mode``
        (signed = identity / softplus otherwise), as for the sustained
        gains.
    """

    parameter_labels = ['x', 'y', 'sd', 'baseline', 'amplitude',
                        'sigma_AF', 'g_HP', 'g_LP',
                        'g_HP_dyn', 'g_LP_dyn']

    def __init__(self, grid_coordinates=None, paradigm=None, data=None,
                 parameters=None, condition_indicator=None,
                 dynamic_indicator=None,
                 ring_positions=None, mode='suppression',
                 weights=None, omega=None,
                 positive_image_values_only=True,
                 verbosity=logging.INFO, **kwargs):
        if dynamic_indicator is None:
            raise ValueError(
                "DynamicAttentionFieldPRF2D_v2 requires a `dynamic_indicator` "
                "array of shape (n_timepoints, n_ring_positions).")

        super().__init__(
            grid_coordinates=grid_coordinates, paradigm=paradigm, data=data,
            parameters=parameters, condition_indicator=condition_indicator,
            ring_positions=ring_positions, mode=mode,
            weights=weights, omega=omega,
            positive_image_values_only=positive_image_values_only,
            verbosity=verbosity, **kwargs)

        self.dynamic_indicator = np.asarray(dynamic_indicator,
                                            dtype=np.float32)
        if self.dynamic_indicator.shape[1] != self.n_conditions:
            raise ValueError(
                f"dynamic_indicator has {self.dynamic_indicator.shape[1]} "
                f"channels but ring_positions has {self.n_conditions}; "
                "channels must align with ring_positions.")
        self._tf_dynamic_indicator = tf.constant(self.dynamic_indicator,
                                                 dtype=tf.float32)

    @tf.function
    def _attention_modulation_dynamic_v2(self, parameters):
        """Per-TR dynamic-AF modulation field on the stimulus grid (v2).

        Returns
        -------
        mod_dyn : tf.Tensor, shape (n_timepoints, n_grid)
            The per-TR sum

                g_HP_dyn · Σ_ℓ ( d_ℓ(t) · is_hp[t, ℓ] ) · A_ℓ(g)
              + g_LP_dyn · Σ_ℓ ( d_ℓ(t) · (1 − is_hp[t, ℓ]) ) · A_ℓ(g)

            BEFORE the sign and the +1 baseline are applied, where
            ``is_hp[t, ℓ] = condition_indicator[t, ℓ]`` (since condition
            c is encoded one-hot and HP_c is ring c by construction).

        Notes
        -----
        ``sigma_AF``, ``g_HP_dyn``, ``g_LP_dyn`` are shared across voxels
        in our intended fitting setup, so we evaluate this with
        parameters from the first batch / first voxel only — yielding a
        (T, G) tensor that we can broadcast cheaply into the predict
        pass. We re-use the SAME ``sigma_AF`` as the sustained term
        (parameter index 5), NOT a separate ``sigma_dyn``.
        """
        # Take shared parameters from the first batch / first voxel.
        sigma_AF = parameters[0, 0, 5]                     # scalar
        g_HP_dyn = parameters[0, 0, 8]                     # scalar
        g_LP_dyn = parameters[0, 0, 9]                     # scalar

        # Grid: (n_grid, 2).
        gx = self._grid_coordinates[:, 0][tf.newaxis, :]   # (1, G)
        gy = self._grid_coordinates[:, 1][tf.newaxis, :]

        # Ring positions: (n_C, 2)  ->  (n_C, 1).
        rx = self._tf_ring_positions[:, 0][:, tf.newaxis]
        ry = self._tf_ring_positions[:, 1][:, tf.newaxis]

        # Per-ring AF Gaussian (peak-normalized to 1): (n_C, G).
        # Uses sigma_AF, NOT a separate sigma_dyn.
        diff_sq = (gx - rx) ** 2 + (gy - ry) ** 2
        A = tf.exp(-diff_sq / (2.0 * sigma_AF ** 2))       # (n_C, G)

        # Per-TR per-ring "is HP" mask: (T, n_C).
        # condition_indicator already one-hot encodes the HP ring per TR.
        is_hp_per_tr = self._tf_condition_indicator        # (T, n_C)

        # d_ℓ(t): (T, n_C). Per-TR per-ring distractor on-fraction.
        d = self._tf_dynamic_indicator                     # (T, n_C)

        # Split into HP-dyn and LP-dyn weights per (t, ℓ).
        w_hp = d * is_hp_per_tr                            # (T, n_C)
        w_lp = d * (1.0 - is_hp_per_tr)                    # (T, n_C)

        # Σ_ℓ w_hp[t, ℓ] · A_ℓ(g) and Σ_ℓ w_lp[t, ℓ] · A_ℓ(g).
        # Both -> (T, G).
        field_hp = tf.einsum('tl,lg->tg', w_hp, A)
        field_lp = tf.einsum('tl,lg->tg', w_lp, A)

        return g_HP_dyn * field_hp + g_LP_dyn * field_lp   # (T, G)

    @tf.function
    def _basis_predictions(self, paradigm, parameters):
        # paradigm: (B, T, G)
        # parameters: (B, V, n_parameters)

        # Per-voxel SD-pRF on the grid: (B, V, G).
        rf = self._get_rf(self.grid_coordinates, parameters)

        # Sustained per-condition modulation field on the grid:
        # (B, V, n_C, G). Identical to the parent / v1 class.
        mod_sustained = self._attention_modulation(parameters)

        # Effective per-condition RF (sustained part): (B, V, n_C, G).
        eff_rf_per_cond = rf[:, :, tf.newaxis, :] * mod_sustained

        # Sustained partial: (B, T, V) via condition_indicator selection.
        partial = tf.einsum('btg,bvcg->btvc', paradigm, eff_rf_per_cond)
        ci = self._tf_condition_indicator       # (T, n_C)
        sustained = tf.einsum('tc,btvc->btv', ci, partial)

        # Dynamic per-TR modulation: (T, G), with HP and LP split.
        mod_dyn = self._attention_modulation_dynamic_v2(parameters)

        # Dynamic partial: (B, T, V).
        # paradigm: (B, T, G); mod_dyn: (T, G); rf: (B, V, G).
        # We want sign · Σ_g paradigm[B, T, g] · mod_dyn[T, g] · rf[B, V, g].
        # mod_dyn is the *additive* dynamic modulation BEFORE sign and +1
        # baseline; the +1 baseline is already implicit in the sustained
        # term (which uses the full M_C(g) including the +1).
        sign = self._tf_sign
        eff_paradigm_dyn = paradigm * mod_dyn[tf.newaxis, :, :]   # (B, T, G)
        dynamic = sign * tf.einsum('btg,bvg->btv', eff_paradigm_dyn, rf)

        result = sustained + dynamic

        # Note: baseline was already added inside the parent's
        # _basis_predictions logic — but we don't call it. So add it now.
        baseline = parameters[:, tf.newaxis, :, 3]
        result = result + baseline

        return result

    @tf.function
    def _transform_parameters_forward(self, parameters):
        # Re-use parent's transform for the first 8 params (x, y, sd,
        # baseline, amplitude, sigma_AF, g_HP, g_LP), then add sign-aware
        # g_HP_dyn and g_LP_dyn. NO sigma_dyn.
        base = AttentionFieldPRF2D._transform_parameters_forward(
            self, parameters[:, :8])
        if self._signed_gains:
            g_hp_dyn = parameters[:, 8][:, tf.newaxis]
            g_lp_dyn = parameters[:, 9][:, tf.newaxis]
        else:
            g_hp_dyn = tf.math.softplus(parameters[:, 8][:, tf.newaxis])
            g_lp_dyn = tf.math.softplus(parameters[:, 9][:, tf.newaxis])
        return tf.concat([base, g_hp_dyn, g_lp_dyn], axis=1)

    @tf.function
    def _transform_parameters_backward(self, parameters):
        base = AttentionFieldPRF2D._transform_parameters_backward(
            self, parameters[:, :8])
        if self._signed_gains:
            g_hp_dyn_unb = parameters[:, 8][:, tf.newaxis]
            g_lp_dyn_unb = parameters[:, 9][:, tf.newaxis]
        else:
            g_hp_dyn_unb = tfp.math.softplus_inverse(
                parameters[:, 8][:, tf.newaxis])
            g_lp_dyn_unb = tfp.math.softplus_inverse(
                parameters[:, 9][:, tf.newaxis])
        return tf.concat([base, g_hp_dyn_unb, g_lp_dyn_unb], axis=1)


class DynamicAttentionFieldPRF2DWithHRF_v2(HRFEncodingModel,
                                           DynamicAttentionFieldPRF2D_v2):
    """HRF-convolved version of :class:`DynamicAttentionFieldPRF2D_v2`.

    The set of free parameters is::

        ['x', 'y', 'sd', 'baseline', 'amplitude',
         'sigma_AF', 'g_HP', 'g_LP', 'g_HP_dyn', 'g_LP_dyn']
        (+ HRF parameters if flexible)

    During joint AF + PRF fitting, pass

        shared_pars=['sigma_AF', 'g_HP', 'g_LP', 'g_HP_dyn', 'g_LP_dyn']

    to the :class:`braincoder.optimize.ParameterFitter`.
    """

    def __init__(self, grid_coordinates=None, paradigm=None, data=None,
                 parameters=None, condition_indicator=None,
                 dynamic_indicator=None,
                 ring_positions=None, mode='suppression',
                 positive_image_values_only=True,
                 weights=None, hrf_model=None,
                 flexible_hrf_parameters=False,
                 verbosity=logging.INFO, **kwargs):

        DynamicAttentionFieldPRF2D_v2.__init__(
            self, grid_coordinates=grid_coordinates, paradigm=paradigm,
            data=data, parameters=parameters,
            condition_indicator=condition_indicator,
            dynamic_indicator=dynamic_indicator,
            ring_positions=ring_positions, mode=mode,
            weights=weights, verbosity=verbosity,
            positive_image_values_only=positive_image_values_only, **kwargs)

        HRFEncodingModel.__init__(self, hrf_model=hrf_model,
                                  flexible_hrf_parameters=flexible_hrf_parameters,
                                  **kwargs)

    @tf.function
    def _transform_parameters_forward(self, parameters):
        if self.flexible_hrf_parameters:
            n_hrf_pars = len(self.hrf_model.parameter_labels)
            encoding_pars = DynamicAttentionFieldPRF2D_v2._transform_parameters_forward(
                self, parameters[:, :-n_hrf_pars])
            hrf_pars = self.hrf_model._transform_parameters_forward(
                parameters[:, -n_hrf_pars:])
            return tf.concat([encoding_pars, hrf_pars], axis=1)
        else:
            return DynamicAttentionFieldPRF2D_v2._transform_parameters_forward(
                self, parameters)

    @tf.function
    def _transform_parameters_backward(self, parameters):
        if self.flexible_hrf_parameters:
            n_hrf_pars = len(self.hrf_model.parameter_labels)
            encoding_pars = DynamicAttentionFieldPRF2D_v2._transform_parameters_backward(
                self, parameters[:, :-n_hrf_pars])
            hrf_pars = self.hrf_model._transform_parameters_backward(
                parameters[:, -n_hrf_pars:])
            return tf.concat([encoding_pars, hrf_pars], axis=1)
        else:
            return DynamicAttentionFieldPRF2D_v2._transform_parameters_backward(
                self, parameters)


class DynamicAttentionFieldPRF2D_v3(AttentionFieldPRF2D):
    """Dynamic Attention-Field-aware 2D Gaussian PRF (v3: separate σ_dyn + split gain).

    Combines features of v1 (separate ``sigma_dyn`` from ``sigma_AF``)
    and v2 (HP/LP split for the dynamic gain). The full forward
    modulation is

        M(g, t) = 1 + sign · [
              g_HP     · A_{H_run(t)}^{sus}(g)
            + g_LP     · Σ_{ℓ ≠ H_run(t)} A_ℓ^{sus}(g)
            + g_HP_dyn · d_{H_run(t)}(t) · A_{H_run(t)}^{dyn}(g)
            + g_LP_dyn · Σ_{ℓ ≠ H_run(t)} d_ℓ(t) · A_ℓ^{dyn}(g)
        ]

    where ``A_ℓ^{sus}`` is a unit-peak Gaussian centered at ring ℓ with
    width ``sigma_AF`` and ``A_ℓ^{dyn}`` is a unit-peak Gaussian at the
    same ring location but with INDEPENDENT width ``sigma_dyn``. The
    distractor disk subtends ~0.4° so a smaller ``sigma_dyn`` than
    ``sigma_AF`` is plausible.

    Per-voxel parameters
    --------------------
    ``x``, ``y``, ``sd``, ``baseline``, ``amplitude``  — standard PRF.

    Shared (across all voxels) parameters
    -------------------------------------
    ``sigma_AF`` : positive
        Width of the SUSTAINED attention-field Gaussians.
    ``g_HP``, ``g_LP`` : sustained HP / LP gains.
    ``sigma_dyn`` : positive
        Width of the DYNAMIC attention-field Gaussians.
    ``g_HP_dyn``, ``g_LP_dyn`` : per-TR phasic HP / LP gains.

    All four gains follow the ``mode`` sign convention (signed = identity,
    softplus otherwise).
    """

    parameter_labels = ['x', 'y', 'sd', 'baseline', 'amplitude',
                        'sigma_AF', 'g_HP', 'g_LP',
                        'sigma_dyn', 'g_HP_dyn', 'g_LP_dyn']

    def __init__(self, grid_coordinates=None, paradigm=None, data=None,
                 parameters=None, condition_indicator=None,
                 dynamic_indicator=None,
                 ring_positions=None, mode='suppression',
                 weights=None, omega=None,
                 positive_image_values_only=True,
                 verbosity=logging.INFO, **kwargs):
        if dynamic_indicator is None:
            raise ValueError(
                "DynamicAttentionFieldPRF2D_v3 requires a `dynamic_indicator` "
                "array of shape (n_timepoints, n_ring_positions).")

        super().__init__(
            grid_coordinates=grid_coordinates, paradigm=paradigm, data=data,
            parameters=parameters, condition_indicator=condition_indicator,
            ring_positions=ring_positions, mode=mode,
            weights=weights, omega=omega,
            positive_image_values_only=positive_image_values_only,
            verbosity=verbosity, **kwargs)

        self.dynamic_indicator = np.asarray(dynamic_indicator,
                                            dtype=np.float32)
        if self.dynamic_indicator.shape[1] != self.n_conditions:
            raise ValueError(
                f"dynamic_indicator has {self.dynamic_indicator.shape[1]} "
                f"channels but ring_positions has {self.n_conditions}; "
                "channels must align with ring_positions.")
        self._tf_dynamic_indicator = tf.constant(self.dynamic_indicator,
                                                 dtype=tf.float32)

    @tf.function
    def _attention_modulation_dynamic_v3(self, parameters):
        """Per-TR dynamic-AF modulation field on the stimulus grid (v3).

        Returns
        -------
        mod_dyn : tf.Tensor, shape (n_timepoints, n_grid)
            The per-TR sum

                g_HP_dyn · Σ_ℓ ( d_ℓ(t) · is_hp[t, ℓ] ) · A_ℓ^{dyn}(g)
              + g_LP_dyn · Σ_ℓ ( d_ℓ(t) · (1 − is_hp[t, ℓ]) ) · A_ℓ^{dyn}(g)

            BEFORE the sign and the +1 baseline are applied. ``A_ℓ^{dyn}``
            uses ``sigma_dyn`` (parameter index 8), INDEPENDENT of
            ``sigma_AF`` (parameter index 5) which governs the sustained
            term.

        Notes
        -----
        ``sigma_dyn``, ``g_HP_dyn``, ``g_LP_dyn`` are shared across voxels
        in our intended fitting setup, so we evaluate this with parameters
        from the first batch / first voxel only.
        """
        # Take shared parameters from the first batch / first voxel.
        sigma_dyn = parameters[0, 0, 8]                    # scalar
        g_HP_dyn = parameters[0, 0, 9]                     # scalar
        g_LP_dyn = parameters[0, 0, 10]                    # scalar

        # Grid: (n_grid, 2).
        gx = self._grid_coordinates[:, 0][tf.newaxis, :]   # (1, G)
        gy = self._grid_coordinates[:, 1][tf.newaxis, :]

        # Ring positions: (n_C, 2)  ->  (n_C, 1).
        rx = self._tf_ring_positions[:, 0][:, tf.newaxis]
        ry = self._tf_ring_positions[:, 1][:, tf.newaxis]

        # Per-ring DYNAMIC AF Gaussian (peak-normalized to 1): (n_C, G).
        # Uses sigma_dyn — separate from the sustained sigma_AF.
        diff_sq = (gx - rx) ** 2 + (gy - ry) ** 2
        A_dyn = tf.exp(-diff_sq / (2.0 * sigma_dyn ** 2))  # (n_C, G)

        # Per-TR per-ring "is HP" mask: (T, n_C).
        is_hp_per_tr = self._tf_condition_indicator        # (T, n_C)

        # d_ℓ(t): (T, n_C). Per-TR per-ring distractor on-fraction.
        d = self._tf_dynamic_indicator                     # (T, n_C)

        # Split into HP-dyn and LP-dyn weights per (t, ℓ).
        w_hp = d * is_hp_per_tr                            # (T, n_C)
        w_lp = d * (1.0 - is_hp_per_tr)                    # (T, n_C)

        # Σ_ℓ w_hp[t, ℓ] · A_ℓ^{dyn}(g) and Σ_ℓ w_lp[t, ℓ] · A_ℓ^{dyn}(g).
        # Both -> (T, G).
        field_hp = tf.einsum('tl,lg->tg', w_hp, A_dyn)
        field_lp = tf.einsum('tl,lg->tg', w_lp, A_dyn)

        return g_HP_dyn * field_hp + g_LP_dyn * field_lp   # (T, G)

    @tf.function
    def _basis_predictions(self, paradigm, parameters):
        # paradigm: (B, T, G)
        # parameters: (B, V, n_parameters)

        # Per-voxel SD-pRF on the grid: (B, V, G).
        rf = self._get_rf(self.grid_coordinates, parameters)

        # Sustained per-condition modulation field on the grid:
        # (B, V, n_C, G). Uses sigma_AF (index 5).
        mod_sustained = self._attention_modulation(parameters)

        # Effective per-condition RF (sustained part): (B, V, n_C, G).
        eff_rf_per_cond = rf[:, :, tf.newaxis, :] * mod_sustained

        # Sustained partial: (B, T, V) via condition_indicator selection.
        partial = tf.einsum('btg,bvcg->btvc', paradigm, eff_rf_per_cond)
        ci = self._tf_condition_indicator       # (T, n_C)
        sustained = tf.einsum('tc,btvc->btv', ci, partial)

        # Dynamic per-TR modulation: (T, G), HP/LP split, with σ_dyn.
        mod_dyn = self._attention_modulation_dynamic_v3(parameters)

        # Dynamic partial: (B, T, V).
        sign = self._tf_sign
        eff_paradigm_dyn = paradigm * mod_dyn[tf.newaxis, :, :]   # (B, T, G)
        dynamic = sign * tf.einsum('btg,bvg->btv', eff_paradigm_dyn, rf)

        result = sustained + dynamic

        # Note: baseline was already added inside the parent's
        # _basis_predictions logic — but we don't call it. So add it now.
        baseline = parameters[:, tf.newaxis, :, 3]
        result = result + baseline

        return result

    @tf.function
    def _transform_parameters_forward(self, parameters):
        # Re-use parent's transform for the first 8 params (x, y, sd,
        # baseline, amplitude, sigma_AF, g_HP, g_LP), then add
        # softplus(sigma_dyn) and sign-aware g_HP_dyn / g_LP_dyn.
        base = AttentionFieldPRF2D._transform_parameters_forward(
            self, parameters[:, :8])
        sigma_dyn = tf.math.softplus(parameters[:, 8][:, tf.newaxis])
        if self._signed_gains:
            g_hp_dyn = parameters[:, 9][:, tf.newaxis]
            g_lp_dyn = parameters[:, 10][:, tf.newaxis]
        else:
            g_hp_dyn = tf.math.softplus(parameters[:, 9][:, tf.newaxis])
            g_lp_dyn = tf.math.softplus(parameters[:, 10][:, tf.newaxis])
        return tf.concat([base, sigma_dyn, g_hp_dyn, g_lp_dyn], axis=1)

    @tf.function
    def _transform_parameters_backward(self, parameters):
        base = AttentionFieldPRF2D._transform_parameters_backward(
            self, parameters[:, :8])
        sigma_dyn_unb = tfp.math.softplus_inverse(
            parameters[:, 8][:, tf.newaxis])
        if self._signed_gains:
            g_hp_dyn_unb = parameters[:, 9][:, tf.newaxis]
            g_lp_dyn_unb = parameters[:, 10][:, tf.newaxis]
        else:
            g_hp_dyn_unb = tfp.math.softplus_inverse(
                parameters[:, 9][:, tf.newaxis])
            g_lp_dyn_unb = tfp.math.softplus_inverse(
                parameters[:, 10][:, tf.newaxis])
        return tf.concat(
            [base, sigma_dyn_unb, g_hp_dyn_unb, g_lp_dyn_unb], axis=1)


class DynamicAttentionFieldPRF2DWithHRF_v3(HRFEncodingModel,
                                           DynamicAttentionFieldPRF2D_v3):
    """HRF-convolved version of :class:`DynamicAttentionFieldPRF2D_v3`.

    The set of free parameters is::

        ['x', 'y', 'sd', 'baseline', 'amplitude',
         'sigma_AF', 'g_HP', 'g_LP',
         'sigma_dyn', 'g_HP_dyn', 'g_LP_dyn']
        (+ HRF parameters if flexible)

    During joint AF + PRF fitting, pass

        shared_pars=['sigma_AF', 'g_HP', 'g_LP',
                     'sigma_dyn', 'g_HP_dyn', 'g_LP_dyn']

    to the :class:`braincoder.optimize.ParameterFitter`.
    """

    def __init__(self, grid_coordinates=None, paradigm=None, data=None,
                 parameters=None, condition_indicator=None,
                 dynamic_indicator=None,
                 ring_positions=None, mode='suppression',
                 positive_image_values_only=True,
                 weights=None, hrf_model=None,
                 flexible_hrf_parameters=False,
                 verbosity=logging.INFO, **kwargs):

        DynamicAttentionFieldPRF2D_v3.__init__(
            self, grid_coordinates=grid_coordinates, paradigm=paradigm,
            data=data, parameters=parameters,
            condition_indicator=condition_indicator,
            dynamic_indicator=dynamic_indicator,
            ring_positions=ring_positions, mode=mode,
            weights=weights, verbosity=verbosity,
            positive_image_values_only=positive_image_values_only, **kwargs)

        HRFEncodingModel.__init__(self, hrf_model=hrf_model,
                                  flexible_hrf_parameters=flexible_hrf_parameters,
                                  **kwargs)

    @tf.function
    def _transform_parameters_forward(self, parameters):
        if self.flexible_hrf_parameters:
            n_hrf_pars = len(self.hrf_model.parameter_labels)
            encoding_pars = DynamicAttentionFieldPRF2D_v3._transform_parameters_forward(
                self, parameters[:, :-n_hrf_pars])
            hrf_pars = self.hrf_model._transform_parameters_forward(
                parameters[:, -n_hrf_pars:])
            return tf.concat([encoding_pars, hrf_pars], axis=1)
        else:
            return DynamicAttentionFieldPRF2D_v3._transform_parameters_forward(
                self, parameters)

    @tf.function
    def _transform_parameters_backward(self, parameters):
        if self.flexible_hrf_parameters:
            n_hrf_pars = len(self.hrf_model.parameter_labels)
            encoding_pars = DynamicAttentionFieldPRF2D_v3._transform_parameters_backward(
                self, parameters[:, :-n_hrf_pars])
            hrf_pars = self.hrf_model._transform_parameters_backward(
                parameters[:, -n_hrf_pars:])
            return tf.concat([encoding_pars, hrf_pars], axis=1)
        else:
            return DynamicAttentionFieldPRF2D_v3._transform_parameters_backward(
                self, parameters)


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
    """Identity mapping from paradigm features to voxel responses.

    Useful when paradigm features already correspond to predicted activity
    (e.g., when estimating weights for design-matrix regressors).  No free
    parameters are tracked, so attempts to set ``parameters`` raise ``ValueError``.
    """
    
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
    """Linear encoding model that adds a voxel-specific baseline parameter."""

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

    @tf.function
    def _predict(self, paradigm, parameters, weights):
        pre_convolve = LinearModelWithBaseline._predict(
            self, paradigm, parameters, weights)

        return self.hrf_model.convolve(pre_convolve)

    @tf.function
    def _predict_no_hrf(self, paradigm, parameters, weights):
        return LinearModelWithBaseline._predict(self, paradigm, parameters, weights)
