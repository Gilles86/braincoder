import tensorflow as tf
import tensorflow_probability as tfp
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
from ..utils import norm, format_data, format_paradigm, format_parameters, format_weights, logit, restrict_radians, lognormalpdf_n, von_mises_pdf, lognormal_pdf_mode_fwhm, norm2d
from tensorflow_probability import distributions as tfd
from ..utils.math import aggressive_softplus, aggressive_softplus_inverse, norm
import pandas as pd
import scipy.stats as ss
from ..stimuli import Stimulus, OneDimensionalRadialStimulus, OneDimensionalGaussianStimulus, OneDimensionalStimulusWithAmplitude, OneDimensionalRadialStimulusWithAmplitude, ImageStimulus, TwoDimensionalStimulus
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
        from .linear import DiscreteModel

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
