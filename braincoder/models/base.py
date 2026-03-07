import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
import keras
from keras import ops
from ..utils import norm, format_data, format_paradigm, format_parameters, format_weights, logit, restrict_radians, lognormalpdf_n, von_mises_pdf, lognormal_pdf_mode_fwhm, norm2d
from ..utils.math import aggressive_softplus, aggressive_softplus_inverse, norm
from ..utils.backend import softplus_inverse, mvn_log_prob, mvt_log_prob, sample_mvn, sample_mvt, sample_student_t, compute_gradients
import scipy.stats as ss
from ..stimuli import Stimulus, OneDimensionalRadialStimulus, OneDimensionalGaussianStimulus, OneDimensionalStimulusWithAmplitude, OneDimensionalRadialStimulusWithAmplitude, ImageStimulus, TwoDimensionalStimulus
from patsy import dmatrix, build_design_matrices

class EncodingModel(object):
    """Abstract base class for encoding models."""

    parameter_labels = None
    stimulus_type = Stimulus

    def _transform_parameters_forward(self, parameters):
        """Map unconstrained optimizer parameters to the model's native space."""
        out = []
        for i, t in enumerate(self.transformations):
            param = parameters[:, i][:, None]
            if isinstance(t, tuple):
                out.append(t[0](param))
            elif t == 'identity':
                out.append(param)
            elif t == 'softplus':
                out.append(ops.softplus(param))
            elif t == 'aggressive_softplus':
                out.append(aggressive_softplus(param))
            elif t == 'sigmoid':
                out.append(ops.sigmoid(param))
            else:
                raise NotImplementedError(f"Unknown transform: {t!r}")
        return ops.concatenate(out, axis=1)

    def _transform_parameters_backward(self, parameters):
        """Inverse of :meth:`_transform_parameters_forward`."""
        out = []
        for i, t in enumerate(self.transformations):
            param = parameters[:, i][:, None]
            if isinstance(t, tuple):
                out.append(t[1](param))
            elif t == 'identity':
                out.append(param)
            elif t == 'softplus':
                out.append(softplus_inverse(param))
            elif t == 'aggressive_softplus':
                out.append(aggressive_softplus_inverse(param))
            elif t == 'sigmoid':
                out.append(ops.log(param / (1.0 - param)))
            else:
                raise NotImplementedError(f"Unknown transform: {t!r}")
        return ops.concatenate(out, axis=1)

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

    def _predict(self, paradigm, parameters, weights=None):
        """Low-level prediction used by ``predict``/``simulate``."""

        if weights is None:
            return self._basis_predictions(paradigm, parameters)
        else:
            return ops.tensordot(self._basis_predictions(paradigm, parameters), weights, axes=[[2], [1]])[:, :, 0, :]

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
            return pd.DataFrame(predictions if isinstance(predictions, np.ndarray) else predictions.numpy(),
                                index=paradigm.index, columns=parameters.index)
        else:
            return pd.DataFrame(predictions if isinstance(predictions, np.ndarray) else predictions.numpy(),
                                index=paradigm.index, columns=weights.columns)

    def simulate(self, paradigm=None, parameters=None, weights=None, noise=1.,
                dof=None,
                n_repeats=1):
        """Generate synthetic data by adding noise to predictions."""

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

        simulated_data = self._simulate(
            stimulus,
            parameters.values[np.newaxis, ...],
            weights_, noise, dof)

        if hasattr(simulated_data, 'numpy'):
            simulated_data = simulated_data.numpy()

        # Collapse the first two dimensions
        simulated_data = np.reshape(simulated_data, (n_repeats*paradigm.shape[0], simulated_data.shape[2]))

        if n_repeats == 1:
            index = pd.Index(paradigm.index, name='stimulus')
        else:
            index = pd.MultiIndex.from_product([np.arange(n_repeats), paradigm.index], names=['repeat', 'stimulus'])

        if weights is None:
            return pd.DataFrame(simulated_data, index=index, columns=parameters.index)
        else:
            return pd.DataFrame(simulated_data, index=index, columns=weights.columns)

    def _simulate(self, paradigm, parameters, weights, noise=1., dof=None):
        """Compute predictions and add noise."""

        n_batches = paradigm.shape[0]
        n_timepoints = paradigm.shape[1]

        if weights is None:
            n_voxels = parameters.shape[1]
        else:
            n_voxels = weights.shape[2]

        if dof is None:
            if np.isscalar(noise):
                noise_samples = keras.random.normal(shape=(n_batches, n_timepoints, n_voxels),
                                                    stddev=noise)
            else:
                noise = noise.astype(np.float32)
                L = ops.cholesky(ops.convert_to_tensor(noise))
                noise_samples = sample_mvn(L, (n_batches, n_timepoints))
        else:
            if np.isscalar(noise):
                noise_samples = sample_student_t(dof, noise, (n_batches, n_timepoints, n_voxels))
            else:
                noise = noise.astype(np.float32)
                L = ops.cholesky(ops.convert_to_tensor(noise))
                noise_samples = sample_mvt(L, dof, (n_batches, n_timepoints))

        return self._predict(paradigm, parameters, weights) + noise_samples

    def _gradient(self, stimuli, parameters):
        """Compute d(predictions)/d(stimuli) using Jacobians. Requires TF backend."""
        import tensorflow as tf
        stimuli = tf.convert_to_tensor(stimuli)

        with tf.GradientTape() as tape:
            tape.watch(stimuli)
            predictions = self._predict(stimuli, parameters)

        jacobians = tape.jacobian(predictions, stimuli)
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

        likelihood = self._likelihood(stimuli.values[np.newaxis, ...],
                                      data.values[np.newaxis, ...],
                                      parameters.values[np.newaxis, ...],
                                      weights[np.newaxis, ...] if weights else None,
                                      omega_chol,
                                      dof,
                                      logp,
                                      normalize)

        if hasattr(likelihood, 'numpy'):
            likelihood = likelihood.numpy()

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

        stimulus_range = self.stimulus._clean_paradigm(stimulus_range)

        if stimulus_range.ndim == 1:
            stimulus_range = stimulus_range[:, np.newaxis, np.newaxis]
        elif stimulus_range.ndim == 2:
            stimulus_range = stimulus_range[:, np.newaxis, :]
        else:
            raise Exception('Stimulus range needs to be either 1D or 2D')

        ll = self._likelihood(stimulus_range,
                              data[np.newaxis, :, :],
                              parameters[np.newaxis, :, :] if parameters is not None else None,
                              weights_,
                              omega,
                              dof,
                              logp=True,
                              normalize=False)

        if hasattr(ll, 'numpy'):
            ll = ll.numpy()

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

        ll = np.exp(ll.apply(lambda d: d-d.max(), 1))

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

    def _likelihood(self, stimuli, data, parameters, weights, omega_chol, dof, logp=False, normalize=False):
        """Compute likelihoods for batches of stimuli."""
        prediction = self._predict(stimuli, parameters, weights)
        return self._likelihood_timeseries(data, prediction, omega_chol, dof, logp, normalize)

    def _likelihood_timeseries(self, data, prediction, omega_chol, dof, logp=False, normalize=False):
        """Evaluate log-probabilities for each residual timeseries."""
        residuals = data - prediction

        # Flatten to 2D for log_prob: (n_batches * n_timepoints, n_voxels)
        n_batches, n_timepoints, n_voxels = residuals.shape
        residuals_2d = ops.reshape(residuals, (n_batches * n_timepoints, n_voxels))

        omega_chol_t = ops.convert_to_tensor(omega_chol, dtype='float32')

        if dof is None:
            p_flat = mvn_log_prob(residuals_2d, omega_chol_t)
        else:
            p_flat = mvt_log_prob(residuals_2d, omega_chol_t, dof)

        p = ops.reshape(p_flat, (n_batches, n_timepoints))

        if logp:
            return p

        if normalize:
            p = p - ops.max(p, axis=1)[:, None]
            p = ops.exp(p)
            p = p / ops.sum(p, axis=1)[:, None]
        else:
            p = ops.exp(p)

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

        L = ops.cholesky(ops.convert_to_tensor(omega, dtype='float32'))

        if analytical:
            import tensorflow as tf
            stimuli_ = tf.Variable(stimuli[np.newaxis, ...], name='stimuli')
            gradient = self._gradient(stimuli_, parameters_)[0]

            y = []
            for i in range(gradient.shape[0]):
                y.append(ops.solve_triangular(L, gradient[i, :, None], lower=True))

            y = ops.concatenate(y, axis=1)
            fisher_info = ops.sum(y ** 2, axis=0)

        else:
            import tensorflow as tf
            stimuli_ = tf.Variable(tf.repeat(stimuli[np.newaxis, ...], n, axis=0), name='stimuli')

            noise = sample_mvn(L, (n,)) if dof is None else sample_mvt(L, dof, (n,))
            pred = self._predict(stimuli_, parameters_, weights_)
            data = pred + noise[:, None, :]

            with tf.GradientTape() as tape:
                ll = self._likelihood(stimuli_, data, parameters_, weights_, L, dof, logp=True, normalize=False)

            dy_dx = tape.gradient(ll, stimuli_)
            fisher_info = tf.reduce_mean(dy_dx**2, 0)[..., 0]

        if hasattr(fisher_info, 'numpy'):
            fisher_info = fisher_info.numpy()

        if stimuli.shape[1] == 1:
            return pd.Series(fisher_info, index=pd.Index(stimuli[:, 0], name='stimulus'), name='Fisher information')
        else:
            return pd.Series(fisher_info, index=pd.MultiIndex.from_frame(pd.DataFrame(stimuli)), name='Fisher information')

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
        """Tensor-ready paradigm (np.array) used inside computation."""

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


        baseline_parameter_values = baseline_parameter_values or {}

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

            parameters.append(ops.sum(np.asarray(design_matrices[parameter], dtype=np.float32)[:, np.newaxis, :] * \
                                      regressor_parameters[:, :, ix:end_ix], axis=2) + self.baseline_parameter_values[parameter])

            ix = end_ix

        parameters = ops.stack(parameters, axis=2)
        parameters = keras.ops.map(self._base_transform_parameters_forward, parameters)

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
        return ops.reshape(result, [1, result.shape[0], -1])

    def _get_paradigm(self, paradigm):
        paradigm = self.stimulus._clean_paradigm(paradigm)
        return paradigm

    def get_conditionspecific_parameters(self, conditions, parameters):

        design_matrices = self.build_design_matrices(conditions)

        if hasattr(parameters, 'values'):
            parameters_ = parameters.values
        else:
            parameters_ = np.array(parameters)

        parameters_ = parameters_[np.newaxis, ...]

        transformed_parameters = self._get_base_parameters(design_matrices, parameters_)
        if hasattr(transformed_parameters, 'numpy'):
            transformed_parameters = transformed_parameters.numpy()

        transformed_parameters = np.reshape(transformed_parameters, (-1, transformed_parameters.shape[-1]))

        transformed_parameters = pd.DataFrame(transformed_parameters,
                                            index=pd.MultiIndex.from_product([conditions.index, parameters.index]),
                                            columns=self.base_parameter_labels)

        return transformed_parameters

    def get_stimulus_pdf(self, data, stimulus_range, parameters=None, weights=None, omega=None, dof=None, normalize=True,
                            include_multidimensional_stimulus_index=False):

        self.set_paradigm(stimulus_range)

        pred = self.predict(stimulus_range, parameters=parameters, weights=weights)

        residuals = data.values[np.newaxis, :, :] - pred.values[:, np.newaxis, :]

        omega_chol = np.linalg.cholesky(omega)
        n_units = data.shape[1]

        n_batches, n_timepoints, n_voxels = residuals.shape
        residuals_2d = ops.reshape(ops.convert_to_tensor(residuals, dtype='float32'),
                                   (n_batches * n_timepoints, n_voxels))
        omega_chol_t = ops.convert_to_tensor(omega_chol, dtype='float32')

        if dof is None:
            ll_flat = mvn_log_prob(residuals_2d, omega_chol_t)
        else:
            ll_flat = mvt_log_prob(residuals_2d, omega_chol_t, dof)

        ll = ops.reshape(ll_flat, (n_batches, n_timepoints))
        if hasattr(ll, 'numpy'):
            ll = ll.numpy()

        ll = pd.DataFrame(ll, index=pd.MultiIndex.from_frame(stimulus_range), columns=data.index).T

        if normalize:
            ll = ll.sub(ll.max(axis=1), axis=0)

        ll = np.exp(ll)

        return ll



class HRFEncodingModel(object):
    """Mixin that equips an encoding model with HRF convolution support."""

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


    def _predict(self, paradigm, parameters, weights):
        """Convolve base predictions with HRF, reapplying amplitude/baseline."""

        n_pars = len(self.parameter_labels)
        # Build modified parameters with baseline=0 and amplitude=1
        parts = [parameters[:, :, i:i+1] for i in range(n_pars)]

        if 'baseline' in self.parameter_labels:
            baseline_idx = self.parameter_labels.index('baseline')
            parts[baseline_idx] = ops.zeros_like(parts[baseline_idx])

        if 'amplitude' in self.parameter_labels:
            amplitude_idx = self.parameter_labels.index('amplitude')
            parts[amplitude_idx] = ops.ones_like(parts[amplitude_idx])

        standardized_parameters = ops.concatenate(parts, axis=2)

        pre_convolve = EncodingModel._predict(
            self, paradigm, standardized_parameters, weights)

        kwargs = {}
        if self.flexible_hrf_parameters:
            for ix, label in enumerate(self.hrf_model.parameter_labels):
                kwargs[label] = parameters[:, :, -len(self.hrf_model.parameter_labels) + ix]

        pred_convolved = self.hrf_model.convolve(pre_convolve, **kwargs)

        if 'amplitude' in self.parameter_labels:
            pred_convolved *= parameters[:, :, amplitude_idx][:, None, :]

        if 'baseline' in self.parameter_labels:
            pred_convolved += parameters[:, :, baseline_idx][:, None, :]

        return pred_convolved

    def _predict_no_hrf(self, paradigm, parameters, weights):
        """Bypass HRF convolution; useful for diagnostics/debugging."""
        return EncodingModel._predict(self, paradigm, parameters, weights)
