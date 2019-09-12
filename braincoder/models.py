import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tqdm.autonotebook import tqdm
import pandas as pd
from .utils import get_rsq, get_r
import scipy.stats as ss
import logging


class EncodingModel(object):

    """

    An encoding model uses the following attributes:

     - parameters: (n_populations, n_pars)
     - weights: (n_populations, n_voxels)

    It can also have the following additional properties:
     - paradigm: (n_timepoints, n_stim_dimensions)
     - data: (n_timepoints, n_voxels)

    It has the following methods:

    -  __init__(self, *args, **kwargs): doesn't do much
    - build_graph(self, paradigm, data, parameters, weights, *args, **kwargs)
        This method is used in situations of
          * simulation
          * fitting
          * predicting

      Paradigm should _always be provided. All the others are optional.

      There are three fit routines:
       - fit_weights()
       - fit_parameters()
       - fit_residuals()


    """

    identity_model = False

    def __init__(self, paradigm=None, data=None, parameters=None,
                 weights=None, verbosity=logging.INFO):

        if paradigm is not None:
            self.paradigm = self._check_input(paradigm, 'paradigm')
        else:
            self.paradigm = None

        if data is not None:
            self.data = self._check_input(data, 'data')
        else:
            self.data = None

        if parameters is not None:
            self.parameters = self._check_input(parameters, 'parameters')
        else:
            self.parameters = None

        if weights is not None:
            self.weights = self._check_input(weights, 'weights')
        else:
            self.weights = None

        self.logger = logging.getLogger(name='EncodingModel logger')
        self.logger.setLevel(logging.INFO)

    def fit_parameters(self, paradigm, data, min_nsteps=100000,
                       ftol=1e-9, progressbar=False,):

        assert(len(data) == len(paradigm)
               ), "paradigm and data should be same length"

        paradigm = self._check_input(paradigm, 'paradigm')
        data = self._check_input(data, 'data')
        init_pars = self.init_parameters(data, paradigm)

        self.build_graph(paradigm, data, init_pars, weights=None)

        with self.graph.as_default():

            self.cost_ = tf.reduce_sum((self.data_ - self.predictions_)**2)

            optimizer = tf.train.AdamOptimizer()
            train = optimizer.minimize(self.cost_)
            init = tf.global_variables_initializer()

            with tf.Session() as session:
                costs = np.ones(min_nsteps) * np.inf
                _ = session.run([init])

                # ftol_ratio = 1 + ftol
                if progressbar:
                    with tqdm(range(min_nsteps)) as pbar:
                        pbar = tqdm(range(min_nsteps))

                        for step in pbar:
                            _, c = session.run(
                                [train, self.cost_])
                            costs[step] = c
                            pbar.set_description(f'Current cost: {c:7g}')

                            if (costs[step - 1] >= c) & (costs[step - 1] - c < ftol):
                                break
                else:
                    for step in range(min_nsteps):
                        _, c, p = session.run(
                            [train, self.cost_, self.parameters_])
                        costs[step] = c
                        if (costs[step - 1] >= c) & (costs[step - 1] - c < ftol):
                            break

                parameters, predictions = session.run(
                    [self.parameters_, self.predictions_])

            costs = pd.Series(costs[:step + 1])
            parameters = pd.DataFrame(
                parameters, columns=self.parameter_labels)

            self.parameters = parameters

            predictions = pd.DataFrame(predictions,
                                       index=data.index,
                                       columns=data.columns)

            return costs, parameters, predictions

    def simulate(self, paradigm=None, parameters=None, weights=None, noise=1.):
        """
        * Parameters should be an array of size M or (M, N),
        where M is the number of parameters and N the number of
        parameter sets.
        * Paradigm should be an array of size N, where N
        is the number of timepoints

        """

        paradigm = self._check_input(paradigm, 'paradigm')
        weights = self._check_input(weights, 'weights')
        parameters = self._check_input(parameters, 'parameters')

        self.logger.info('Size paradigm: {}'.format(paradigm.shape))
        self.logger.info('Size parameters: {}'.format(parameters.shape))

        if hasattr(weights, 'shape'):
            self.logger.info('Size weights: {}'.format(weights.shape))

        self.build_graph(paradigm=paradigm, data=None,
                         parameters=parameters, weights=weights)

        noise = np.array(noise)

        if noise.ndim < 2:
            noise = np.ones((1, self.n_voxels)) * noise

        with self.graph.as_default():
            init = tf.global_variables_initializer()
            with tf.Session() as session:
                _ = session.run(init)
                simulated_data_ = session.run(self.noisy_prediction_,
                                              feed_dict={self.noise_level_: noise})

        if hasattr(weights, 'columns'):
            columns = weights.columns
        else:
            columns = np.arange(self.n_voxels)

        simulated_data = pd.DataFrame(simulated_data_,
                                      index=paradigm.index, columns=columns)

        return simulated_data

    def build_graph(self,
                    paradigm,
                    data=None,
                    parameters=None,
                    weights=None):

        self.graph = tf.Graph()

        (n_populations, n_parameters, n_voxels,
         n_timepoints, n_stim_dimensions) = \
            self._get_graph_properties(
            paradigm, data, weights, parameters)

        self.n_voxels = n_voxels
        weights = self._check_input(weights, 'weights')
        paradigm = self._check_input(paradigm, 'paradigm')
        data = self._check_input(data, 'data')
        parameters = self._check_input(parameters, 'parameters')

        self.logger.info((n_populations, n_parameters,
                          n_voxels, n_timepoints, n_stim_dimensions))

        with self.graph.as_default():

            self.paradigm_ = tf.constant(paradigm.astype(np.float32),
                                         name='paradigm')

            self.parameters_ = tf.get_variable(initializer=parameters.astype(np.float32),
                                               name='parameters',
                                               dtype=tf.float32)
            self.basis_predictions_ = self.build_basis_function(
                self.graph, self.parameters_, self.paradigm_)

            weights_shape = (
                n_populations, n_voxels) if weights is None else None

            self.logger.info('N populations {}, N voxels: {}'.format(
                n_populations, n_voxels))

            if issubclass(self.__class__, IsolatedPopulationsModel):
                self.weights_ = None
            else:
                self.weights_ = tf.get_variable(initializer=weights,
                                                shape=weights_shape,
                                                name='weights',
                                                dtype=tf.float32)

            self.logger.info('Size predictions: {}'.format(
                self.basis_predictions_))

            if issubclass(self.__class__, IsolatedPopulationsModel):
                self.predictions_ = self.basis_predictions_
            else:
                self.predictions_ = tf.tensordot(self.basis_predictions_,
                                                 self.weights_, (1, 0))

            # Simulation
            self.noise_level_ = tf.get_variable(
                dtype=tf.float32, shape=(1, n_voxels), name='noise')

            self.noise_ = tf.random.normal(shape=(n_timepoints, n_voxels),
                                           mean=0.0,
                                           stddev=self.noise_level_,
                                           dtype=tf.float32)

            self.noisy_prediction_ = self.predictions_ + self.noise_

            if data is not None:
                self.data_ = tf.constant(data.values, name='data')
                self.residuals_ = self.data_ - self.predictions_
                self.cost_ = tf.squeeze(tf.reduce_sum(self.residuals_**2))
                self.mean_data_ = tf.reduce_mean(self.data_, 0)
                self.ssq_data_ = tf.reduce_sum(
                    (self.data_ - tf.expand_dims(self.mean_data_, 0))**2, 0)
                self.rsq_ = 1 - (self.cost_ / self.ssq_data_)

    def build_residuals_graph(self, distance_matrix=None):

        with self.graph.as_default():
            self.rho_trans_ = tf.get_variable(
                'rho_trans', shape=(), dtype=tf.float32)
            self.rho_ = tf.math.sigmoid(self.rho_trans_, name='rho')

            self.tau_trans_ = tf.get_variable('tau_trans',
                                              initializer=_inverse_softplus_tensor(tfp.stats.stddev(self.data_)[:, tf.newaxis]))

            self.tau_ = _softplus_tensor(self.tau_trans_, name='tau')

            self.sigma2_trans_ = tf.Variable(
                _inverse_softplus(1e-9), dtype=tf.float32, name='sigma2_trans')
            self.sigma2_ = _softplus_tensor(
                self.sigma2_trans_, name='sigma2')

            if distance_matrix is not None:
                self.alpha_trans_ = tf.get_variable(
                    name='alpha_trans', shape=(), dtype=tf.float32)
                self.alpha_ = tf.math.sigmoid(self.alpha_trans_, name='alpha')

                self.beta_ = tf.get_variable(
                    name='beta', shape=(), dtype=tf.float32)
                D = tf.constant(distance_matrix.astype(np.float32))

                self.sigma_D = self.alpha_ * tf.exp(-self.beta_ * D) * tf.tensordot(self.tau_,
                                                                                    tf.transpose(
                                                                                        self.tau_),
                                                                                    axes=1)

                self.sigma0_ = self.sigma_D + \
                    self.rho_ * tf.tensordot(self.tau_,
                                             tf.transpose(self.tau_),
                                             axes=1) + \
                    (1 - self.rho_ - self.alpha_) * tf.linalg.tensor_diag(tf.squeeze(self.tau_**2)) + \
                    self.sigma2_ * tf.squeeze(tf.tensordot(self.weights_,
                                                           self.weights_, axes=(-2, -2)))

            else:
                self.sigma0_ = self.rho_ * tf.tensordot(self.tau_,
                                                        tf.transpose(
                                                            self.tau_),
                                                        axes=1) + \
                    (1 - self.rho_) * tf.linalg.tensor_diag(tf.squeeze(self.tau_**2)) + \
                    self.sigma2_ * tf.squeeze(tf.tensordot(self.weights_,
                                                           self.weights_, axes=(-2, -2)))

            self.empirical_covariance_matrix_ = tfp.stats.covariance(
                self.data_)

            self.lambd_ = tf.get_variable(name='lambda',
                                          shape=())

            self.sigma_ = self.lambd_ * self.sigma0_ +  \
                (1 - self.lambd_) * self.empirical_covariance_matrix_

            self.residual_dist = tfd.MultivariateNormalFullCovariance(
                tf.zeros(self.data_.shape[1]),
                self.sigma_)
            self.likelihood_ = self.residual_dist.log_prob(self.residuals_)

    def _check_input(self, par, name=None):

        if par is None:
            if name is not None:
                if hasattr(self, name):
                    return getattr(self, name)
                else:
                    return None
            else:
                return None

        if name == 'parameters':
            if hasattr(par, 'values'):
                return par.values
            else:
                return par
        else:
            if name == 'paradigm':
                self.paradigm = pd.DataFrame(par).astype(np.float32)
            return pd.DataFrame(par.astype(np.float32))

    def _get_graph_properties(self, paradigm, data, weights, parameters):

        n_pars = self.n_parameters

        if data is not None:
            n_voxels = data.shape[-1]
        elif issubclass(self.__class__, IsolatedPopulationsModel):
            n_voxels = parameters.shape[0]
        else:
            n_voxels = weights.shape[-1]

        if issubclass(self.__class__, IsolatedPopulationsModel):
            n_populations = n_voxels
        else:
            if parameters is not None:
                n_populations = parameters.shape[0]
            else:
                n_populations = self.n_populations

        n_timepoints = paradigm.shape[0]
        n_stim_dimensions = paradigm.shape[1]

        return n_populations, n_pars, n_voxels, n_timepoints, n_stim_dimensions

    def get_basis_function_activations(self, paradigm=None, parameters=None):

        paradigm = self._check_input(paradigm, 'paradigm')
        parameters = self._check_input(parameters, 'parameters')

        with self.graph.as_default():
            with tf.Session() as session:
                basis_predictions = session.run(self.basis_predictions_, feed_dict={
                    self.paradigm_: paradigm.values,
                    self.parameters_: parameters})

        return pd.DataFrame(basis_predictions, index=paradigm.index)

    def get_predictions(self, paradigm=None, parameters=None, weights=None):

        paradigm = self._check_input(paradigm, name='paradigm')
        weights = self._check_input(weights, name='weights')
        parameters = self._check_input(parameters, name='parameters')

        self.build_graph(paradigm=paradigm,
                         parameters=parameters, weights=weights)

        with self.graph.as_default():
            with tf.Session() as session:
                feed_dict = {self.paradigm_: paradigm.values,
                             self.parameters_: parameters}

                if not issubclass(self.__class__, IsolatedPopulationsModel):
                    feed_dict[self.weights_] = weights.values
                    columns = weights.columns
                else:
                    columns = None

                predictions = session.run(
                    self.predictions_, feed_dict=feed_dict)

        return pd.DataFrame(predictions, index=paradigm.index,
                            columns=columns)

    def fit_weights(self, paradigm, data, parameters=None, l2_cost=0.0):

        if issubclass(self.__class__, IsolatedPopulationsModel):
            raise Exception('This is a model with exactly one population per feature. This means '
                            'you can  not meaningfully fit the weights')

        paradigm = self._check_input(paradigm, name='paradigm')
        weights = self._check_input(data, name='weights')
        parameters = self._check_input(parameters, name='parameters')

        self.build_graph(paradigm=paradigm, data=data,
                         parameters=parameters, weights=None)

        with self.graph.as_default():
            init = tf.global_variables_initializer()
            ols_solver = tf.linalg.lstsq(self.basis_predictions_,
                                         self.data_,
                                         l2_regularizer=l2_cost)

            with tf.Session() as session:
                _ = session.run(init)
                weights = session.run(ols_solver)

        self.weights = pd.DataFrame(weights)

        return self.weights

    def get_rsq(self, data, paradigm=None, parameters=None, weights=None):
        predictions = self.get_predictions(
            paradigm=paradigm, parameters=parameters, weights=weights)
        predictions.index = data.index
        rsq = get_rsq(data, predictions)
        return rsq

    def get_r(self, data, paradigm=None, parameters=None, weights=None):
        predictions = self.get_predictions(
            paradigm=paradigm, parameters=parameters, weights=weights)
        r = get_r(data, predictions)
        return r

    def fit_residuals(self,
                      paradigm=None,
                      data=None,
                      lambd=1.,
                      distance_matrix=None,
                      rho_init=1e-9,
                      min_nsteps=100000,
                      ftol=1e-12,
                      also_fit_weights=False,
                      progressbar=True):

        paradigm = self._check_input(paradigm, 'paradigm')
        data = self._check_input(data, 'data')

        self.build_graph(paradigm, data)
        self.build_residuals_graph(distance_matrix=distance_matrix)

        with self.graph.as_default():
            optimizer = tf.train.AdamOptimizer()
            cost = -tf.reduce_sum(self.likelihood_)
            var_list = [self.tau_trans_, self.rho_trans_, self.sigma2_trans_]
            if also_fit_weights:
                var_list.append(self.weights_)

            if distance_matrix is not None:
                var_list.append(self.alpha_trans_)
                var_list.append(self.beta_)

            train = optimizer.minimize(cost, var_list=var_list)

            costs = []

            init = tf.global_variables_initializer()

            with tf.Session() as session:
                session.run(init)

                self.weights_.load(self.weights.values, session)
                self.rho_trans_.load(rho_init, session)
                self.lambd_.load(lambd, session)

                if distance_matrix is not None:
                    self.alpha_trans_.load(-1, session)
                    self.beta_.load(1., session)

                costs = np.ones(min_nsteps) * -np.inf

                if progressbar:
                    with tqdm(range(min_nsteps)) as pbar:
                        for step in pbar:
                            _, c, rho_, sigma2, weights = session.run(
                                [train, cost, self.rho_, self.sigma2_, self.weights_],)
                            costs[step] = c
                            pbar.set_description(f'Current cost: {c:7g}')

                            if (costs[step - 1] >= c) & (costs[step - 1] - c < ftol):
                                break
                else:
                    for step in range(min_nsteps):
                        _, c, rho_, sigma2, weights = session.run(
                            [train, cost, self.rho_, self.sigma2_, self.weights_],)
                        costs[step] = c
                        if (costs[step - 1] >= c) & (costs[step - 1] - c < ftol):
                            break

                costs = costs[:step+1]
                self.rho = session.run(self.rho_)
                self.tau = session.run(self.tau_)
                self.omega = session.run(self.sigma_)
                self.sigma2 = session.run(self.sigma2_)
                predictions = session.run(self.predictions_)

                if distance_matrix is not None:
                    self.alpha = session.run(self.alpha_)
                    self.beta = session.run(self.beta_)

                predictions = pd.DataFrame(
                    predictions, index=data.index, columns=data.columns)

                self.ols_weights = self.weights.copy()

                if also_fit_weights:
                    self.weights = pd.DataFrame(session.run(self.weights_),
                                                index=self.weights.index,
                                                columns=self.weights.columns)

        return costs, (self.rho, self.sigma2, self.tau, self.omega), predictions

    def build_decoding_graph(self, data, stimulus_range, normalize=False, precision=1e-6):
        """
        data is (n_timepoints, n_voxels)
        stimulus_range is (n_stim_dimensions, n_potential_stimuli)

        """

        decode_graph = tf.Graph()

        if not hasattr(self, 'weights') or not hasattr(self, 'omega'):
            raise Exception('Please firs fit weights and/or residuals')

        self.decode_nodes = {}
        with decode_graph.as_default():

            data_to_decode = tf.constant(data, name='data',)

            # n_dimensions x n_stimuli
            stimulus = tf.get_variable(
                name='stimulus', initializer=stimulus_range, dtype=tf.float32)

            # n_dimensions x n_voxels
            weights = tf.get_variable(
                name='weights', initializer=self.weights.values)

            # n_populations x stim_dimension
            parameters = tf.get_variable(
                name='parameters', initializer=self.parameters)

            # n_dimensions x n_stimuli
            basis_functions = self.build_basis_function(
                decode_graph, parameters, stimulus)

            #  n_stimuli x n_voxels
            hypothetical_timeseries = tf.tensordot(
                basis_functions, weights, (1, 0), name='hypothetical_timeseries')

            # # n_timepoints x n_stimuli x n_vox
            residuals = data_to_decode[:, tf.newaxis, :] - \
                hypothetical_timeseries[tf.newaxis, ...]

            mvn = tfd.MultivariateNormalFullCovariance(
                loc=tf.zeros(data.shape[1]), covariance_matrix=self.omega)

            # n_timepoints x n_stimuli
            self.decode_pdf_log_ = mvn.log_prob(residuals, name='pdf')

            if normalize:
                max_ll = tf.reduce_max(self.decode_pdf_log_, 1)
                thr0 = tf.log(
                    precision) - tf.log(tf.cast(self.decode_pdf_log_.shape[1], tf.float32))
                nonzeromask = self.decode_pdf_log_ - \
                    max_ll[:, tf.newaxis] > thr0
                self.decode_pdf_ = tf.cast(
                    nonzeromask, tf.float32) * tf.exp(self.decode_pdf_log_ - max_ll[:, tf.newaxis])

            else:
                self.decode_pdf_ = tf.exp(self.decode_pdf_log_)

            # n_timepoints x n_stim_dimensions
            self.decode_map_ = tf.gather(
                stimulus, tf.math.argmax(self.decode_pdf_log_, 1), axis=0)

            # n_timepoints x n_stim_dimensions x n_stimuli
            summed_hist = (self.decode_map_[..., tf.newaxis] - tf.transpose(stimulus)[
                           tf.newaxis, ...])**2 * self.decode_pdf_[:, tf.newaxis, :]

            self.decode_sd_ = tf.sqrt(tf.reduce_sum(
                summed_hist, -1) * tf.reduce_sum(self.decode_pdf_, -1))

            cdf = tf.cumsum(self.decode_pdf_, 1) / \
                tf.reduce_sum(self.decode_pdf_, 1)[:, tf.newaxis]
            tmp_ix = tf.where(cdf >= 0.025)
            lower_ix = tf.segment_min(tmp_ix[:, 1], tmp_ix[:, 0])

            tmp_ix = tf.where(cdf >= 0.975)
            upper_ix = tf.segment_min(tmp_ix[:, 1], tmp_ix[:, 0])

            self.lower_ci_ = tf.gather(stimulus, lower_ix, axis=0)
            self.upper_ci_ = tf.gather(stimulus, upper_ix, axis=0)

        return decode_graph

    def _check_stimulus_range(self, stimulus_range=None):
        if stimulus_range is None:
            stimulus = np.linspace(-5, 5, 1000)
        elif type(stimulus_range) is tuple:
            stimulus = np.linspace(stimulus_range[0], stimulus_range[1], 1000)
        else:
            stimulus = stimulus_range

        if stimulus.ndim == 1:
            stimulus = stimulus[:, np.newaxis]

        return stimulus.astype(np.float32)

    def get_stimulus_posterior(self, data,
                               stimulus_range=None,
                               normalize=False):

        stimulus_range = self._check_stimulus_range(stimulus_range)
        data = self._check_input(data, 'data')

        decode_graph = self.build_decoding_graph(
            data, stimulus_range, normalize)

        with decode_graph.as_default():
            init = tf.global_variables_initializer()
            with tf.Session() as session:
                session.run(init)
                pdf = session.run(self.decode_pdf_)
                map_ = session.run(self.decode_map_)
                sd = session.run(self.decode_sd_)
                lower_ci, higher_ci = session.run(
                    [self.lower_ci_, self.upper_ci_])

        return pdf, map_, sd, (lower_ci, higher_ci)


class IsolatedPopulationsModel(object):
    """
    This subclass of EncodingModel assumes every population maps onto
    exactly one voxel.
    """
    pass


class GLMModel(EncodingModel):

    n_parameters = 0
    n_populations = None

    def __init__(self, parameters=None, verbosity=logging.INFO):
        """
        parameters is a NxD or  array, where N is the number
        of basis functions and P is the number of parameters
        """
        return super().__init__()

    def build_graph(self,
                    paradigm,
                    data=None,
                    weights=None,
                    parameters=None):

        parameters = self._get_dummy_parameters(
            paradigm=paradigm)
        super().build_graph(paradigm, data, parameters, weights)

    def simulate(self, paradigm=None, parameters=None, weights=None, noise=1.):
        """
        paradigm is a N or NxM matrix, where N is the number
        of time points and M is the number of stimulus dimensions.
        weights is a BxV matrix, where B is the number
        of basis functions and V is the number of
        features (e.g., voxels, time series).
        Noise is either a scalar for equal noise across voxels
        or a V-array with the amount of noise for every voxel.

        """

        if parameters is not None:
            raise Exception(
                'GLMModel has no meaningful parameters (only weights)')

        parameters = self._get_dummy_parameters(paradigm)

        return super().simulate(paradigm=paradigm,
                                parameters=parameters, weights=weights, noise=noise)

    def fit_weights(self, paradigm, data, l2_cost=0.0):
        parameters = self._get_dummy_parameters(paradigm)

        return super().fit_weights(paradigm=paradigm,
                                   data=data,
                                   parameters=parameters,
                                   l2_cost=l2_cost)

    def get_predictions(self, paradigm=None, parameters=None, weights=None):
        parameters = self._get_dummy_parameters(paradigm=paradigm)

        return super().get_predictions(paradigm, parameters, weights)

    def build_basis_function(self, graph, parameters, x):
        # time x basis_functions
        with graph.as_default():
            basis_predictions_ = x

        return basis_predictions_

    def _get_graph_properties(self, paradigm, data, weights, parameters):
        _, n_pars, n_voxels, n_timepoints, n_stim_dimensions = super(
        )._get_graph_properties(paradigm, data, weights, parameters)

        n_populations = paradigm.shape[1]

        return n_populations, n_pars, n_voxels, n_timepoints, n_stim_dimensions

    def _check_input(self, par, name=None):
        if (par is None) and (name == 'parameters'):
            return self._get_dummy_parameters()
        else:
            return super()._check_input(par, name)

    def _get_dummy_parameters(self, paradigm=None):
        paradigm = self._check_input(paradigm, 'paradigm')
        return np.zeros((paradigm.shape[1], 0))


class StickModel(EncodingModel):
    n_populations = None
    n_parameters = 1

    def build_basis_function(self, graph, parameters, x):
        self.logger.warning('Note that first parameter of StickModel '
                            'corresponds to the Intercept (baseline) and '
                            'is NOT used.')

        with graph.as_default():
            basis_predictions_ = tf.cast(
                tf.equal(x, parameters[tf.newaxis, 1:, 0]), tf.float32)
            intercept = tf.ones((basis_predictions_.shape[0], 1))
            basis_predictions_ = tf.concat((intercept, basis_predictions_), 1)

            return basis_predictions_


class Discrete1DModel(EncodingModel):

    def __init__(self,
                 basis_values=None,
                 paradigm=None,
                 weights=None,
                 parameters=None):
        """
        basis_values is a 2d Nx2 array. The first columns are
        coordinates on the line, the second column contains the
        intesity of the basis functions at that value.
        """

        if parameters is None:
            parameters = np.ones((0, 0, 0))

        if parameters.ndim == 2:
            parameters = parameters[:, :, np.newaxis]

        self.weights = weights
        self.paradigm = paradigm
        self.parameters = parameters


class GaussianReceptiveFieldModel(EncodingModel):

    n_parameters = 4
    parameter_labels = ['mu', 'sd', 'amplitude', 'baseline']

    def __init__(self, positive_amplitudes=True):

        super().__init__()
        self.positive_amplitudes = positive_amplitudes

    def build_graph(self,
                    paradigm,
                    data=None,
                    parameters=None,
                    weights=None):

        super().build_graph(paradigm=paradigm,
                            data=data,
                            parameters=parameters,
                            weights=weights)

    def fit_parameters(self,
                       paradigm,
                       data,
                       min_nsteps=100000,
                       ftol=1e-6,
                       progressbar=False):

        costs, parameters, predictions = super().fit_parameters(paradigm,
                                                                data,
                                                                min_nsteps,
                                                                ftol,
                                                                progressbar=progressbar)

        parameters['sd'] = _softplus(parameters['sd'])

        if self.positive_amplitudes:
            parameters['amplitude'] = _softplus(
                parameters['amplitude'])

            self.parameters = parameters

        return costs, parameters, predictions

    def build_basis_function(self, graph, parameters, x):
        with graph.as_default():
            self.mu_ = self.parameters_[:, 0]
            self.sd_ = tf.math.softplus(parameters[:, 1])

            if self.positive_amplitudes:
                self.amplitude__ = parameters[:, 2]
                self.amplitude_ = tf.math.softplus(self.amplitude__)
            else:
                self.amplitude_ = parameters[:, 2]
            self.baseline_ = parameters[:, 3]

            basis_predictions_ = self.baseline_[tf.newaxis, :] + \
                norm(x, self.mu_[tf.newaxis, :],
                     self.sd_[tf.newaxis, :]) *  \
                self.amplitude_[tf.newaxis, :]

            return basis_predictions_

    def init_parameters(self, data, paradigm):
        baselines = data.min(0)
        data_ = data - baselines
        mus = (data_.values * paradigm.values).sum(0) / data_.values.sum(0)
        sds = (data_.values * (paradigm.values - mus)
               ** 2).sum(0) / data_.values.sum(0)
        sds = np.sqrt(sds)
        amplitudes = data_.max(0)

        (n_populations, n_parameters, n_voxels,
         n_timepoints, n_stim_dimensions) = \
            self._get_graph_properties(
            paradigm, data, None, None)

        pars = np.zeros(
            (n_populations, self.n_parameters), dtype=np.float32)

        pars[:, 0] = mus
        pars[:, 1] = sds
        pars[:, 2] = amplitudes
        pars[:, 3] = baselines

        return pars

    def simulate(self, paradigm=None, parameters=None, noise=1.):

        paradigm = self._check_input(paradigm, 'paradigm')
        parameters = self._check_input(parameters, 'parameters').copy()

        parameters[:, 1] = _inverse_softplus(parameters[:, 1])

        if self.positive_amplitudes:
            parameters[:, 2] = _inverse_softplus(parameters[:, 2])

        data = super().simulate(paradigm, parameters, noise=noise)
        return data

    def _get_graph_properties(self, paradigm, data, weights, parameters):

        if data is not None:
            self.n_populations = data.shape[-1]

        if parameters is not None:
            self.n_populations = parameters.shape[0]

        n_populations = self.n_populations
        n_pars = self.n_parameters

        n_voxels = n_populations

        n_timepoints = paradigm.shape[0]
        n_stim_dimensions = paradigm.shape[1]

        return n_populations, n_pars, n_voxels, n_timepoints, n_stim_dimensions

    def to_stickmodel(self, basis_stimuli=None):

        if basis_stimuli is None:
            basis_stimuli = np.unique(self.paradigm.values)

        basis_stimuli = np.hstack(([0], basis_stimuli)).astype(np.float32)

        weights = np.zeros((len(basis_stimuli), self.parameters.shape[0]))
        weights[0, :] = self.parameters['baseline']

        conversion_graph = tf.Graph()

        mu = self.parameters['mu'].values[np.newaxis, :]
        sd = self.parameters['sd'].values[np.newaxis, :]
        x = basis_stimuli[1:, np.newaxis]
        amplitude = self.parameters['amplitude'].values[np.newaxis, :]

        with conversion_graph.as_default():
            w_tf = amplitude * norm(x, mu, sd)

            with tf.Session() as session:
                weights[1:, :] = session.run(w_tf)

        sm = StickModel(self.paradigm,
                        self.data,
                        basis_stimuli[:, np.newaxis],
                        weights)
        sm.build_graph(paradigm=sm.paradigm, parameters=sm.parameters,
                       data=sm.data, weights=sm.weights)

        return sm


class VoxelwiseGaussianReceptiveFieldModel(GaussianReceptiveFieldModel, IsolatedPopulationsModel):
    pass


def _softplus(x):
    return np.log(1 + np.exp(x))


def _softplus_tensor(x, name=None):
    return tf.log(1 + tf.exp(x), name=name)


def _logit(x):
    """ Computes the logit function, i.e. the logistic sigmoid inverse. """
    return - np.log(1. / x - 1.)


def _logit_tensor(x):
    """ Computes the logit function, i.e. the logistic sigmoid inverse. """
    return - tf.log(1. / x - 1.)


def _inverse_softplus_tensor(x):
    return tf.log(tf.exp(x) - 1)


def _inverse_softplus(x):
    return np.log(np.exp(x) - 1)


def norm(x, mu, sigma):
    # Z = (2. * np.pi * sigma**2.)**0.5
    kernel = tf.math.exp(-.5 * (x - mu)**2. / sigma**2)
    return kernel
