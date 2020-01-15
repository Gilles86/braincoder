import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tqdm.autonotebook import tqdm
import pandas as pd
from .utils import get_rsq, get_r
import scipy.stats as ss
import logging
from tensorflow.math import softplus as _softplus_tensor


class EncodingModel(object):

    """

    An encoding model uses the following attributes:

     - parameters: (n_populations, n_pars)
     - w) & (x < right_bins)eights: (n_populations, n_voxels)

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
          NOTE: build_graph already assumes _parameters_ has been transformed
          when necessary

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
            self.parameters = pd.DataFrame(parameters,
                                           columns=self.get_parameter_labels(parameters))
        else:
            self.parameters = None

        if weights is not None:
            self.weights = self._check_input(weights, 'weights')
        else:
            self.weights = None

        self.logger = logging.getLogger(name='EncodingModel logger')
        self.logger.setLevel(logging.INFO)

    def fit_parameters(self, paradigm, data, init_pars=None, max_n_iterations=100000,
                       atol=1e-9, learning_rate=0.001, patience=1000, progressbar=False,
                       also_fit_weights=False, l2_cost=0.0):

        assert(len(data) == len(paradigm)
               ), "paradigm and data should be same length"

        data_cols = pd.DataFrame(data).columns

        paradigm = self._check_input(paradigm, 'paradigm')
        data = self._check_input(data, 'data')

        if init_pars is None:
            init_pars = self.init_parameters(data, paradigm)

        if not issubclass(self.__class__, IsolatedPopulationsModel) and not hasattr(self, 'weights'):
            self.fit_weights(paradigm, data, parameters=init_pars, l2_cost=l2_cost)

        init_pars = self.inverse_transform_parameters(init_pars)

        self.build_graph(paradigm, data, init_pars, weights=None)

        with self.graph.as_default():

            self.cost_ = tf.reduce_sum((self.data_ - self.predictions_)**2)

            if l2_cost > 0:
                self.cost_ += l2_cost * tf.reduce_sum(self.weights_**2)

            var_list = [self.parameters_]
            if also_fit_weights:
                var_list += [self.weights_]

            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            train = optimizer.minimize(self.cost_, var_list=var_list)
            init = tf.global_variables_initializer()

            with tf.Session() as session:
                costs = np.ones(max_n_iterations) * np.inf
                _ = session.run([init])


                patience_counter = 0
                if progressbar:
                    pbar = tqdm(total=max_n_iterations)
                for step in range(max_n_iterations):
                    _, c = session.run(
                        [train, self.cost_])
                    #if also_fit_weights:
                        #_ = session.run(ols_solver)

                    costs[step] = c
                    if progressbar:
                        pbar.update(1)
                        pbar.set_description(f'Current cost: {c:7g}')

                    if costs[step - 1] - c < atol:
                        patience_counter += 1
                    if patience_counter == patience:
                        if progressbar:
                            pbar.close()
                        break

                parameters, predictions = session.run(
                    [self.parameters_, self.predictions_])

                if also_fit_weights:
                    weights = session.run(self.weights_)

            costs = pd.Series(costs[:step + 1])

            parameters = self.transform_parameters(parameters)

            self.parameters = parameters

            if also_fit_weights:
                self.weights = pd.DataFrame(weights,
                        index=self.parameters.index,
                        columns=data.columns)

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

        self.logger.info('Size paradigm: {}'.format(paradigm.shape))

        if parameters is None:
            parameters = self.parameters

        parameters = self.inverse_transform_parameters(parameters.copy())
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

    def build_residuals_graph(self, tau_init=None, distance_matrix=None, residual_dist='gaussian'):

        self.residual_dist_type = residual_dist

        with self.graph.as_default():
            self.rho_trans_ = tf.get_variable(
                'rho_trans', shape=(), dtype=tf.float32)
            self.rho_ = tf.math.sigmoid(self.rho_trans_, name='rho') * 0.9

            if tau_init is None:
                self.tau_trans_ = tf.get_variable('tau_trans',
                                                  initializer=_inverse_softplus_tensor(tfp.stats.stddev(self.data_)[:, tf.newaxis]))
            else:
                self.tau_trans_ = tf.get_variable('tau_trans',
                                              initializer=_inverse_softplus_tensor(tau_init))

            self.tau_ = _softplus_tensor(self.tau_trans_, name='tau') + 1e-6

            self.sigma2_trans_ = tf.Variable(
                _inverse_softplus(1e-9), dtype=tf.float32, name='sigma2_trans')
            self.sigma2_ = _softplus_tensor(
                self.sigma2_trans_, name='sigma2')

            if distance_matrix is not None:
                self.alpha_trans_ = tf.get_variable(
                    name='alpha_trans', shape=(), dtype=tf.float32)
                self.alpha_ = tf.math.sigmoid(
                    self.alpha_trans_, name='alpha') * 0.99

                self.beta_trans_ = tf.get_variable(
                    name='beta_trans', shape=(), dtype=tf.float32)
                self.beta_ = _softplus_tensor(self.beta_trans_, name='beta') + 1e-1

                D = tf.constant(distance_matrix.astype(np.float32))

                self.sigma_D = self.rho_ * self.alpha_ * tf.exp(-self.beta_ * D) * tf.tensordot(self.tau_,
                                                                                                tf.transpose(
                                                                                                    self.tau_),
                                                                                                axes=1)

                self.sigma0_ = self.sigma_D + \
                    self.rho_ * (1 - self.alpha_) * tf.tensordot(self.tau_,
                                                                 tf.transpose(
                                                                     self.tau_),
                                                                 axes=1) + \
                    (1 - self.rho_) * tf.linalg.tensor_diag(tf.squeeze(self.tau_**2)) + \
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

            if residual_dist == 'gaussian':
                self.residual_dist = tfd.MultivariateNormalFullCovariance(
                    tf.zeros(self.data_.shape[1]),
                    self.sigma_,
                    allow_nan_stats=False)
            elif residual_dist == 't':
                self.dof_trans_ = tf.get_variable(
                    'dof_trans', dtype=tf.float32, initializer=_inverse_softplus(8.).astype(np.float32))
                self.dof_ = _softplus_tensor(
                    self.dof_trans_, name='dof') + 2. + 1e-6

                self.sigma_t_ = (self.dof_ - 2) * self.sigma_ / self.dof_
                chol = tf.cholesky(self.sigma_t_)
                self.residual_dist = tfd.MultivariateStudentTLinearOperator(df=self.dof_,
                                                                            loc=tf.zeros(
                                                                                self.data_.shape[1]),
                                                                            scale=tf.linalg.LinearOperatorLowerTriangular(chol))

            else:
                raise NotImplementedError(
                    f'{residual_dist} is not implemented as a residual distribution')
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
            raise Exception('Use self.transform_parameters')

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

        if parameters is None:
            parameters = self.parameters

        parameters = self.inverse_transform_parameters(parameters)

        with self.graph.as_default():
            with tf.Session() as session:
                basis_predictions = session.run(self.basis_predictions_, feed_dict={
                    self.paradigm_: paradigm.values,
                    self.parameters_: parameters})

        return pd.DataFrame(basis_predictions, index=paradigm.index)

    def get_predictions(self, paradigm=None, parameters=None, weights=None):

        paradigm = self._check_input(paradigm, name='paradigm')
        weights = self._check_input(weights, name='weights')

        if parameters is None:
            parameters = self.parameters

        parameters = self.inverse_transform_parameters(parameters)

        self.build_graph(paradigm=paradigm,
                         parameters=parameters, weights=weights)

        with self.graph.as_default():
            with tf.Session() as session:
                feed_dict = {self.paradigm_: paradigm,
                             self.parameters_: parameters}

                if not issubclass(self.__class__, IsolatedPopulationsModel):
                    feed_dict[self.weights_] = weights.values
                    columns = weights.columns
                else:
                    columns = None

                predictions = session.run(
                    self.predictions_, feed_dict=feed_dict)

        return pd.DataFrame(predictions, columns=columns)

    def fit_weights(self, paradigm, data, parameters=None, l2_cost=0.0):

        if issubclass(self.__class__, IsolatedPopulationsModel):
            raise Exception('This is a model with exactly one population per feature. This means '
                            'you can  not meaningfully fit the weights')

        paradigm = self._check_input(paradigm, name='paradigm')
        weights = self._check_input(data, name='weights')

        if parameters is None:
            parameters = self.parameters

        parameters = self.inverse_transform_parameters(parameters)

        # n_populations, n_pars, n_voxels, n_timepoints,  \
        # n_stim_dimensions = self._get_graph_properties(paradigm

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

        self.weights = pd.DataFrame(weights, index=self.parameters.index, columns=data.columns)

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
                      residual_dist='gaussian',
                      lambd=1.,
                      distance_matrix=None,
                      rho_init=0.1,
                      tau_init=None,
                      alpha_init=.5,
                      beta_init=.5,
                      max_n_iterations=100000,
                      patience=1000,
                      atol=1e-12,
                      also_fit_weights=False,
                      progressbar=True):

        paradigm = self._check_input(paradigm, 'paradigm')
        data = self._check_input(data, 'data')

        parameters = self.parameters
        if parameters is not None:
            parameters = self.transform_parameters(self.parameters.copy())

        self.build_graph(paradigm, data, parameters=parameters)
        self.build_residuals_graph(
            distance_matrix=distance_matrix, residual_dist=residual_dist,
        tau_init=tau_init)

        with self.graph.as_default():
            optimizer = tf.train.AdamOptimizer()
            cost = -tf.reduce_sum(self.likelihood_)
            var_list = [self.tau_trans_, self.rho_trans_, self.sigma2_trans_]
            if also_fit_weights:
                var_list.append(self.weights_)

            if distance_matrix is not None:
                var_list.append(self.alpha_trans_)
                var_list.append(self.beta_trans_)
            if residual_dist == 't':
                var_list.append(self.dof_trans_)

            train = optimizer.minimize(cost, var_list=var_list)

            costs = []

            init = tf.global_variables_initializer()

            with tf.Session() as session:
                session.run(init)

                self.weights_.load(self.weights.values, session)
                self.rho_trans_.load(_logit(rho_init / 0.9), session)
                self.lambd_.load(lambd, session)

                if distance_matrix is not None:
                    self.alpha_trans_.load(_logit(alpha_init / 0.99), session)
                    self.beta_trans_.load(_inverse_softplus(beta_init - 0.1), session)

                costs = np.ones(max_n_iterations) * -np.inf

                patience_counter = 0

                if progressbar:
                    pbar = tqdm(total=max_n_iterations)
                for step in range(max_n_iterations):
                    _, c, rho_, sigma2, weights = session.run(
                        [train, cost, self.rho_, self.sigma2_, self.weights_],)
                    if distance_matrix is not None:
                        alpha, beta = session.run([self.alpha_, self.beta_])

                    costs[step] = c
                    if progressbar:
                        pbar.update(1)
                        if distance_matrix is None:
                            pbar.set_description(f'Current cost: {c:7g}, rho:{rho_:.3f}')
                        else:
                            pbar.set_description(f'Current cost: {c:7g}, rho:{rho_:0.3f}, alpha: {alpha:0.3f}, beta: {beta:0.3f}')


                    if (costs[step - 1] - c < atol):
                        patience_counter += 1
                    if patience_counter == patience:
                        if progressbar:
                            pbar.close()
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

                if residual_dist == 't':
                    self.dof = session.run(self.dof_)

                predictions = pd.DataFrame(
                    predictions, index=data.index, columns=data.columns)

                self.ols_weights = self.weights.copy()

                if also_fit_weights:
                    self.weights = pd.DataFrame(session.run(self.weights_),
                                                index=self.weights.index,
                                                columns=self.weights.columns)

        return costs, (self.rho, self.sigma2, self.tau, self.omega), predictions

    def build_decoding_graph(self, data, stimulus_range, normalize=False, precision=1e-6,
                             residual_dist=None):
        """
        data is (n_timepoints, n_voxels)
        stimulus_range is (n_stim_dimensions, n_potential_stimuli)

        """

        decode_graph = tf.Graph()

        if residual_dist is None:
            residual_dist = self.residual_dist_type

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
            if self.parameters is None:
                parameters = tf.get_variable(
                    name='parameters', initializer=self._get_dummy_parameters(),
                    dtype=tf.float32)
            else:
                parameters = tf.get_variable(
                    name='parameters',
                    initializer=self.inverse_transform_parameters(self.parameters),
                    dtype=tf.float32)

            # n_dimensions x n_stimuli
            basis_functions = self.build_basis_function(
                decode_graph, parameters, stimulus)

            #  n_stimuli x n_voxels
            hypothetical_timeseries = tf.tensordot(
                basis_functions, weights, (1, 0), name='hypothetical_timeseries')

            # # n_timepoints x n_stimuli x n_vox
            residuals = data_to_decode[:, tf.newaxis, :] - \
                hypothetical_timeseries[tf.newaxis, ...]

            if residual_dist == 'gaussian':
                residual_dist = tfd.MultivariateNormalFullCovariance(
                    loc=tf.zeros(data.shape[1]), covariance_matrix=self.omega)
            elif residual_dist == 't':
                print('USING RESIDUAL T in decoding')
                sigma_t_ = (self.dof - 2) * self.omega / self.dof
                chol = tf.cholesky(sigma_t_)
                residual_dist = tfd.MultivariateStudentTLinearOperator(df=self.dof,
                                                                       loc=tf.zeros(
                                                                           data.shape[1]),
                                                                       scale=tf.linalg.LinearOperatorLowerTriangular(chol))

            # n_timepoints x n_stimuli
            self.decode_pdf_log_ = residual_dist.log_prob(
                residuals, name='pdf')

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

            # n_timepoints x  n_stimuli x n_stim_dimensions
            summed_hist = (self.decode_map_[:, tf.newaxis, :] -
                           stimulus[tf.newaxis, ...])**2 \
                * self.decode_pdf_[..., tf.newaxis]

            self.decode_sd_ = tf.sqrt(tf.reduce_sum(
                summed_hist, 1) / tf.reduce_sum(self.decode_pdf_, 1)[..., tf.newaxis])

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
        data = data.astype(np.float32)

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

        if stimulus_range.shape[1] > 1:
            levels = [f'stim_dim{i}' for i in range(
                1, stimulus_range.shape[1]+1)]
            columns = pd.MultiIndex.from_arrays(
                list(stimulus_range.T), names=levels)
        else:
            levels = ['stimulus']
            columns = pd.Index(stimulus_range.ravel(), name=levels[0])

        pdf = pd.DataFrame(pdf, index=data.index, columns=columns)
        map_ = pd.DataFrame(map_, index=data.index, columns=levels)
        sd = pd.DataFrame(sd, index=data.index)
        lower_ci = pd.DataFrame(lower_ci, index=data.index, columns=levels)
        higher_ci = pd.DataFrame(higher_ci, index=data.index, columns=levels)

        return pdf, map_, sd, (lower_ci, higher_ci)

    def get_parameter_labels(self, parameters):
        if hasattr(self, 'parameter_labels'):
            return self.parameter_labels
        else:
            return ['par_{}'.format(d+1) for d in range(parameters.shape[1])]

    def inverse_transform_parameters(self, parameters):
        # base case: just get rid of Dataframe
        if hasattr(parameters, 'values'):
            return parameters.values.astype(np.float32)
        else:
            return parameters.astype(np.float32)

    def transform_parameters(self, parameters):
        #base case: make dataframe
        labels = self.get_parameter_labels(parameters)
        return pd.DataFrame(parameters, columns=labels)

    def apply_mask(self, mask):

        if hasattr(self, 'weights') and (self.weights is not None):
            self.weights = self.weights.loc[:, mask]
        else:
            self.weights = None

        if hasattr(self, 'data') and (self.data is not None):
            self.data = self.data.loc[:, mask]
        else:
            self.data = None

        
        self.build_graph(self.paradigm, self.data, self.parameters, self.weights)

class IsolatedPopulationsModel(object):
    """
    This subclass of EncodingModel assumes every population maps onto
    exactly one voxel.
    """
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

    def apply_mask(self, mask):
        super().apply_mask(mask)

        if hasattr(self, 'parameters') and (self.parameters is not None):
            self.parameters = self.parameters.loc[mask]
        else:
            self.parameters = None


class GLMModel(EncodingModel):

    n_parameters = 0
    n_populations = None

    def __init__(self, parameters=None, verbosity=logging.INFO,
                 intercept=True):
        """
        parameters is a NxD or  array, where N is the number
        of basis functions and P is the number of parameters
        """
        self.include_intercept=True
        return super().__init__()

    def build_graph(self,
                    paradigm,
                    data=None,
                    parameters=None,
                    weights=None):

        parameters = self._get_dummy_parameters(
            paradigm=paradigm)
        
        return super().build_graph(paradigm, data, parameters, weights)

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

        data = data.astype(np.float32)

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
            if self.include_intercept:
                basis_predictions_ = tf.concat((tf.ones((x.shape[0], 1)),
                                               x), axis=1)
            else:
                basis_predictions = x


        return basis_predictions_

    def _get_graph_properties(self, paradigm, data, weights, parameters):
        _, n_pars, n_voxels, n_timepoints, n_stim_dimensions = super(
        )._get_graph_properties(paradigm, data, weights, parameters)

        n_populations = paradigm.shape[1]
        if self.include_intercept:
            n_populations += 1

        return n_populations, n_pars, n_voxels, n_timepoints, n_stim_dimensions

    def _check_input(self, par, name=None):
        if (par is None) and (name == 'parameters'):
            return self._get_dummy_parameters()
        else:
            return super()._check_input(par, name)

    def _get_dummy_parameters(self, paradigm=None):
        paradigm = self._check_input(paradigm, 'paradigm')
        return np.zeros((paradigm.shape[1], 0), dtype=np.float32)

class SigmoidModel(EncodingModel):
    n_parameters = 4
    parameter_labels = ['baseline', 'range', 'middle', 'slope']


    def __init__(self, paradigm=None, data=None, parameters=None,
            weights=None,
                 monotonitcaly_increasing=True):

        super().__init__(paradigm=paradigm,
                         data=data,
                         weights=weights,
                         parameters=parameters)
        self.monotonitcaly_increasing = monotonitcaly_increasing

    def build_basis_function(self, graph, parameters, x):
        with graph.as_default():
            if self.monotonitcaly_increasing:
                slope = tf.math.softplus(parameters[:, 3])
            else:
                slope = parameters[:, 3]

            sigmoid_range = tf.math.softplus(parameters[:, 1])

            basis_predictions_ = parameters[:, 0] + \
                    sigmoid_range * tf.math.sigmoid(slope * (x - parameters[:, 2]))

            return basis_predictions_


    def transform_parameters(self, parameters):

        parameters = super().transform_parameters(parameters)

        if self.monotonitcaly_increasing:
            parameters['slope'] = _softplus(parameters['slope'])

        parameters['range'] = _softplus(parameters['range'])


        return parameters

    def inverse_transform_parameters(self, parameters):

        parameters = super().inverse_transform_parameters(parameters)
        parameters[:, 1] = _inverse_softplus(parameters[:, 1])

        if self.monotonitcaly_increasing:
            parameters[:, 3] = _inverse_softplus(parameters[:, 3])

        return parameters

    def init_parameters(self, data, paradigm):
        baselines = data.min(0)
        data_ = data - baselines
        range_ = data_.max()
        slopes = np.ones(data.shape[1])
        middle = np.ones(data.shape[1]) * paradigm.mean().mean()
        
        n_populations = data.shape[1]
        pars = np.zeros(
            (n_populations, self.n_parameters), dtype=np.float32)

        pars[:, 0] = baselines
        pars[:, 1] = range_
        pars[:, 2] = middle
        pars[:, 3] = slopes

        pars = pd.DataFrame(pars, columns=self.get_parameter_labels(pars))

        return pars

class VoxelwiseSigmoidModel(SigmoidModel, IsolatedPopulationsModel):
    pass

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

class BinModel(EncodingModel):
    n_populations = None
    n_parameters = 1

    def build_basis_function(self, graph, parameters, x):
        self.logger.warning('Note that first parameter of BinModel '
                            'corresponds to the Intercept (baseline) and '
                            'is NOT used.')

        with graph.as_default():
            left_bins = parameters[tf.newaxis, 1:-1, 0]
            right_bins = parameters[tf.newaxis, 2:, 0]
            basis_predictions_ = tf.cast((left_bins <= x) & (x < right_bins), tf.float32)
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

    def __init__(self, paradigm=None, data=None, parameters=None,
                 positive_amplitudes=True):

        super().__init__(paradigm=paradigm,
                         data=data,
                         parameters=parameters)
        self.positive_amplitudes = positive_amplitudes


        if parameters is not None:
            self.n_populations = parameters.shape[0]

    def build_graph(self,
                    paradigm,
                    data=None,
                    parameters=None,
                    weights=None):

        super().build_graph(paradigm=paradigm,
                            data=data,
                            parameters=parameters,
                            weights=weights)

    def transform_parameters(self, parameters):

        parameters = super().transform_parameters(parameters)

        parameters['sd'] = _softplus(parameters['sd'])

        if self.positive_amplitudes:
            parameters['amplitude'] = _softplus(parameters['amplitude'])

        return parameters

    def inverse_transform_parameters(self, parameters):

        parameters = super().inverse_transform_parameters(parameters)

        parameters[:, 1] = _inverse_softplus(parameters[:, 1])

        if self.positive_amplitudes:
            parameters[:, 2] = _inverse_softplus(
                    parameters[:, 2])

        return parameters

    def build_basis_function(self, graph, parameters, x):
        with graph.as_default():
            mu_ = parameters[:, 0]
            sd_ = tf.math.softplus(parameters[:, 1])

            if self.positive_amplitudes:
                amplitude__ = parameters[:, 2]
                amplitude_ = tf.math.softplus(amplitude__)
            else:
                amplitude_ = parameters[:, 2]
            baseline_ = parameters[:, 3]

            basis_predictions_ = baseline_[tf.newaxis, :] + \
                norm(x, mu_[tf.newaxis, :],
                     sd_[tf.newaxis, :]) *  \
                amplitude_[tf.newaxis, :]

            return basis_predictions_

    def init_parameters(self, data, paradigm):
        baselines = data.min(0)
        data_ = data - baselines
        mus = (data_.values * paradigm.values).sum(0) / data_.values.sum(0)
        sds = (data_.values * (paradigm.values - mus)
               ** 2).sum(0) / data_.values.sum(0)
        sds[np.isnan(sds)] = 1
        mus[np.isnan(mus)] = 0
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

        pars = pd.DataFrame(pars, columns=self.get_parameter_labels(pars))

        return pars

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

    def to_bin_model(self, bins):

        bins = np.hstack(([0], bins)).astype(np.float32)

        weights = np.zeros((len(bins)-1, self.parameters.shape[0]))
        weights[0, :] = self.parameters['baseline']

        conversion_graph = tf.Graph()

        mu = self.parameters['mu'].values[np.newaxis, :]
        sd = self.parameters['sd'].values[np.newaxis, :]
        x = (bins[1:-1, np.newaxis] + bins[2:, np.newaxis]) / 2
        amplitude = self.parameters['amplitude'].values[np.newaxis, :]

        with conversion_graph.as_default():
            w_tf = amplitude * norm(x, mu, sd)

            with tf.Session() as session:
                weights[1:, :] = session.run(w_tf)

        bm = BinModel(self.paradigm,
                      self.data,
                      bins[:, np.newaxis],
                      weights)

        bm.build_graph(paradigm=bm.paradigm, parameters=bm.parameters,
                       data=bm.data, weights=bm.weights)

        return bm

class MexicanHatReceptiveFieldModel(GaussianReceptiveFieldModel):

    n_parameters = 6
    parameter_labels = ['mu', 'sd', 'amplitude', 'baseline', 'supression_sd_frac',
            'supression_amplitude_frac']

    parameter_labels_to_fit = ['mu', 'softplus(sd)',
            'softplus(amplitude)', 'baseline', 'inverse_softplus(supression_sd_frac - 1)',
            'logit(supression_amplitude_frac)']

    def __init__(self, paradigm=None, data=None, parameters=None):

        # parameters = self._check_input(parameters, 'parameters')
        if parameters is not None:
            parameters['supression_sd (frac)'] = _inverse_softplus(parameters['supression_sd (frac)'])
            parameters['supression_amplitude (frac)'] = _logit(parameters['supression_amplitude (frac)'])


        super().__init__(paradigm=paradigm,
                         data=data,
                         parameters=parameters)
        self.positive_amplitudes = True

    def build_basis_function(self, graph, parameters, x):
        with graph.as_default():
            self.mu_ = parameters[:, 0]
            self.sd_ = tf.math.softplus(parameters[:, 1])

            if self.positive_amplitudes:
                self.amplitude__ = parameters[:, 2]
                self.amplitude_ = tf.math.softplus(self.amplitude__)
            else:
                self.amplitude_ = parameters[:, 2]
            self.baseline_ = parameters[:, 3]

            self.supression_sd_ = (tf.math.softplus(parameters[:, 4])) * self.sd_
            self.supression_amplitude_ = tf.math.sigmoid(parameters[:, 5]) * self.amplitude_

            basis_predictions_ = self.baseline_[tf.newaxis, :] + \
                norm(x, self.mu_[tf.newaxis, :],
                     self.sd_[tf.newaxis, :]) *  \
                self.amplitude_[tf.newaxis, :] - \
                (norm(x, self.mu_[tf.newaxis, :], self.supression_sd_[tf.newaxis, :]) *  self.supression_amplitude_)

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


class VoxelwiseGaussianReceptiveFieldModel(IsolatedPopulationsModel, GaussianReceptiveFieldModel):
    pass

class GaussianReceptiveFieldModel2D(EncodingModel):
    """

    Paradigm x has shape (n_timepoints, n, m), where n is the number of pixels
    in the x and m the number of pixels in the z-direction.
    """

    n_parameters = 5
    parameter_labels = ['mu_x', 'mu_y', 'sd', 'baseline', 'amplitude']

    def __init__(self,
            paradigm=None,
            data=None,
            extent=[-1, 1, -1, 1],
            parameters=None):

        self.extent = np.array(extent, dtype=np.float32)

        super().__init__(paradigm=paradigm,
                         data=data,
                         parameters=parameters)


    def build_basis_function(self, graph, parameters, x):
        with graph.as_default():
            self.mu_ = parameters[:, :2]
            self.sd_ = tf.math.softplus(parameters[:, 2])
            self.amplitude_ = tf.math.softplus(parameters[:, 3])
            self.baseline_ = tf.math.softplus(parameters[:, 4])


            size = x.shape[1:]

            extent = self.extent
            coords_x, coords_y = tf.meshgrid(tf.linspace(extent[0], extent[1], size[0]), 
                                             tf.linspace(extent[2], extent[3], size[1]))

            coords = tf.stack((tf.reshape(coords_x, (size[0]*size[1],)),
                tf.reshape(coords_y, (size[0]*size[1], ))), 1)

            gauss = tfd.MultivariateNormalDiag(loc=self.mu_,
                    scale_identity_multiplier=self.sd_)
            peak_height = gauss.prob(self.mu_)

            kernel = tf.reshape(gauss.prob(coords[:, tf.newaxis, :]) / peak_height,
                    tf.concat((size, [self.parameters.shape[0]]), axis=0))

            basis_predictions_ = tf.reduce_sum((kernel[tf.newaxis, ...] * x[..., tf.newaxis]), [1, 2])

            return basis_predictions_

    

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
            raise Exception('Use self.transform_parameters')

        if name == 'paradigm':
            self.paradigm = pd.concat([pd.DataFrame(x) for x in par],
                    axis=0,
                    keys=[(e,) for e in range(1, len(par))],
                    names=['frame'])
            return par

        return pd.DataFrame(par.astype(np.float32))


class VoxelwiseGaussianReceptiveFieldModel2D(GaussianReceptiveFieldModel2D, IsolatedPopulationsModel):
    pass

def _softplus(x):
    return x * (x >= 0) + np.log1p(np.exp(-np.abs(x)))


def _logit(x):
    """ Computes the logit function, i.e. the logistic sigmoid inverse. """
    return - np.log(1. / x - 1.)


def _logit_tensor(x):
    """ Computes the logit function, i.e. the logistic sigmoid inverse. """
    return - tf.log(1. / x - 1.)


def _inverse_softplus_tensor(x):
    return tf.log(tf.exp(x) - 1)


def _inverse_softplus(x):
    return np.min((np.log(np.exp(x) - 1), x), 0)

def norm(x, mu, sigma):
    # Z = (2. * np.pi * sigma**2.)**0.5
    kernel = tf.math.exp(-.5 * (x - mu)**2. / sigma**2)
    return kernel
