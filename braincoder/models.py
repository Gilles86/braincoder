import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tqdm.autonotebook import tqdm
import pandas as pd
from .utils import get_rsq, get_r
import scipy.stats as ss
import logging


class EncodingModel(object):

    """

    An encoding model uses the following attributes:

     - parameters: (n_populations, n_pars, n_voxels)
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

    def __init__(self, verbosity=logging.INFO):
        self.logger = logging.getLogger(name='EncodingModel logger')
        self.logger.setLevel(logging.INFO)

    def fit_parameters(self, paradigm, data, min_nsteps=100000,
                       ftol=1e-9, progressbar=False,):

        assert(len(data) == len(paradigm)
               ), "paradigm and data should be same length"

        paradigm = self._check_input(paradigm)
        data = self._check_input(data)
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

                ftol_ratio = 1 + ftol
                if progressbar:
                    with tqdm(range(min_nsteps)) as pbar:
                        pbar = tqdm(range(min_nsteps))

                        for step in pbar:
                            _, c = session.run(
                                [train, self.cost)])
                            costs[step] = c
                            pbar.set_description(f'Current cost: {c:7g}')

                            if (costs[step - 1] >= c) & (costs[step - 1] / c < ftol_ratio):
                                break
                else:
                    for step in range(min_nsteps):
                        _, c, p = session.run(
                            [train, self.cost, self.parameters])
                        costs[step] = c
                        if (costs[step - 1] >= c) & (costs[step - 1] / c < ftol_ratio):
                            break

                parameters, predictions = session.run(
                    [self.parameters, self.predictions])

            costs = pd.Series(costs[:step + 1])
            parameters = pd.DataFrame(np.squeeze(parameters),
                                      index=self.parameter_labels,
                                      columns=data.columns)

            self.parameters_ = parameters

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

        if parameters.ndim == 1:
            parameters = parameters[np.newaxis, :, np.newaxis]
        elif parameters.ndim == 2:
            parameters = parameters[..., np.newaxis]

        parameters = parameters.astype(np.float32)

        self.logger.info('Size paradigm: {}'.format(paradigm.shape))
        self.logger.info('Size parameters: {}'.format(parameters.shape))
        self.logger.info('Size weights: {}'.format(weights.shape))

        self.build_graph(paradigm=paradigm, data=None,
                         parameters=parameters, weights=weights)

        noise = np.array(noise) 

        if noise.ndim < 2:
            noise = np.ones((1, weights.shape[-1])) * noise

        with self.graph.as_default():
            init = tf.global_variables_initializer()
            with tf.Session() as session:
                _ = session.run(init)
                simulated_data_ = session.run(self.noisy_prediction_,
                        feed_dict={self.noise_level_:noise})

        simulated_data = pd.DataFrame(simulated_data_,
                                      index=paradigm.index, columns=weights.columns)

        return simulated_data

    def build_graph(self, paradigm, data=None, parameters=None, weights=None):

        self.graph = tf.Graph()

        (n_populations, n_parameters, n_voxels,
         n_timepoints, n_stim_dimensions) = \
            self._get_graph_properties(
            paradigm, data, parameters, weights)

        weights = self._check_input(weights)
        paradigm = self._check_input(paradigm)
        data = self._check_input(data)
        parameters = parameters.astype(np.float32)

        self.logger.info((n_populations, n_parameters,
                          n_voxels, n_timepoints, n_stim_dimensions))

        with self.graph.as_default():

            self.paradigm_ = tf.constant(paradigm.astype(np.float32),
                                         name='paradigm')

            self.parameters_ = tf.get_variable(initializer=parameters.astype(np.float32),
                                               name='parameters',
                                               dtype=tf.float32)
            self.build_basis_function()

            weights_shape = (
                n_populations, n_voxels) if weights is None else None

            self.weights_ = tf.get_variable(initializer=weights,
                                            shape=weights_shape,
                                            name='weights',
                                            dtype=tf.float32)

            self.logger.info('Size predictions: {}'.format(
                self.basis_predictions_))
            self.predictions_ = tf.squeeze(tf.tensordot(self.basis_predictions_,
                                                        self.weights_, (1, 0)))

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

    def build_residuals_graph(self):

        self.rho_trans = tf.Variable(rho_init, dtype=tf.float32,
                                     name='rho_trans')
        self.rho_ = tf.math.sigmoid(self.rho_trans, name='rho')

        self.tau_trans = tf.Variable(_inverse_softplus(data.std().values[:, np.newaxis]),
                                     name='tau_trans')

        self.tau_ = _softplus_tensor(self.tau_trans, name='tau')

        self.sigma2_trans = tf.Variable(
            0., dtype=tf.float32, name='sigma2_trans')
        self.sigma2_ = _softplus_tensor(
            self.sigma2_trans, name='sigma2')

        sigma0 = self.rho_ * tf.tensordot(self.tau_,
                                          tf.transpose(self.tau_),
                                          axes=1) + \
            (1 - self.rho_) * tf.linalg.tensor_diag(tf.squeeze(self.tau_**2)) + \
            self.sigma2_ * tf.squeeze(tf.tensordot(self.weights_,
                                                   self.weights_, axes=(-2, -2)))

        self.empirical_covariance_matrix_ = tf.constant(
            data.cov().values.astype(np.float32), name='empirical_covariance_matrix')

        self.sigma_ = lambd * sigma0 +  \
            (1 - lambd) * self.empirical_covariance_matrix_

        self.residual_dist = tfd.MultivariateNormalFullCovariance(
            tf.zeros(data.shape[1]),
            self.sigma_)
        self.likelihood_ = self.residual_dist.log_prob(self.residuals_)

    def _check_input(self, par, name=None):
        if par is None:
            if name is not None:
                if hasattr(self, name):
                    return getattr(self, name)
            else:
                return None

        return pd.DataFrame(par.astype(np.float32))

    def build_basis_function(self):
        # time x basis_functions
        with self.graph.as_default():
            self.basis_predictions_ = self.paradigm_

    def _get_graph_properties(self, paradigm, data, weights, parameters):

        n_populations = self.n_populations
        n_pars = self.n_parameters

        if data is not None:
            n_voxels = data.shape[-1]
        else:
            n_voxels = weights.shape[-1]

        n_timepoints = paradigm.shape[0]
        n_stim_dimensions = paradigm.shape[1]

        return n_populations, n_pars, n_voxels, n_timepoints, n_stim_dimensions

    def get_basis_function_activations(self, paradigm, parameters):

        paradigm = pd.DataFrame(paradigm)

        with self.graph.as_default():
            with tf.Session() as session:
                basis_predictions = session.run(self.basis_predictions_, feed_dict={
                    self.paradigm_: paradigm.values,
                    self.parameters_: parameters})

        return pd.DataFrame(basis_predictions, index=paradigm.index)

    def get_predictions(self, paradigm=None, parameters=None, weights=None):

        paradigm = self._check_input(paradigm, name='paradigm')
        weights = self._check_input(weights, name='weights')

        self.build_graph(paradigm=paradigm,
                         parameters=parameters, weights=weights)

        with self.graph.as_default():
            with tf.Session() as session:
                predictions = session.run(self.predictions_, feed_dict={
                    self.paradigm_: paradigm.values,
                    self.weights_: weights.values,
                    self.parameters_: parameters})

        return pd.DataFrame(predictions, index=paradigm.index, columns=weights.columns)

    def fit_weights(self, paradigm, data, parameters, l2_cost=0.0):

        paradigm = self._check_input(paradigm, name='paradigm')
        weights = self._check_input(data, name='weights')

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
        self.paradigm = pd.DataFrame(paradigm)

        return self.weights


class IdentityModel(EncodingModel):

    n_parameters = 0
    n_populations = None

    def __init__(self):
        """
        parameters is a NxD or  array, where N is the number
        of basis functions and P is the number of parameters
        """
        return super().__init__()

    def build_graph(self,
                    paradigm,
                    data=None,
                    weights=None,
                    parameters=None,
                    rho_init=.5,
                    lambd=1.):

        (n_populations, n_pars, n_voxels,
         n_timepoints, n_stim_dimensions) = \
            self._get_graph_properties(
            paradigm, data, parameters, weights)

        parameters = np.zeros((n_populations, n_pars, n_voxels))

        super().build_graph(paradigm, data, parameters, weights)

    def get_rsq(self, data, paradigm=None, weights=None):
        predictions = self.get_predictions(paradigm, weights)
        predictions.index = data.index
        rsq = get_rsq(data, predictions)
        return rsq

    def get_r(self, data, paradigm=None, weights=None):
        predictions = self.get_predictions(paradigm, weights)
        r = get_r(data, predictions)
        return r

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

        n_populations, n_parameters, n_voxels, n_timepoints, n_stim_dimensions = self._get_graph_properties(
            paradigm, None, weights, parameters)

        parameters = np.zeros((n_populations, n_parameters, weights.shape[1]))

        return super().simulate(paradigm=paradigm,
                                parameters=parameters, weights=weights, noise=noise)

    def fit_weights(self, paradigm, data, l2_cost=0.0):
        parameters = np.zeros((paradigm.shape[1],
                               0,
                               data.shape[1]))

        return super().fit_weights(paradigm=paradigm,
                                   data=data,
                                   parameters=parameters,
                                   l2_cost=l2_cost)

    def fit_residual_model(self,
                           paradigm=None,
                           data=None,
                           lambd=1.,
                           min_nsteps=100000,
                           ftol=1e-12,
                           also_fit_weights=False,
                           progressbar=True):

        with self.graph.as_default():
            optimizer = tf.train.AdamOptimizer()
            cost = -tf.reduce_sum(self.likelihood_)
            var_list = [self.tau_trans, self.rho_trans, self.sigma2_trans]
            if also_fit_weights:
                var_list.append(self.weights_)

            train = optimizer.minimize(cost, var_list=var_list)

            costs = []

            init = tf.global_variables_initializer()

            with tf.Session() as session:
                session.run(init)

                self.weights_.load(self.weights.values[np.newaxis, np.newaxis, :, :],
                                   session)
                costs = np.zeros(min_nsteps)
                ftol_ratio = 1 + ftol

                if progressbar:
                    with tqdm(range(min_nsteps)) as pbar:
                        for step in pbar:
                            _, c, rho_, sigma2, weights = session.run(
                                [train, cost, self.rho_, self.sigma2_, self.weights_],)
                            costs[step] = c
                            pbar.set_description(f'Current cost: {c:7g}')

                            if (costs[step - 1] >= c) & (costs[step - 1] / c < ftol_ratio):
                                break
                else:
                    for step in range(min_nsteps):
                        _, c, rho_, sigma2, weights = session.run(
                            [train, cost, self.rho_, self.sigma2_, self.weights_],)
                        costs[step] = c
                        if (costs[step - 1] >= c) & (costs[step - 1] / c < ftol_ratio):
                            break

                costs = costs[:step+1]
                self.rho = session.run(self.rho_)
                self.tau = session.run(self.tau_)
                self.omega = session.run(self.sigma_)
                self.sigma2 = session.run(self.sigma2_)
                predictions = session.run(self.predictions_)

                predictions = pd.DataFrame(
                    predictions, index=data.index, columns=data.columns)

                self.ols_weights = self.weights.copy()

                if also_fit_weights:
                    self.weights = pd.DataFrame(np.squeeze(session.run(self.weights_)),
                                                index=self.weights.index,
                                                columns=self.weights.columns)

        return costs, predictions, self.weights

    def get_predictions(self, paradigm=None, parameters=None, weights=None):

        paradigm = self._check_input(paradigm, name='paradigm')
        weights = self._check_input(weights, name='weights')

        parameters = self._get_dummy_parameters(
            paradigm=paradigm, data=None, weights=weights)
        return super().get_predictions(paradigm, parameters, weights)

    def get_stimulus_posterior(self, data, stimulus_range=None, log_p=True, normalize=False):

        data = pd.DataFrame(data)

        if stimulus_range is None:
            stimulus = np.linspace(-5, 5, 1000)
        elif type(stimulus_range) is tuple:
            stimulus = np.linspace(stimulus_range[0], stimulus_range[1], 1000)
        else:
            stimulus = stimulus_range

        # n_stimuli x n_pop x n_vox
        hypothetical_timeseries = self.weights.values[:, np.newaxis, :] * \
            stimulus[np.newaxis, :, np.newaxis]

        # n_timepoints x n_stimuli x n_populations x n_voxels
        residuals = data.values[:, np.newaxis, np.newaxis,
                                :] - hypothetical_timeseries[np.newaxis, ...]

        mv_norm = ss.multivariate_normal(mean=np.zeros(self.omega.shape[0]),
                                         cov=self.omega)

        if log_p:
            logp_ds = mv_norm.logpdf(residuals)
            p_ds = np.exp(logp_ds - logp_ds.max(-1)[..., np.newaxis])

        else:
            # n_timepoints x n_stimuli x n_stimulus_populations
            p_ds = mv_norm.pdf(residuals)

        # Normalize
        if normalize:
            p_ds /= (p_ds * np.gradient(s)
                     [np.newaxis, np.newaxis, :]).sum(-1)[..., np.newaxis]

        return stimulus, p_ds

    def get_map_stimulus_timeseries(self, data, stimulus_range=None):

        data = pd.DataFrame(data)
        s, p_ds = self.get_stimulus_posterior(
            data, stimulus_range=stimulus_range)
        map_ = (s[np.newaxis, np.newaxis, :] * p_ds).sum(-1) / p_ds.sum(-1)
        map_ = pd.DataFrame(map_, index=data.index, columns=self.weights.index)

        return map_

    def get_map_sd_stimulus_timeseries(self, data, stimulus_range=None):

        data = pd.DataFrame(data)

        s, p_ds = self.get_stimulus_posterior(
            data, stimulus_range=stimulus_range)
        map_ = (s[np.newaxis, np.newaxis, :] * p_ds).sum(-1) / p_ds.sum(-1)
        map_ = pd.DataFrame(map_, index=data.index, columns=self.weights.index)

        dev = (s[np.newaxis, np.newaxis, :] - map_.values[..., np.newaxis])**2
        sd = np.sqrt(((dev * p_ds) / p_ds.sum(-1)[..., np.newaxis]).sum(-1))
        sd = pd.DataFrame(sd, index=data.index, columns=self.weights.index)

        return map_, sd

    def _get_paradigm_and_weights(self, paradigm, weights):
        if paradigm is None:
            if self.paradigm is None:
                raise Exception("please provide paradigm.")
            else:
                paradigm = self.paradigm

        if weights is None:
            if self.weights is None:
                raise Exception("please provide basis function weights.")
            else:
                weights = self.weights

        paradigm = pd.DataFrame(paradigm)
        weights = pd.DataFrame(weights)

        return paradigm, weights

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

    def _get_dummy_parameters(self, paradigm, data, weights):
        if weights is None:
            return np.zeros((paradigm.shape[1], 0, data.shape[1]))
        else:
            return np.zeros((paradigm.shape[1], 0, weights.shape[1]))


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

    def optimize(self,
                 paradigm,
                 data,
                 min_nsteps=100000,
                 ftol=1e-6):

        costs, parameters, predictions = super().optimize(paradigm,
                                                          data,
                                                          min_nsteps,
                                                          ftol)

        parameters.loc['sd'] = _softplus(parameters.loc['sd'])

        if self.positive_amplitudes:
            parameters.loc['amplitude'] = _softplus(
                parameters.loc['amplitude'])

        return costs, parameters, predictions

    def build_graph(self, paradigm, parameters=None):

        super().build_graph(paradigm, parameters)

        with self.graph.as_default():
            self.mu = self.parameters[:, 0, :]
            self.sd = tf.math.softplus(self.parameters[:, 1, :])

            if self.positive_amplitudes:
                self.amplitude_ = self.parameters[:, 2, :]
                self.amplitude = tf.math.softplus(self.amplitude_)
            else:
                self.amplitude = self.parameters[:, 2, :]
            self.baseline = self.parameters[:, 3, :]

            self.predictions = self.baseline + \
                norm(self.paradigm, self.mu, self.sd) * self.amplitude

    def init_parameters(self, data, paradigm):
        baselines = data.min(0)
        data_ = data - baselines
        mus = (data_ * paradigm[:, np.newaxis]).sum(0) / data_.sum(0)
        sds = (data_ * (paradigm[:, np.newaxis] -
                        mus[np.newaxis, :])**2).sum(0) / data_.sum(0)
        sds = np.sqrt(sds)
        amplitudes = data_.max(0)

        pars = np.zeros(
            (1, self.n_parameters, data.shape[1]), dtype=np.float32)

        pars[:, 0, :] = mus
        pars[:, 1, :] = sds
        pars[:, 2, :] = amplitudes
        pars[:, 3, :] = baselines

        return pars

    def simulate(self, parameters=None, paradigm=None, noise=1.):

        paradigm, parameters = self._get_paradigm_and_parameters(
            paradigm, parameters)

        parameters[:, 1] = _inverse_softplus(parameters[:, 1])
        parameters[:, 2] = _inverse_softplus(parameters[:, 2])

        data = super().simulate(parameters, paradigm, noise)
        return data


def _softplus(x):
    return np.log(1 + np.exp(x))


def _softplus_tensor(x, name=None):
    return tf.log(1 + tf.exp(x), name=name)


def _inverse_softplus(x):
    return np.log(np.exp(x) - 1)


def norm(x, mu, sigma):
    # Z = (2. * np.pi * sigma**2.)**0.5
    kernel = tf.math.exp(-.5 * (x - mu)**2. / sigma**2)
    return kernel
