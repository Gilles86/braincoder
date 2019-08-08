import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tqdm.autonotebook import tqdm
import pandas as pd
from .utils import get_rsq, get_r
import scipy.stats as ss


class EncodingModel(object):

    def __init__(self):
        pass

    def optimize(self,
                 paradigm,
                 data,
                 min_nsteps=100000,
                 ftol=1e-9,
                 ):

        assert(len(data) == len(paradigm))

        self.paradigm_ = paradigm

        paradigm = paradigm.astype(np.float32)[:, np.newaxis]
        data = pd.DataFrame(data.astype(np.float32))

        init_pars = self.init_parameters(data.values, paradigm[:, 0])

        self.init_graph(paradigm, init_pars)

        with self.graph.as_default():

            self.cost = tf.reduce_sum((data.values - self.predictions)**2)

            optimizer = tf.train.AdamOptimizer()

            train = optimizer.minimize(self.cost)

            init = tf.global_variables_initializer()

            with tf.Session() as session:
                costs = np.ones(min_nsteps) * np.inf

                _ = session.run([init])

                pbar = tqdm(range(min_nsteps))

                ftol_ratio = 1 + ftol
                for step in pbar:
                    _, c, p = session.run([train, self.cost, self.parameters])
                    costs[step] = c
                    pbar.set_description(f'Current cost: {c:7g}')

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

    def simulate(self, parameters=None, paradigm=None, noise=1.):
        """
        * Parameters should be an array of size M or (M, N),
        where M is the number of parameters and N the number of
        parameter sets.
        * Paradigm should be an array of size N, where N
        is the number of timepoints

        """

        # paradigm, parameters = self._get_paradigm_and_parameters(paradigm, parameters)
        paradigm = pd.DataFrame(paradigm.astype(np.float32))

        if parameters.ndim == 1:
            parameters = parameters[np.newaxis, :, np.newaxis]
        elif parameters.ndim == 2:
            parameters = parameters.T[np.newaxis, ...]

        parameters = parameters.astype(np.float32)

        self.init_graph(paradigm, parameters)

        with self.graph.as_default():

            noise = tf.random_normal(shape=(paradigm.shape[0],
                                            parameters.shape[2]),
                                     mean=0.0,
                                     stddev=noise,
                                     dtype=tf.float32)

            noisy_prediction = self.predictions + noise

            with tf.Session() as session:
                self.parameters.load(parameters, session)
                predictions_ = session.run(noisy_prediction)

        return pd.DataFrame(predictions_)

    def init_graph(self, paradigm, parameters=None):

        self.graph = tf.Graph()

        with self.graph.as_default():
            self.parameters = tf.Variable(parameters.astype(np.float32),
                                          name='parameters')
            self.paradigm = tf.constant(paradigm.astype(np.float32),
                                        name='paradigm')

    def _get_paradigm_and_parameters(self, paradigm, parameters):

        if paradigm is None:
            if self.paradigm_ is None:
                raise Exception("please provide paradigm.")
            else:
                paradigm = self.paradigm_

        if parameters is None:
            if self.parameters_ is None:
                raise Exception("please provide parameters.")
            else:
                parameters = self.parameters_.values

        paradigm = paradigm
        parameters = parameters

        return paradigm.copy(), parameters.copy()


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

    def init_graph(self, paradigm, parameters=None):

        super().init_graph(paradigm, parameters)

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
                norm(self.paradigm, self.mu, self.sd) * \
                self.amplitude

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


class WeightedEncodingModel(object):

    def __init__(self,
                 paradigm=None,
                 weights=None,
                 parameters=None):
        """
        parameters is a NxD or  array, where N is the number
        of basis functions and P is the number of parameters
        """

        if parameters is None:
            parameters = np.ones((0, 0, 0))

        if parameters.ndim == 2:
            parameters = parameters[:, :, np.newaxis]

        self.weights = weights
        self.paradigm = paradigm
        self.parameters = parameters

    def build_graph(self,
                    paradigm,
                    weights,
                    parameters,
                    data=None,
                    rho_init=.5,
                    lambd=1.):

        self.graph = tf.Graph()

        with self.graph.as_default():
            self.parameters_ = tf.constant(parameters)

            # n_timepoints x n_stim_dimensions x n_basis functions x n_voxels
            self.paradigm_ = tf.constant(paradigm.values[..., np.newaxis, np.newaxis],
                                         dtype=tf.float32,
                                         name='paradigm')

            self.weights_ = tf.Variable(weights.values[np.newaxis, np.newaxis, ...],
                                        dtype=tf.float32,
                                        name='basis_weights')
            self.build_basis_function()

            # n_timepoints x n_voxels
            self.predictions_ = tf.squeeze(tf.tensordot(self.basis_predictions_,
                                                        self.weights_, (1, 2)))

            # Simulation
            self.noise_ = tf.placeholder(tf.float32, shape=(1, None),
                                         name='noise')

            n_timepoints, n_voxels = tf.shape(
                self.paradigm_)[0], tf.shape(self.weights_)[-1]
            noise = tf.random_normal(shape=(n_timepoints, n_voxels),
                                     mean=0.0,
                                     stddev=self.noise_,
                                     dtype=tf.float32)

            self.noisy_predictions_ = self.predictions_ + noise

            # Data and residuals
            if data is not None:

                data = pd.DataFrame(data)

                self.data_ = tf.constant(data.values, name='data')
                self.residuals_ = self.data_ - self.predictions_

                # Residual model
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
                    (1 - self.rho_) * tf.linalg.tensor_diag(tf.squeeze(self.tau_)) + \
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

    def build_basis_function(self):
        # time x basis_functions
        with self.graph.as_default():
            self.basis_predictions_ = tf.squeeze(self.paradigm_)

    def get_basis_function_activations(self, paradigm):

        paradigm = pd.DataFrame(paradigm)

        with self.graph.as_default():
            with tf.Session() as session:
                basis_predictions = session.run(self.basis_predictions_, feed_dict={
                    self.paradigm_: paradigm.values[..., np.newaxis, np.newaxis]})

        return pd.DataFrame(basis_predictions, index=paradigm.index)

    def get_predictions(self, paradigm=None, weights=None):

        paradigm, weights = self._get_paradigm_and_weights(paradigm, weights)
        self.build_graph(paradigm, weights, self.parameters)

        with self.graph.as_default():
            with tf.Session() as session:
                predictions = session.run(self.predictions_, feed_dict={
                    self.paradigm_: paradigm.values[..., np.newaxis, np.newaxis],
                    self.weights_: weights.values[np.newaxis, np.newaxis, ...]})

        return pd.DataFrame(predictions, index=paradigm.index, columns=weights.columns)

    def get_rsq(self, data, paradigm=None, weights=None):
        predictions = self.get_predictions(paradigm, weights)
        predictions.index = data.index
        rsq = get_rsq(data, predictions)
        return rsq

    def get_r(self, data, paradigm=None, weights=None):
        predictions = self.get_predictions(paradigm, weights)
        r = get_r(data, predictions)
        return r

    def simulate(self, paradigm=None, weights=None, noise=1.):
        """
        paradigm is a N or NxM matrix, where N is the number
        of time points and M is the number of stimulus dimensions.
        weights is a BxV matrix, where B is the number
        of basis functions and V is the number of
        features (e.g., voxels, time series).
        Noise is either a scalar for equal noise across voxels
        or a V-array with the amount of noise for every voxel.

        """

        noise = np.atleast_2d(noise)

        paradigm, weights = self._get_paradigm_and_weights(paradigm, weights)

        self.build_graph(paradigm, weights, self.parameters)

        with self.graph.as_default(), tf.Session() as session:
            # with tf.Session() as session:

            predictions = session.run(self.noisy_predictions_,
                                      feed_dict={self.paradigm_: paradigm.values[..., np.newaxis, np.newaxis],
                                                 self.weights_: weights.values[np.newaxis, np.newaxis, ...],
                                                 self.noise_: np.atleast_1d(noise)})

        return pd.DataFrame(predictions,
                            index=paradigm.index,
                            columns=weights.columns)

    def fit(self, paradigm, data, rho_init=1e-9, lambd=1., fit_residual_model=True, refit_weights=False):

        paradigm = pd.DataFrame(paradigm).astype(np.float32)
        data = pd.DataFrame(data).astype(np.float32)

        init_weights = pd.DataFrame(np.zeros((paradigm.shape[1], data.shape[1]),
                                             dtype=np.float32))

        self.build_graph(paradigm, init_weights, self.parameters, data,
                         lambd=lambd, rho_init=rho_init)

        basis_predictions = self.get_basis_function_activations(paradigm)
        data = pd.DataFrame(data)

        weights, _, _, _ = np.linalg.lstsq(basis_predictions, data, rcond=None)

        self.weights = pd.DataFrame(weights,
                                    index=basis_predictions.columns,
                                    columns=data.columns)

        self.paradigm = paradigm

        if fit_residual_model:
            costs = self.fit_residual_model(data=data,
                                            also_fit_weights=refit_weights)

            return costs

    def fit_residual_model(self,
                           lambd=1.,
                           paradigm=None,
                           data=None,
                           min_nsteps=100000,
                           ftol=1e-12,
                           also_fit_weights=True):

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
                pbar = tqdm(range(min_nsteps))
                ftol_ratio = 1 + ftol
                for step in pbar:
                    _, c, rho_, sigma2, weights = session.run(
                        [train, cost, self.rho_, self.sigma2_, self.weights_],)
                    costs[step] = c
                    pbar.set_description(f'Current cost: {c:7g}')

                    if (costs[step - 1] >= c) & (costs[step - 1] / c < ftol_ratio):
                        break

                costs = costs[:step+1]
                self.rho = session.run(self.rho_)
                self.tau = session.run(self.tau_)
                self.omega = session.run(self.sigma_)
                self.sigma2 = session.run(self.sigma2_)

                self.ols_weights = self.weights.copy()

                if also_fit_weights:
                    self.weights = pd.DataFrame(np.squeeze(session.run(self.weights_)),
                                                index=self.weights.index,
                                                columns=self.weights.columns)

        return costs

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


class Discrete1DModel(WeightedEncodingModel):

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
