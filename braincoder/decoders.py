import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tqdm import tqdm
import pandas as pd
from .utils import get_rsq


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

            predictions = pd.DataFrame(predictions,
                                       index=data.index,
                                       columns=data.columns)

            return costs, parameters, predictions

    def simulate(self, parameters, paradigm, noise=1.):
        """
        * Parameters should be an array of size M or (M, N),
        where M is the number of parameters and N the number of
        parameter sets.
        * Paradigm should be an array of size N, where N
        is the number of timepoints

        """

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

    def simulate(self, parameters, paradigm, noise=1.):

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

        self.build_graph(parameters)

        self.weights = weights
        self.paradigm = paradigm

    def build_graph(self, parameters):

        self.graph = tf.Graph()

        with self.graph.as_default():
            self.parameters_ = tf.constant(parameters)

            # n_timepoints x n_stim_dimensions x n_basis functions x n_voxels
            self.paradigm_ = tf.placeholder(tf.float32,
                                            shape=(None, None, 1, 1),
                                            name='paradigm')

            self.weights_ = tf.placeholder(tf.float32,
                                           shape=(1, 1, None, None),
                                           name='basis_weights')
            self.build_basis_function()

            # n_timepoints x n_voxels
            self.predictions_ = tf.squeeze(tf.tensordot(self.basis_predictions_,
                                                        self.weights_, (1, 2)))

            # Data and residuals
            self.data_ = tf.placeholder(tf.float32,
                                        shape=(None, None),
                                        name='data')
            self.residuals_ = self.data_ - self.predictions_

            # Residual model
            self.rho_trans = tf.Variable(0., dtype=tf.float32,
                                         name='rho_trans')
            self.rho_ = tf.math.sigmoid(self.rho_trans, name='rho')

            self.tau_trans_init = tf.placeholder(tf.float32, shape=(None,))

            # self.tau_trans = tf.Variable(self.tau_trans_init,
            # validate_shape=False,
            # name='tau_trans')
            self.tau_trans = tf.Variable(np.zeros(4).astype(np.float32), )

            self.tau_ = _softplus_tensor(self.tau_trans, name='tau')

            sigma0 = self.rho_ * tf.tensordot(self.tau_,
                                              tf.transpose(self.tau_),
                                              axes=1) + \
                (1 - self.rho_) * tf.linalg.tensor_diag(tf.squeeze(self.tau_))

            self.lambd_ = tf.placeholder(tf.float32, shape=1, name='lambda')
            self.empirical_covariance_matrix_ph = tf.placeholder(tf.float32,
                                                                 shape=(
                                                                     None, None),
                                                                 name='empirical_covariance_matrix')

            self.empirical_covariance_matrix_ = tf.Variable(
                self.empirical_covariance_matrix_ph, validate_shape=False)

            self.sigma_ = self.lambd_ * sigma0 + \
                (1 - self.lambd_) * self.empirical_covariance_matrix_

            # self.sigma_ = sigma0

            self.residual_dist = tfd.MultivariateNormalFullCovariance(
                tf.zeros(tf.shape(self.data_)[1]),
                self.sigma_)
            self.likelihood_ = self.residual_dist.log_prob(self.data_)

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

        with self.graph.as_default():
            with tf.Session() as session:
                predictions = session.run(self.predictions_, feed_dict={
                    self.paradigm_: paradigm.values[..., np.newaxis, np.newaxis],
                    self.weights_: weights.values[np.newaxis, np.newaxis, ...]})

        return pd.DataFrame(predictions, index=paradigm.index)

    def get_rsq(self, data, paradigm=None, weights=None):
        predictions = self.get_predictions(paradigm, weights)
        rsq = get_rsq(data, predictions)
        return rsq

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

        with self.graph.as_default(), tf.Session() as session:
            # with tf.Session() as session:

            predictions = session.run(self.noisy_predictions_,
                                      feed_dict={self.paradigm_: paradigm.values[..., np.newaxis, np.newaxis],
                                                 self.weights_: weights.values[np.newaxis, np.newaxis, ...],
                                                 self.noise_: np.atleast_1d(noise)})

        return pd.DataFrame(predictions,
                            index=paradigm.index,
                            columns=weights.columns)

    def fit(self, paradigm, data, fit_residual_model=True):
        basis_predictions = self.get_basis_function_activations(paradigm)
        data = pd.DataFrame(data)

        weights, _, _, _ = np.linalg.lstsq(basis_predictions, data, rcond=None)

        self.weights = pd.DataFrame(weights,
                                    index=basis_predictions.columns,
                                    columns=data.columns)
        self.paradigm = pd.DataFrame(paradigm)

        self.fit_residual_model(data=data)

    def fit_residual_model(self,
                           lambd=1.,
                           paradigm=None,
                           data=None):

        with self.graph.as_default():
            optimizer = tf.train.AdamOptimizer()
            cost = -tf.reduce_sum(self.likelihood_)
            train = optimizer.minimize(cost,
                    var_list=(self.tau_trans, self.rho_trans))

            feed_dict = {self.tau_trans_init: _inverse_softplus(data.std().values),
                         self.empirical_covariance_matrix_ph: data.cov().values,
                         self.paradigm_: self.paradigm.values[..., np.newaxis, np.newaxis],
                         self.weights_: self.weights.values[np.newaxis, np.newaxis],
                         self.data_: data,
                         self.lambd_: np.atleast_1d(lambd)}

            costs, rhos, sigmas = [], [], []

            init = tf.global_variables_initializer()

            with tf.Session() as session:
                session.run(init, feed_dict=feed_dict)
                c, tau, rho = session.run([cost,
                                           self.tau_, self.rho_])
                costs.append(c)

                print(f"Starting at {rho}, {tau}, cost={c})")

                feed_dict.pop(self.tau_trans_init)

                for step in range(10000):
                    # _, c, rho, tau, sigma = session.run(
                        # [train, cost, self.rho_, self.tau_, self.sigma_],
                        # feed_dict=feed_dict)
                    _, c, rho, tau, sigma = session.run(
                        [train, cost, self.rho_, self.tau_, self.sigma_],
                        feed_dict=feed_dict)

                    if step % 500 == 0:
                        print(f"{rho}, {tau}, cost={c})")
                    costs.append(c)
                    rhos.append(rho)
                    sigmas.append(sigma)

            return costs, rhos, sigmas

        # residual_graph =

    def _get_paradigm_and_weights(self, paradigm, weights):
        if paradigm is None:
            if self.paradigm is None:
                raise Exception("Please provide paradigm.")
            else:
                paradigm = self.paradigm

        if weights is None:
            if self.weights is None:
                raise Exception("Please provide basis function weights.")
            else:
                weights = self.weights

        paradigm = pd.DataFrame(paradigm)
        weights = pd.DataFrame(weights)

        return paradigm, weights


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
