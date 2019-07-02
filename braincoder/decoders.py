import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tqdm import tqdm
import pandas as pd


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

                parameters, predictions = session.run([self.parameters, self.predictions])

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


def _softplus(x):
    return np.log(1 + np.exp(x))


def _inverse_softplus(x):
    return np.log(np.exp(x) - 1)


def norm(x, mu, sigma):
    # Z = (2. * np.pi * sigma**2.)**0.5
    kernel = tf.math.exp(-.5 * (x - mu)**2. / sigma**2)
    return kernel
