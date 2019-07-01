import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tqdm import tqdm
import pandas as pd


class EncodingModelFitter(object):

    def __init__(self,
                 paradigm,
                 data):

        assert(len(data) == len(paradigm))

        self.n_features = data.shape[1]
        self.n_timepoints = data.shape[0]

        self.data = pd.DataFrame(data)
        self.paradigm = np.atleast_2d(paradigm).T

        self.graph = tf.Graph()

        with self.graph.as_default():
            self.parameters = tf.Variable(np.ones((1,
                                                   self.n_parameters,
                                                   self.n_features)))

    def optimize(self,
                 min_nsteps=1000,
                 ftol=1e-9,
                 ):

        ftol_ = 1 + ftol

        with self.graph.as_default():
            init = tf.global_variables_initializer()
            with tf.Session() as session:
                session.run(init)

                costs = np.ones(min_nsteps) * np.inf

                pbar = tqdm(range(min_nsteps))
                for step in pbar:
                    _, c = session.run([self.train, self.cost])
                    costs[step] = c
                    pbar.set_description(f'Current cost: {c:7g}')

                    if (costs[step - 1] >= c) & (costs[step - 1] / c < ftol_):
                        break

                costs = pd.Series(costs[:step + 1])
                parameters, predictions = session.run(
                    [self.parameters, self.predictions])
                parameters = pd.DataFrame(np.squeeze(parameters),
                                          index=self.parameter_labels,
                                          columns=self.data.columns)

                predictions = pd.DataFrame(predictions,
                                           index=self.data.index,
                                           columns=self.data.columns)

            session.close()
            return costs, parameters, predictions


class GaussianReceptiveFieldFitter(EncodingModelFitter):

    n_parameters = 4
    parameter_labels = ['mu', 'sd', 'amplitude', 'baseline']

    def __init__(self,
                 paradigm,
                 data,
                 positive_amplitudes=True):

        super().__init__(paradigm,
                         data)

        self.positive_amplitudes = positive_amplitudes

        with self.graph.as_default():
            self.mu = self.parameters[:, 0, :]
            self.sd = tf.math.softplus(self.parameters[:, 1, :])

            if self.positive_amplitudes:
                self.amplitude_ = self.parameters[:, 2, :]
                self.amplitude = tf.math.softplus(self.amplitude_)
            else:
                self.amplitude = self.parameters[:, 2, :]
            self.baseline = self.parameters[:, 3, :]
            self.norm = tfd.Normal(self.mu, self.sd)
            self.predictions = self.baseline + \
                self.norm.prob(self.paradigm) / \
                self.norm.prob(self.mu) * self.amplitude

            self.cost = tf.reduce_sum((self.predictions - self.data.values)**2)

            self.optimizer = tf.train.AdamOptimizer()
            self.train = self.optimizer.minimize(self.cost)

    def optimize(self,
                 min_nsteps=1000):

        costs, parameters, predictions = super().optimize(min_nsteps=min_nsteps)

        parameters.loc['sd'] = _softplus(parameters.loc['sd'])
        if self.positive_amplitudes:
            parameters.loc['amplitude'] = _softplus(
                parameters.loc['amplitude'])

        return costs, parameters, predictions

class GaussianReceptiveFieldFitter(EncodingModelFitter):

    n_parameters = 4
    parameter_labels = ['mu', 'sd', 'amplitude', 'baseline']

    def __init__(self,
                 paradigm,
                 data,
                 positive_amplitudes=True):

        super().__init__(paradigm,
                         data)

        self.positive_amplitudes = positive_amplitudes

        with self.graph.as_default():
            self.mu = self.parameters[:, 0, :]
            self.sd = tf.math.softplus(self.parameters[:, 1, :])

            if self.positive_amplitudes:
                self.amplitude_ = self.parameters[:, 2, :]
                self.amplitude = tf.math.softplus(self.amplitude_)
            else:
                self.amplitude = self.parameters[:, 2, :]
            self.baseline = self.parameters[:, 3, :]
            self.norm = tfd.Normal(self.mu, self.sd)
            self.predictions = self.baseline + \
                self.norm.prob(self.paradigm) / \
                self.norm.prob(self.mu) * self.amplitude

            self.cost = tf.reduce_sum((self.predictions - self.data.values)**2)

            self.optimizer = tf.train.AdamOptimizer()
            self.train = self.optimizer.minimize(self.cost)

    def optimize(self,
                 min_nsteps=1000):

        costs, parameters, predictions = super().optimize(min_nsteps=min_nsteps)

        parameters.loc['sd'] = _softplus(parameters.loc['sd'])
        if self.positive_amplitudes:
            parameters.loc['amplitude'] = _softplus(
                parameters.loc['amplitude'])

        return costs, parameters, predictions


def _softplus(x):
    return np.log(1 + np.exp(x))
