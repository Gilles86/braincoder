import pandas as pd
import numpy as np
import datetime
import tensorflow as tf
import os.path as op
import os
from tqdm import tqdm
from .utils import format_data, format_parameters, format_paradigm
import logging
from tensorflow.math import softplus, sigmoid
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

softplus_inverse = tfp.math.softplus_inverse


def logit(x):
    """ Computes the logit function, i.e. the logistic sigmoid inverse. """
    return - np.log(1. / x - 1.)


class ParameterFitter(object):

    def __init__(self, model, data, paradigm, log_dir=False, progressbar=True):
        self.model = model
        self.data = data
        self.paradigm = paradigm

        self.progressbar = True

        self.log_dir = log_dir

        if log_dir is None:
            log_dir = op.abspath('logs/fit')

        if log_dir is not False:
            if not op.exists(log_dir):
                os.makedirs(log_dir)
            self.summary_writer = tf.summary.create_file_writer(op.join(log_dir,
                                                                        datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))

    def fit(self, max_n_iterations=1000,
            min_n_iterations=100,
            init_pars=None,
            confounds=None,
            optimizer=None,
            store_intermediate_parameters=True,
            r2_atol=0.000001,
            learning_rate=0.01):

        n_voxels, n_pars = self.data.shape[1], len(self.model.parameter_labels)

        y = self.data.values

        if optimizer is None:
            opt = tf.optimizers.Adam(learning_rate=learning_rate)

        if init_pars is None:
            init_pars = self.model.get_init_pars(
                data=y, paradigm=self.paradigm, confounds=confounds)
            print('using get_init_pars')
        elif hasattr(init_pars, 'values'):
            init_pars = init_pars.values.astype(np.float32)

        init_pars = self.model._transform_parameters_backward(init_pars)

        parameters = tf.Variable(initial_value=init_pars,
                                 shape=(n_voxels, n_pars), name='estimated_parameters', dtype=tf.float32)

        trainable_variables = [parameters]

        ssq_data = tf.math.reduce_variance(y, 0)
        ssq_data = tf.clip_by_value(ssq_data, 1e-6, 1e12)


        if confounds is not None:
            # n_voxels x 1 x n_timepoints x n variables
            confounds = tf.repeat(
                confounds[tf.newaxis, tf.newaxis, :, :], n_voxels, 0)

        if store_intermediate_parameters:
            intermediate_parameters = []

        mean_r2s = []


        if confounds is None:
            @tf.function
            def get_ssq(parameters):
                predictions = self.model._predict(
                    self.paradigm, parameters, None)

                residuals = y - predictions

                ssq = tf.squeeze(tf.reduce_sum(residuals**2))
                ssq = tf.clip_by_value( tf.math.reduce_variance(residuals, 0), 1e-6, 1e12)
                return ssq

        else:
            @tf.function
            def get_ssq(parameters):
                predictions_ = self.model._predict(
                    self.paradigm, parameters, None)

                predictions = tf.transpose(predictions_)[
                    :, tf.newaxis, :, tf.newaxis]
                X = tf.concat([predictions, confounds], -1)
                beta = tf.linalg.lstsq(X, tf.transpose(
                    y)[:, tf.newaxis, :, tf.newaxis])
                predictions = tf.transpose((X @ beta)[:, 0, :, 0])

                residuals = y - predictions

                ssq = tf.squeeze(tf.reduce_sum(residuals**2))
                ssq = tf.clip_by_value( tf.math.reduce_variance(residuals, 0), 1e-6, 1e12)
                return ssq

        if optimizer is None:
            pbar = tqdm(range(max_n_iterations))
            for step in pbar:
                try:
                    with tf.GradientTape() as t:

                        untransformed_parameters = self.model._transform_parameters_forward(
                            parameters)

                        ssq = get_ssq(untransformed_parameters)
                        cost = tf.reduce_sum(ssq)

                    gradients = t.gradient(cost, trainable_variables)
                    r2 = (1 - (ssq / ssq_data))
                    mean_r2 = tf.reduce_mean(r2)

                    if step >= min_n_iterations:
                        r2_diff = mean_r2 - mean_r2s[-1]
                        if (r2_diff >= 0.0) & (r2_diff < r2_atol):
                            pbar.close()
                            break

                    mean_r2s.append(mean_r2)

                    if hasattr(self, 'summary_write'):
                        with self.summary_writer.as_default():
                            tf.summary.scalar('mean R2', mean_r2, step=step)

                    if store_intermediate_parameters:
                        p = untransformed_parameters.numpy().T
                        intermediate_parameters.append(
                            np.reshape(p, np.prod(p.shape)))
                        intermediate_parameters[-1] = np.concatenate(
                            (intermediate_parameters[-1], r2), 0)

                    opt.apply_gradients(zip(gradients, trainable_variables))

                    pbar.set_description(f'Mean R2: {mean_r2:0.5f}')

                except Exception as e:
                    logging.log(f'Exception at step {step}: {e}')

            if store_intermediate_parameters:
                columns = pd.MultiIndex.from_product([self.model.parameter_labels + ['r2'],
                                                      np.arange(n_voxels)],
                                                     names=['parameter', 'voxel'])

                self.intermediate_parameters = pd.DataFrame(intermediate_parameters,
                                                            columns=columns,
                                                            index=pd.Index(np.arange(len(intermediate_parameters)),
                                                                           name='step'))

            self.estimated_parameters = format_parameters(
                untransformed_parameters.numpy(), self.model.parameter_labels)

        elif optimizer.endswith('bfgs'):

            def bfgs_cost(trans_parameters):
                parameters = self.model._transform_parameters_forward(trans_parameters)
                return tfp.math.value_and_gradient(get_ssq, parameters)

            if optimizer == 'bfgs':
                optim_results = tfp.optimizer.bfgs_minimize(bfgs_cost,
                    initial_position=init_pars, tolerance=1e-6,
                    max_iterations=500)
            elif optimizer == 'lbfgs':
                optim_results = tfp.optimizer.lbfgs_minimize(bfgs_cost,
                    initial_position=init_pars, tolerance=1e-6,
                    max_iterations=500)
        
            self.estimated_parameters = format_parameters(optim_results.position, self.model.parameter_labels)

            ssq = get_ssq(optim_results.position)
            r2 = (1 - (ssq / ssq_data))

        self.estimated_parameters.index = self.data.columns

        self.predictions = self.model.predict(self.paradigm, self.estimated_parameters, self.model.weights)
        self.r2 = pd.Series(r2.numpy(), index=self.data.columns)

        return self.estimated_parameters

    def fit_grid(self, *args, **kwargs):

        # Calculate a proper chunk size for cutting up the parameter grid
        n_timepoints, n_voxels = self.data.shape
        max_array_elements = int(4e9 / 6)  # 8 GB?
        chunk_size = max_array_elements / n_voxels / n_timepoints
        chunk_size = int(kwargs.pop('chunk_size', chunk_size))
        print(f'Working with chunk size of {chunk_size}')

        # Make sure that ranges for all parameters are given ing
        # *args or **kwargs
        if len(args) == len(self.model.parameter_labels):
            kwargs = dict(zip(self.model.parameter_labels, args))

        if not list(kwargs.keys()) == self.model.parameter_labels:
            raise ValueError(
                f'Please provide parameter ranges for all these parameters: {self.model.parameter_labels}')

        def _create_grid(model, *args):
            parameters = pd.MultiIndex.from_product(
                args, names=model.parameter_labels).to_frame(index=False)
            return parameters

        grid_args = [kwargs[key] for key in self.model.parameter_labels]
        par_grid = _create_grid(self.model, *grid_args).astype(np.float32)

        # Add chunks to the parameter columns, to process them chunk-wise and save memory
        par_grid = par_grid.set_index(
            pd.Index(par_grid.index // chunk_size, name='chunk'), append=True)

        logging.info('Built grid of {len(par_grid)} parameter settings...')

        @tf.function
        def _get_ssq_for_predictions(par_grid, data, model, paradigm):
            grid_predictions = model._predict(paradigm, par_grid, None)

            # time x voxels x parameters
            ssq = tf.math.reduce_variance(
                grid_predictions[:, tf.newaxis, :] - data[:, :, tf.newaxis], 0)
            # ssq = pd.DataFrame(ssq.numpy(), index=data.columns,
            # columns=pd.MultiIndex.from_frame(par_grid))
            return ssq

        ssq = []
        for chunk, pg in tqdm(par_grid.groupby('chunk')):
            ssq.append(_get_ssq_for_predictions(pg.values, self.data.values,
                                                self.model, self.paradigm.values))

        # ssq = pd.concat(ssq, 1)
        ssq = tf.concat(ssq, axis=1).numpy()
        ssq = pd.DataFrame(ssq, index=self.data.columns,
                           columns=pd.MultiIndex.from_frame(par_grid))

        return ssq.idxmin(1).apply(lambda row: pd.Series(row, index=self.model.parameter_labels))

    def get_predictions(self, parameters):
        return self.model.predict(self.paradigm, parameters, None)

    def get_residuals(self, parameters):
        return self.data - self.get_predictions(parameters).values

    def get_r2(self, parameters):
        residuals = self.get_residuals(parameters)
        return 1 - (residuals.var() / self.data.var())

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
    def paradigm(self):
        return self._paradigm

    @paradigm.setter
    def paradigm(self, paradigm):
        self._paradigm = format_paradigm(paradigm)


class ResidualFitter(object):

    def __init__(self, model, data, paradigm=None, parameters=None, weights=None):

        self.model = model
        self.data = format_data(data)

        if paradigm is None:
            if self.model.paradigm is None:
                raise ValueError('Need to have paradigm')
        else:
            self.model.paradigm = paradigm

        if parameters is None:
            if self.model.parameters is None:
                raise ValueError('Need to have parameters')
        else:
            self.model.parameters = parameters

        if weights is not None:
            self.model.weights = weights

    def fit(self, init_rho=.1, init_tau=None, init_sigma2=1e-9, D=None, max_n_iterations=1000,
            method='likelihood',
            learning_rate=0.1, rtol=1e-6, lag=100):

        n_voxels = self.data.shape[1]

        residuals = (self.data - self.model.predict()).values

        if init_tau is None:
            init_tau = self.data.std().values[np.newaxis, :]

        tau_ = tf.Variable(initial_value=softplus_inverse(init_tau), shape=(
            1, n_voxels), name='tau_trans', dtype=tf.float32)
        rho_ = tf.Variable(initial_value=logit(
            init_rho), shape=None, name='rho_trans', dtype=tf.float32)
        sigma2_ = tf.Variable(initial_value=softplus_inverse(
            init_sigma2), shape=None, name='sigma2_trans', dtype=tf.float32)

        weights = self.model.weights.values
        WtW = tf.transpose(weights) @ weights

        trainable_variables = [tau_, rho_, sigma2_]


        if method == 'likelihood':
            @tf.function
            def likelihood(tau, rho, sigma2):
                omega = self._get_omega(tau, rho, sigma2, WtW)

                residual_dist = tfd.MultivariateNormalFullCovariance(
                    tf.zeros(n_voxels),
                    omega, allow_nan_stats=False)

                return tf.reduce_sum(residual_dist.log_prob(residuals))

            fit_stat = likelihood

        elif method == 'ssq_cov':
            
            sample_cov = tfp.stats.covariance(self.data)

            @tf.function
            def ssq(tau, rho, sigma2):
                omega = self._get_omega(tau, rho, sigma2, WtW)
                ssq = tf.reduce_sum((omega - sample_cov)**2)

                return -ssq

            fit_stat = ssq

        elif method == 'slogsq_cov':
            
            sample_cov = tfp.stats.covariance(self.data)

            @tf.function
            def ssq(tau, rho, sigma2):
                omega = self._get_omega(tau, rho, sigma2, WtW)
                ssq = tf.reduce_sum(tf.math.log(1+(omega - sample_cov)**2))

                return -ssq

            fit_stat = ssq

        opt = tf.optimizers.Adam(learning_rate=learning_rate)
        
        pbar = tqdm(range(max_n_iterations))

        costs = np.zeros(max_n_iterations)

        for step in pbar:
            with tf.GradientTape() as tape:
                tau = softplus(tau_)
                rho = sigmoid(rho_)
                sigma2 = softplus(sigma2_)
                cost = -fit_stat(tau, rho, sigma2)
                gradients = tape.gradient(cost,
                        trainable_variables)
                opt.apply_gradients(zip(gradients, trainable_variables))

                pbar.set_description(f'fit stat: {-cost.numpy():0.4f}, rho: {rho.numpy():0.3f}, sigma2: {sigma2.numpy():0.3f}')

                costs[step] = cost.numpy()

                previous_cost = costs[np.min(step-lag, 0)]
                if np.sign(previous_cost) == np.sign(cost):
                    if (cost / previous_cost) > 1 - rtol:
                        break
    

        return self._get_omega(tau, rho, sigma2, WtW).numpy()


    @tf.function
    def _get_omega(self, tau, rho, sigma2, WtW):
        return rho * tf.transpose(tau) @ tau  + \
                (1 - rho) * tf.linalg.tensor_diag(tau[0, :]**2) + \
                sigma2 * WtW

# weights = np.identity(3)
# parameters = np.diag([1, 2, 3])
# paradigm = np.array([[1, 2, 3]]).T
# discretemodel = DiscreteModel(
# weights=weights, parameters=parameters, paradigm=paradigm)
# data = discretemodel.predict()
