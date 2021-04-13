import pandas as pd
import numpy as np
import datetime
import tensorflow as tf
import os.path as op
import os
from tqdm import tqdm
from .utils import format_data, format_parameters, format_paradigm, logit, get_rsq
import logging
from tensorflow.math import softplus, sigmoid
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

softplus_inverse = tfp.math.softplus_inverse


class ParameterFitter(object):

    def __init__(self, model, data, paradigm, memory_limit=666666666, log_dir=False, progressbar=True):
        self.model = model
        self.data = data
        self.paradigm = paradigm

        self.memory_limit = memory_limit  # 8 GB?
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
            lag=100,
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

        ssq_data = tf.reduce_sum(
            (y - tf.reduce_mean(y, 0)[tf.newaxis, :])**2, 0)

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
                    self.paradigm.values[tf.newaxis, ...], parameters[tf.newaxis, ...], None)

                residuals = y - predictions[0]

                # ssq = tf.clip_by_value( tf.math.reduce_variance(residuals, 0), 1e-6, 1e12)
                ssq = tf.reduce_sum(residuals**2, 0)
                return ssq

        else:
            @tf.function
            def get_ssq(parameters):
                predictions_ = self.model._predict(
                    self.paradigm.values[tf.newaxis, ...], parameters[tf.newaxis, ...], None)

                predictions = tf.transpose(predictions_)[
                    :, tf.newaxis, :, tf.newaxis]
                X = tf.concat([predictions, confounds], -1)
                beta = tf.linalg.lstsq(X, tf.transpose(
                    y)[:, tf.newaxis, :, tf.newaxis])
                predictions = tf.transpose((X @ beta)[:, 0, :, 0])

                residuals = y - predictions[0]

                ssq = tf.squeeze(tf.reduce_sum(residuals**2))
                ssq = tf.clip_by_value(
                    tf.math.reduce_variance(residuals, 0), 1e-6, 1e12)
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
                    r2 = (1 - (ssq / ssq_data)).numpy()
                    r2[~np.isfinite(r2)] = 0.0
                    mean_r2 = r2.mean()

                    if step >= min_n_iterations:
                        r2_diff = mean_r2 - mean_r2s[np.max((step - lag, 0))]
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
                parameters = self.model._transform_parameters_forward(
                    trans_parameters)
                return tfp.math.value_and_gradient(get_ssq, parameters)

            if optimizer == 'bfgs':
                optim_results = tfp.optimizer.bfgs_minimize(bfgs_cost,
                                                            initial_position=init_pars, tolerance=1e-6,
                                                            max_iterations=500)
            elif optimizer == 'lbfgs':
                optim_results = tfp.optimizer.lbfgs_minimize(bfgs_cost,
                                                             initial_position=init_pars, tolerance=1e-6,
                                                             max_iterations=500)

            self.estimated_parameters = format_parameters(
                optim_results.position, self.model.parameter_labels)

            ssq = get_ssq(optim_results.position)
            r2 = (1 - (ssq / ssq_data))

        self.estimated_parameters.index = self.data.columns

        self.predictions = self.model.predict(
            self.paradigm, self.estimated_parameters, self.model.weights)
        self.r2 = pd.Series(r2.numpy(), index=self.data.columns)

        return self.estimated_parameters

    def fit_grid(self, *args, **kwargs):

        # Calculate a proper chunk size for cutting up the parameter grid
        n_timepoints, n_voxels = self.data.shape
        chunk_size = self.memory_limit / n_voxels / n_timepoints
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
            grid_predictions = model._predict(paradigm[tf.newaxis, ...],
                                              par_grid[tf.newaxis, ...], None)

            # time x voxels x parameters
            ssq = tf.math.reduce_sum(
                (grid_predictions[0, :, tf.newaxis, :] - data[:, :, tf.newaxis])**2, 0)
            return ssq

        # n_voxels x n_parameters
        ssq = pd.DataFrame(np.zeros((self.data.shape[1], len(par_grid)), dtype=np.float32),
                           index=self.data.columns,
                           columns=pd.MultiIndex.from_frame(par_grid.reset_index('chunk')))

        for chunk, pg in tqdm(par_grid.groupby('chunk')):
            ssq[chunk] = _get_ssq_for_predictions(pg.values, self.data.values,
                                                  self.model, self.paradigm.values).numpy()

        ssq = ssq.droplevel('chunk', axis=1,).fillna(np.inf)

        # best_pars = ssq.columns.to_frame(
            # index=False).iloc[ssq.values.argmin(1)]

        best_pars = ssq.columns.to_frame(
            index=False).iloc[tf.math.argmin(ssq, 1)]
        best_pars.index = self.data.columns

        return best_pars

    def get_predictions(self, parameters=None):

        if parameters is None:
            parameters = self.estimated_parameters

        return self.model.predict(self.paradigm, parameters, None)

    def get_residuals(self, parameters=None):

        if parameters is None:
            parameters = self.estimated_parameters

        return self.data - self.get_predictions(parameters).values

    def get_rsq(self, parameters=None):

        if parameters is None:
            parameters = self.estimated_parameters

        return get_rsq(self.data, self.get_predictions(parameters))

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

    def __init__(self, model, data, paradigm=None, parameters=None, weights=None, lambd=0.0):

        self.model = model
        self.data = format_data(data)
        self.lambd = lambd

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

    def fit(self, init_rho=.1, init_tau=None, init_dof=1., init_sigma2=1e-3, D=None, max_n_iterations=1000,
            resid_dist='gauss',
            min_n_iterations=100,
            method='gauss',
            residuals=None,
            normalize_WWT=True,
            learning_rate=0.1, rtol=1e-6, lag=100):

        n_voxels = self.data.shape[1]

        if residuals is None:
            residuals = (self.data - self.model.predict()).values

        sample_cov = tfp.stats.covariance(residuals)

        if init_tau is None:
            init_tau = residuals.std(0)[np.newaxis, :]

        tau_ = tf.Variable(initial_value=softplus_inverse(init_tau), shape=(
            1, n_voxels), name='tau_trans', dtype=tf.float32)
        rho_ = tf.Variable(initial_value=logit(
            init_rho), shape=None, name='rho_trans', dtype=tf.float32)
        sigma2_ = tf.Variable(initial_value=softplus_inverse(
            init_sigma2), shape=None, name='sigma2_trans', dtype=tf.float32)

        # weights = self.model.weights.values
        # WtW = tf.transpose(weights) @ weights
        if hasattr(self.model, 'get_pseudoWWT'):
            print('USING A PSEUDO-WWT!')
            WWT = self.model.get_pseudoWWT()
        else:
            WWT = self.model.get_WWT()
        
        if hasattr(WWT, 'values'):
                WWT = WWT.values

        WWT = tf.clip_by_value(WWT, -1e10, 1e10)
        print(f'WWT max: {np.max(WWT)}')
        if normalize_WWT:
            WWT /= np.max(WWT)

        trainable_variables = [tau_, rho_, sigma2_]

        if method == 'gauss':
            @tf.function
            def likelihood(tau, rho, sigma2):
                if self.lambd > 0.0:
                    omega = self._get_omega_lambda(
                        tau, rho, sigma2, WWT, self.lambd, sample_cov)
                else:
                    omega = self._get_omega(tau, rho, sigma2, WWT)

                residual_dist = tfd.MultivariateNormalFullCovariance(
                    tf.zeros(n_voxels),
                    omega, allow_nan_stats=False)

                return tf.reduce_sum(residual_dist.log_prob(residuals))

            fit_stat = likelihood

        elif method == 't':

            dof_ = tf.Variable(initial_value=softplus_inverse(
                init_dof), name='tau_trans', dtype=tf.float32)

            trainable_variables += [dof_]

            @tf.function
            def likelihood(dof, tau, rho, sigma2):
                if self.lambd > 0.0:
                    omega = self._get_omega_lambda(
                        tau, rho, sigma2, WWT, self.lambd, sample_cov)
                else:
                    omega = self._get_omega(tau, rho, sigma2, WWT)

                # omega = (dof - 2) * omega / dof
                chol = tf.linalg.cholesky(omega)

                residual_dist = tfd.MultivariateStudentTLinearOperator(
                    dof,
                    tf.zeros(n_voxels),
                    tf.linalg.LinearOperatorLowerTriangular(chol), allow_nan_stats=False)

                return tf.reduce_sum(residual_dist.log_prob(residuals))

            fit_stat = likelihood

        elif method == 'ssq_cov':

            @tf.function
            def ssq(tau, rho, sigma2):
                if self.lambd > 0.0:
                    omega = self._get_omega_lambda(
                        tau, rho, sigma2, WWT, self.lambd, sample_cov)
                else:
                    omega = self._get_omega(tau, rho, sigma2, WWT)

                ssq = tf.reduce_sum((omega - sample_cov)**2)

                return -ssq

            fit_stat = ssq

        elif method == 'slogsq_cov':

            sample_cov = tfp.stats.covariance(self.data)

            @tf.function
            def ssq(tau, rho, sigma2):
                if self.lambd > 0.0:
                    omega = self._get_omega_lambda(
                        tau, rho, sigma2, WWT, self.lambd, sample_cov)
                else:
                    omega = self._get_omega(tau, rho, sigma2, WWT)

                ssq = tf.reduce_sum(tf.math.log(1+(omega - sample_cov)**2))

                return -ssq

            fit_stat = ssq

        opt = tf.optimizers.Adam(learning_rate=learning_rate)

        pbar = tqdm(range(max_n_iterations))

        self.costs = np.zeros(max_n_iterations)

        for step in pbar:
            with tf.GradientTape() as tape:
                tau = softplus(tau_)
                rho = sigmoid(rho_)
                sigma2 = softplus(sigma2_)

                if method == 't':
                    dof = softplus(dof_)
                    cost = -fit_stat(dof, tau, rho, sigma2)
                else:
                    cost = -fit_stat(tau, rho, sigma2)

                gradients = tape.gradient(cost,
                                          trainable_variables)
                opt.apply_gradients(zip(gradients, trainable_variables))

                mean_tau = tf.reduce_mean(tau).numpy()

                if method == 't':
                    pbar.set_description(
                        f'fit stat: {-cost.numpy():0.4f}, rho: {rho.numpy():0.3f}, sigma2: {sigma2.numpy():0.3f}, mean tau: {mean_tau:0.4f}, dof: {dof:0.2f}')
                else:
                    pbar.set_description(
                        f'fit stat: {-cost.numpy():0.4f}, rho: {rho.numpy():0.3f}, sigma2: {sigma2.numpy():0.3f}, mean tau: {mean_tau:0.4f}')

                self.costs[step] = cost.numpy()

                previous_cost = self.costs[np.max((step-lag, 0))]
                if (step > min_n_iterations) & (np.sign(previous_cost) == np.sign(cost)):
                    if np.sign(cost) == -1:
                        if (cost / previous_cost) < 1 - rtol:
                            break
                    else:
                        if (cost / previous_cost) > 1 - rtol:
                            break

        if self.lambd > 0.0:
            omega = self._get_omega_lambda(
                tau, rho, sigma2, WWT, self.lambd, sample_cov).numpy()
        else:
            omega = self._get_omega(tau, rho, sigma2, WWT).numpy()

        if method == 't':
            return omega, dof.numpy()
        else:
            return omega, None

    @tf.function
    def _get_omega(self, tau, rho, sigma2, WWT):
        return rho * tf.transpose(tau) @ tau + \
            (1 - rho) * tf.linalg.tensor_diag(tau[0, :]**2) + \
            sigma2 * WWT

    @tf.function
    def _get_omega_lambda(self, tau, rho, sigma2, WWT, lambd, sample_covariance, eps=1e-9):
        return (1-lambd) * (rho * tf.transpose(tau) @ tau +
                            (1 - rho) * tf.linalg.tensor_diag(tau[0, :]**2) +
                            sigma2 * WWT) + \
            lambd * sample_covariance +  \
            tf.linalg.diag(tf.ones(tau.shape[1]) * eps)


class StimulusFitter(object):

    def __init__(self, data, model, stimulus_size, omega, dof=None):

        self.data = format_data(data)
        self.model = model
        self.stimulus_size = stimulus_size
        self.model.omega = omega
        self.model.dof = dof

        if self.model.weights is None:
            self.model.weights

    def fit(self, init_stimulus=None, learning_rate=0.1, max_n_iterations=1000, min_n_iterations=100, lag=100, rtol=1e-6,
            spike_and_slab_prior=False, sigma_prior=1., alpha=.5):

        size_stimulus_var = (1, len(self.data), self.stimulus_size)

        if init_stimulus is None:
            init_stimulus = np.zeros(size_stimulus_var)

        if len(init_stimulus.shape) == 2:
            init_stimulus = init_stimulus[:, np.newaxis, :]

        decoded_stimulus = tf.Variable(initial_value=init_stimulus, shape=size_stimulus_var,
                                       name='decoded_stimulus', dtype=tf.float32)

        trainable_variables = [decoded_stimulus]

        opt = tf.optimizers.Adam(learning_rate=learning_rate)

        pbar = tqdm(range(max_n_iterations))

        self.costs = np.ones(max_n_iterations) * 1e12

        data_ = self.data.values[np.newaxis, :, :]
        parameters = self.model.parameters.values[np.newaxis, :, :]
        weights = None if self.model.weights is None else self.model.weights.values[
            np.newaxis, :, :]

        if spike_and_slab_prior:
            @tf.function
            def logprior(stimulus):
                prior_dist = tfd.Mixture(cat=tfd.Categorical(probs=[alpha, 1.-alpha]),
                                         components=[
                    tfd.Laplace(loc=0, scale=sigma_prior),
                    tfd.Normal(loc=0, scale=sigma_prior), ])

                return prior_dist.log_prob(stimulus)

        for step in pbar:
            with tf.GradientTape() as tape:
                ll = self.model._likelihood(decoded_stimulus,
                                            data_, parameters,
                                            weights, self.model.omega, self.model.dof, logp=True, normalize=False)

                cost = -tf.reduce_sum(ll)

                if spike_and_slab_prior:
                    cost += -tf.reduce_sum(logprior(decoded_stimulus))

                self.costs[step] = cost.numpy()

            gradients = tape.gradient(cost, trainable_variables)
            opt.apply_gradients(zip(gradients, trainable_variables))

            pbar.set_description(f'fit stat: {-cost.numpy():6.2f}')

            self.costs[step] = cost.numpy()

            previous_cost = self.costs[np.max((step-lag, 0))]
            if step > min_n_iterations:
                if np.sign(previous_cost) == np.sign(cost):
                    if np.sign(cost) == 1:
                        if (cost / previous_cost) > 1 - rtol:
                            break
                    else:
                        if (cost / previous_cost) < 1 - rtol:
                            break

        return decoded_stimulus[0].numpy()

    def firstpass(self):
        raise NotImplementedError("Best to start fitting with empty image")
