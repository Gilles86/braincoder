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
        self.data = data.astype(np.float32)
        self.paradigm = paradigm.astype(np.float32)

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
            fixed_pars=None,
            store_intermediate_parameters=False,
            r2_atol=0.000001,
            lag=100,
            learning_rate=0.01,
            **kwargs):

        n_voxels, n_pars = self.data.shape[1], len(self.model.parameter_labels)

        y = self.data.values

        if optimizer is None:
            opt = tf.optimizers.Adam(learning_rate=learning_rate, **kwargs)

        if init_pars is None:
            init_pars = self.model.get_init_pars(
                data=y, paradigm=self.paradigm, confounds=confounds)
            print('using get_init_pars')
        elif hasattr(init_pars, 'values'):
            init_pars = init_pars.values.astype(np.float32)

        init_pars = self.model._transform_parameters_backward(init_pars)

        ssq_data = tf.reduce_sum(
            (y - tf.reduce_mean(y, 0)[tf.newaxis, :])**2, 0)

        # Voxels with no variance to explain can confuse the optimizer to a large degree,
        # since the gradient landscape is completely flat.
        # Therefore, we only optimize voxels where there is variance to explain
        meaningful_ts = ssq_data > 0.0

        if fixed_pars is None:
            parameter_ix = range(n_pars)
        else:
            parameter_ix = [ix for ix, label in enumerate(self.model.parameter_labels) if label not in fixed_pars]

            print('*** Only fitting: ***')
            for ix in parameter_ix:
                print(f' * {self.model.parameter_labels[ix]}')

        parameter_ix = tf.constant(parameter_ix, dtype=tf.int32)

        n_meaningful_ts = tf.reduce_sum(tf.cast(meaningful_ts, tf.int32))
        n_trainable_pars = len(parameter_ix)

        update_feature_ix, update_parameter_ix = tf.meshgrid(tf.cast(tf.where(meaningful_ts), tf.int32), parameter_ix)
        update_ix = tf.stack((tf.reshape(update_feature_ix, tf.size(update_feature_ix)),
                              tf.reshape(update_parameter_ix, tf.size(update_parameter_ix))), 1)

        print(
            f'Number of problematic voxels (mask): {tf.reduce_sum(tf.cast(meaningful_ts == False, tf.int32))}')
        print(
            f'Number of voxels remaining (mask): {tf.reduce_sum(tf.cast(meaningful_ts == True, tf.int32))}')

        trainable_parameters = tf.Variable(initial_value=tf.gather_nd(init_pars, update_ix),
                                           shape=(n_meaningful_ts*n_trainable_pars),
                                           name='estimated_parameters', dtype=tf.float32)

        trainable_variables = [trainable_parameters]

        if confounds is not None:
            # n_voxels x 1 x n_timepoints x n variables
            confounds = tf.repeat(
                confounds[tf.newaxis, tf.newaxis, :, :], n_voxels, 0)

        if store_intermediate_parameters:
            intermediate_parameters = []

        mean_best_r2s = []

        if confounds is None:
            @tf.function
            def get_ssq(parameters):
                predictions = self.model._predict(
                    self.paradigm.values[tf.newaxis, ...], parameters[tf.newaxis, ...], None)

                residuals = y - predictions[0]

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
            best_r2 = tf.zeros(y.shape[1])
            best_parameters = tf.zeros(init_pars.shape)

            for step in pbar:
                with tf.GradientTape() as t:
                    parameters = tf.tensor_scatter_nd_update(
                        init_pars, update_ix, trainable_parameters)
                    untransformed_parameters = self.model._transform_parameters_forward(
                        parameters)

                    ssq = get_ssq(untransformed_parameters)
                    cost = tf.reduce_sum(ssq)

                gradients = t.gradient(cost, trainable_variables)
                r2 = (1 - (ssq / ssq_data))

                improved_r2s = r2 > best_r2
                best_parameters = tf.where(
                    improved_r2s[:, tf.newaxis], untransformed_parameters, best_parameters)
                best_r2 = tf.where(improved_r2s, r2, best_r2)

                mean_current_r2 = r2[meaningful_ts].numpy().mean()
                mean_best_r2 = best_r2[meaningful_ts].numpy().mean()

                if step >= min_n_iterations:
                    r2_diff = mean_best_r2 - \
                        mean_best_r2s[np.max((step - lag, 0))]
                    if (r2_diff >= 0.0) & (r2_diff < r2_atol):
                        pbar.close()
                        break

                mean_best_r2s.append(mean_best_r2)

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

                pbar.set_description(
                    f'Current R2: {mean_current_r2:0.5f}/Best R2: {mean_best_r2:0.5f}')

            if store_intermediate_parameters:
                columns = pd.MultiIndex.from_product([self.model.parameter_labels + ['r2'],
                                                      np.arange(n_voxels)],
                                                     names=['parameter', 'voxel'])

                self.intermediate_parameters = pd.DataFrame(intermediate_parameters,
                                                            columns=columns,
                                                            index=pd.Index(np.arange(len(intermediate_parameters)),
                                                                           name='step'))

            self.estimated_parameters = format_parameters(
                best_parameters.numpy(), self.model.parameter_labels)

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

        self.estimated_parameters.index = self.data.columns

        self.predictions = self.model.predict(
            self.paradigm, self.estimated_parameters, self.model.weights)
        self.r2 = pd.Series(best_r2.numpy(), index=self.data.columns)

        return self.estimated_parameters

    def fit_grid(self, *args, fixed_pars=None, **kwargs):

        # Calculate a proper chunk size for cutting up the parameter grid
        n_timepoints, n_voxels = self.data.shape
        chunk_size = self.memory_limit / n_voxels / n_timepoints
        chunk_size = int(kwargs.pop('chunk_size', chunk_size))
        print(f'Working with chunk size of {chunk_size}')

        if fixed_pars is not None:
            return self._partly_fit_grid(fixed_pars, n_timepoints, n_voxels,
                                         chunk_size, **kwargs)

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

        # n_permutations x n_pars
        par_grid = _create_grid(self.model, *grid_args).astype(np.float32)

        # # Add chunks to the parameter columns, to process them chunk-wise and save memory
        par_grid = par_grid.set_index(
            pd.Index(par_grid.index // chunk_size, name='chunk'), append=True)

        n_chunks = ((len(par_grid) - 1) // chunk_size) + 1
        n_pars = par_grid.shape[1]
        n_features = self.data.shape[1]

        logging.info('Built grid of {len(par_grid)} parameter settings...')

        data = self.data.values
        paradigm = self.paradigm.values

        @tf.function
        def _get_ssq_for_predictions(par_grid):
            grid_predictions = self.model._predict(paradigm[tf.newaxis, ...],
                                                   par_grid[tf.newaxis, ...], None)

            # time x features x parameters
            ssq = tf.math.reduce_sum(
                (grid_predictions[0, :, tf.newaxis, :] - data[:, :, tf.newaxis])**2, 0)

            best_ixs = tf.argmin(ssq, 1)

            return ssq, best_ixs

        # n features x n_chunks x n_pars
        best_pars = np.zeros((n_features, n_chunks, n_pars))
        best_ssq = np.zeros((n_features, n_chunks))

        vox_ix = tf.range(n_features, dtype=tf.int64)

        for chunk, pg in tqdm(par_grid.groupby('chunk')):
            ssq_, best_ix = _get_ssq_for_predictions(pg.values)

            gather_ix = tf.stack((vox_ix, best_ix), 1)
            best_ssq[:, chunk] = tf.gather_nd(ssq_, gather_ix)
            best_pars[:, chunk] = tf.gather(pg.values, best_ix)

        best_chunks = tf.argmin(best_ssq, 1)
        best_pars = tf.gather_nd(
            best_pars, tf.stack((vox_ix, best_chunks), axis=1))

        best_pars = pd.DataFrame(best_pars.numpy(), index=self.data.columns,
                                 columns=self.model.parameter_labels).astype(np.float32)

        return best_pars

    def refine_baseline_and_amplitude(self, parameters, n_iterations=2):

        data = self.model.data
        predictions = self.model.predict(parameters=parameters)
        parameters = parameters.copy()

        assert(('baseline' in parameters.columns) and (
            'amplitude' in parameters)), "Need parameters with amplitude and baseline"

        orig_r2 = get_rsq(data, predictions)
        print(f"Original mean r2: {orig_r2.mean()}")

        # n batches (voxels) x n_timepoints x regressors (2)
        X = np.stack((np.ones(
            (predictions.shape[1], predictions.shape[0])), predictions.values.T[:, :]), 2)

        # n batches (voxels) x n_timepoints x 1
        Y = data.T.values[..., np.newaxis]
        beta = tf.linalg.lstsq(X, Y, fast=False).numpy()[..., 0]
        Y_ = tf.reduce_sum(beta[:, tf.newaxis, :] * X, 2).numpy().T

        new_r2 = get_rsq(data, Y_)
        ix = (new_r2 > orig_r2) & (orig_r2 > 1e-3)

        parameters.loc[ix, 'baseline'] += beta[ix.values, 0]
        parameters.loc[ix, 'amplitude'] *= beta[ix.values, 1]

        r2 = get_rsq(self.model.data, self.model.predict(parameters=parameters))
        print(f"New mean r2 after OLS: {r2.mean()}")

        if n_iterations == 1:
            return parameters
        else:
            return self.refine_baseline_and_amplitude(parameters, n_iterations - 1)

    def _partly_fit_grid(self, fixed_pars, n_timepoints, n_voxels, chunk_size, **kwargs):

        for key in kwargs.keys():
            if key in fixed_pars:
                print(
                    f'Dropping {key} from fixed_pars since it is in the grid parameters')
                fixed_pars = fixed_pars.drop(key, 1)

        if not set(list(kwargs.keys()) + fixed_pars.columns.tolist()) == set(self.model.parameter_labels):
            raise ValueError(
                f'Please provide parameter ranges for all these parameters: {self.model.parameter_labels},'
                'either in the grid or in the fixed_pars')

        # Note that we have way more to calculate than in hte 
        # fit_grid case
        chunk_size = chunk_size // self.data.shape[1] + 1

        grid_key_ixs = [self.model.parameter_labels.index(
            key) for key in kwargs.keys()]
        init_par_ixs = [self.model.parameter_labels.index(
            key) for key in fixed_pars.columns]

        # n_permutations x n_pars
        par_grid1 = pd.MultiIndex.from_product(
            kwargs.values(), names=kwargs.keys()).to_frame(index=False)
        par_grid1 = np.repeat(par_grid1.values[np.newaxis, :, :], n_voxels, 0)

        n_par_permutations = par_grid1.shape[1]
        n_chunks = ((n_par_permutations - 1) // chunk_size) + 1
        n_features = self.data.shape[1]
        n_pars = len(self.model.parameter_labels)

        par_grid = np.zeros((n_par_permutations, n_features, n_pars), dtype=np.float32)

        for old_ix, new_ix in enumerate(grid_key_ixs):
            par_grid[:, :, new_ix] = par_grid1[:, :, old_ix].T

        for old_ix, new_ix in enumerate(init_par_ixs):
            par_grid[:, :, new_ix] = fixed_pars.values[np.newaxis, :, old_ix]

        # n features x n_chunks x n_pars
        best_pars = np.zeros((n_features, n_chunks, n_pars))
        best_ssq = np.zeros((n_features, n_chunks))

        vox_ix = tf.range(n_features, dtype=tf.int64)

        data = self.data.values
        paradigm = self.paradigm.values

        @tf.function
        def _get_ssq_for_predictions(par_grid):

            # paradigm: n_batches x n_timepoints x n_stimulus_features
            # parameters:: n_batches x n_voxels x n_parameters
            # norm: n_batches x n_timepoints x n_voxels

            grid_predictions = self.model._predict(paradigm[tf.newaxis, :, :],
                                                   par_grid, None)

            resid = data[tf.newaxis, ...] - grid_predictions

            ssq = tf.reduce_sum(resid**2, 1)

            return ssq, tf.argmin(ssq, 0)

        for chunk in tqdm(range(n_chunks)):

            pg = par_grid[chunk*chunk_size:(chunk+1)*chunk_size, :, :]
            print(pg.shape)

            ssq_, best_ix = _get_ssq_for_predictions(pg)

            gather_ix = tf.stack((best_ix, vox_ix), 1)
            best_ssq[:, chunk] = tf.gather_nd(ssq_, gather_ix)
            best_pars[:, chunk, :] = tf.gather_nd(pg, gather_ix)

        best_chunks = tf.argmin(best_ssq, 1)
        best_pars = tf.gather_nd(
            best_pars, tf.stack((vox_ix, best_chunks), axis=1))

        best_pars = pd.DataFrame(best_pars.numpy(), index=self.data.columns,
                                 columns=self.model.parameter_labels).astype(np.float32)

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
            learning_rate=0.02, rtol=1e-6, lag=100,
            init_alpha=0.99,
            init_beta=0.0):

        n_voxels = self.data.shape[1]

        if residuals is None:
            residuals = (self.data - self.model.predict()).values

        sample_cov = tfp.stats.covariance(residuals)

        if init_tau is None:
            init_tau = residuals.std(0)[np.newaxis, :]

        print(f'init_tau: {init_tau.min()}, {init_tau.max()}')

        tau_ = tf.Variable(initial_value=softplus_inverse(init_tau), shape=(
            1, n_voxels), name='tau_trans', dtype=tf.float32)
        rho_ = tf.Variable(initial_value=logit(
            init_rho), shape=None, name='rho_trans', dtype=tf.float32)
        sigma2_ = tf.Variable(initial_value=softplus_inverse(
            init_sigma2), shape=None, name='sigma2_trans', dtype=tf.float32)

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
            WWT /= np.mean(WWT)

        trainable_variables = [tau_, rho_, sigma2_]



        if D is None:

            @tf.function
            def transform_variables(trainable_variables):
                tau_, rho_, sigma2_ = trainable_variables[:3]

                tau = softplus(tau_)
                rho = sigmoid(rho_)
                sigma2 = softplus(sigma2_)

                return tau, rho, sigma2

            @tf.function
            def get_omega(trainable_variables):
                tau, rho, sigma2 = transform_variables(trainable_variables)
                omega = self._get_omega(tau, rho, sigma2, WWT)

                return omega

            def get_pbar_description(cost, best_cost, trainable_variables):
                tau, rho, sigma2 = transform_variables(trainable_variables)
                mean_tau = tf.reduce_mean(tau).numpy()

                return f'fit stat: {cost.numpy():0.4f} (best: {best_cost:0.4f}, rho: {rho.numpy():0.3f}, sigma2: {sigma2.numpy():0.3f}, mean tau: {mean_tau:0.4f} '

        else:

            alpha_ = tf.Variable(initial_value=logit(init_alpha), shape=None,
                                 name='alpha_trans', dtype=tf.float32)
            beta = tf.Variable(initial_value=init_beta, shape=None,
                               name='beta', dtype=tf.float32)

            trainable_variables += [alpha_, beta]

            @tf.function
            def transform_variables(trainable_variables):
                tau_, rho_, sigma2_, alpha_, beta = trainable_variables[:5]

                tau = softplus(tau_)
                rho = sigmoid(rho_)
                sigma2 = softplus(sigma2_)
                alpha = sigmoid(alpha_)

                return tau, rho, sigma2, alpha, beta

            @tf.function
            def get_omega(trainable_variables):
                tau, rho, sigma2, alpha, beta = transform_variables(trainable_variables)

                omega = self._get_omega_distance(
                    tau, rho, sigma2, WWT, alpha, beta, D)

                return omega

            def get_pbar_description(cost, best_cost, trainable_variables):

                tau, rho, sigma2, alpha, beta = transform_variables(trainable_variables)

                mean_tau = tf.reduce_mean(tau).numpy()

                return f'fit stat: {cost.numpy():0.4f} (best: {best_cost:0.4f}, rho: {rho.numpy():0.3f}, sigma2: {sigma2.numpy():0.3f}, mean tau: {mean_tau:0.4f}, alpha: {alpha.numpy():0.3f}, beta: {beta.numpy():0.3f}'

        if method == 'gauss':
            @tf.function
            def likelihood(omega):
                omega_chol = tf.linalg.cholesky(omega)

                residual_dist = tfd.MultivariateNormalTriL(
                    tf.zeros(n_voxels),
                    omega_chol, allow_nan_stats=False)

                return tf.reduce_sum(residual_dist.log_prob(residuals))

            fit_stat = likelihood

        elif method == 't':

            dof_ = tf.Variable(initial_value=softplus_inverse(
                init_dof), name='tau_trans', dtype=tf.float32)

            trainable_variables += [dof_]

            @tf.function
            def likelihood(omega):
                omega_chol = tf.linalg.cholesky(omega)

                dof = softplus(trainable_variables[-1])

                residual_dist = tfd.MultivariateStudentTLinearOperator(
                    dof,
                    tf.zeros(n_voxels),
                    tf.linalg.LinearOperatorLowerTriangular(omega_chol), allow_nan_stats=False)

                return tf.reduce_sum(residual_dist.log_prob(residuals))

            fit_stat = likelihood

        elif method == 'ssq_cov':
            raise NotImplementedError()

        elif method == 'slogsq_cov':
            raise NotImplementedError()

        opt = tf.optimizers.Adam(learning_rate=learning_rate)
        pbar = tqdm(range(max_n_iterations))
        self.costs = np.zeros(max_n_iterations)

        def copy_variables(traiable_variables):
            return [tf.identity(e) for e in trainable_variables]

        best_cost = np.inf
        # best_omega = np.diag(init_tau)
        best_omega = get_omega(trainable_variables)
        best_variables = copy_variables(trainable_variables)

        for step in pbar:
            with tf.GradientTape() as tape:
                try:
                    omega = get_omega(trainable_variables)
                    cost = -fit_stat(omega)

                    gradients = tape.gradient(cost,
                                              trainable_variables)

                    opt.apply_gradients(zip(gradients, trainable_variables))
                    self.costs[step] = cost.numpy()

                    if self.costs[step] < best_cost:
                        best_omega = omega.numpy()
                        best_cost = self.costs[step]
                        best_variables = copy_variables(trainable_variables)

                except Exception as e:
                    learning_rate = 0.9 * learning_rate
                    opt = tf.optimizers.Adam(learning_rate=learning_rate)
                    trainable_variables = copy_variables(best_variables)
                    self.costs[step] = np.inf

                pbar.set_description(get_pbar_description(
                    cost, best_cost, best_variables))

                previous_cost = self.costs[np.max((step-lag, 0))]
                if (step > min_n_iterations) & (np.sign(previous_cost) == np.sign(cost)):
                    if np.sign(cost) == -1:
                        if (cost / previous_cost) < 1 + rtol:
                            break
                    else:
                        if (cost / previous_cost) > 1 - rtol:
                            break
        omega = best_omega

        fitted_parameters = [e.numpy() for e in transform_variables(best_variables)]
        self.fitted_omega_parameters = dict(zip(['tau', 'rho', 'sigma2'], fitted_parameters[:3]))

        if D is not None:
            self.fitted_omega_parameters['alpha'] = fitted_parameters[3]
            self.fitted_omega_parameters['beta'] = fitted_parameters[4]

        if method == 't':
            self.fitted_omega_parameters['dof'] = fitted_parameters[-1]
            return omega, trainable_variables[-1].numpy()
        else:
            return omega, None

    @tf.function
    def _get_omega(self, tau, rho, sigma2, WWT):
        return rho * tf.transpose(tau) @ tau + \
            (1 - rho) * tf.linalg.tensor_diag(tau[0, :]**2) + \
            sigma2 * WWT

    @tf.function
    def _get_omega_distance(self, tau, rho, sigma2, WWT, alpha, beta, D):

        tautau = tf.transpose(tau) @ tau
        return rho * (alpha * (tf.exp(-beta * D) * tautau) + (1-alpha) * tautau) + \
            (1-rho) * tf.linalg.tensor_diag(tau[0, :]**2) + \
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
