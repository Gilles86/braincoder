import pandas as pd
import numpy as np
import datetime
import tensorflow as tf
import os.path as op
import os
from tqdm.auto import tqdm
from .utils import format_data, format_parameters, format_paradigm, logit, get_rsq
from braincoder.stimuli import ImageStimulus
import logging
from tensorflow.math import softplus, sigmoid
from tensorflow.linalg import lstsq
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from .models import LinearModelWithBaseline

softplus_inverse = tfp.math.softplus_inverse


class WeightFitter(object):

    def __init__(self, model, parameters, data, paradigm):
        self.model = model
        self.parameters = format_parameters(parameters)
        self.data = format_data(data)
        self.paradigm = self.model.get_paradigm(paradigm)

    def fit(self, alpha=0.0):
        parameters = self.model._get_parameters(self.parameters)
        parameters_ = parameters.values[np.newaxis, ...] if parameters is not None else None

        basis_predictions = self.model._basis_predictions(self.paradigm.values[np.newaxis, ...], parameters_)

        weights = lstsq(basis_predictions, self.data.values, l2_regularizer=alpha)[0]

        if (parameters is None) or type(self.model) == LinearModelWithBaseline:
            weights = pd.DataFrame(weights.numpy(),
                               columns=self.data.columns)
        else:
            weights = pd.DataFrame(weights.numpy(), index=self.parameters.index,
                               columns=self.data.columns)

        return weights

class ParameterFitter(object):

    def __init__(self, model, data, paradigm, memory_limit=666666666, log_dir=False):
        self.model = model
        self.data = data.astype(np.float32)
        self.paradigm = model.get_paradigm(paradigm)

        self.memory_limit = memory_limit  # 8 GB?

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
            progressbar=True,
            legacy_adam=False,
            **kwargs):

        n_voxels, n_pars = self.data.shape[1], len(self.model.parameter_labels)

        y = self.data.values

        if optimizer is None:

            if legacy_adam:
                opt = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate, **kwargs)
            else:
                opt = tf.optimizers.Adam(learning_rate=learning_rate, **kwargs)

        if init_pars is None:
            init_pars = self.model.get_init_pars(
                data=y, paradigm=self.paradigm, confounds=confounds)
            print('using get_init_pars')

        init_pars = self.model._get_parameters(init_pars)
        init_pars = self.model._transform_parameters_backward(init_pars.values.astype(np.float32))

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

        paradigm_ = self.model.stimulus._clean_paradigm(self.paradigm)

        if confounds is None:
            @tf.function
            def get_ssq(parameters):
                predictions = self.model._predict(
                    paradigm_[tf.newaxis, ...], parameters[tf.newaxis, ...], None)

                residuals = y - predictions[0]

                ssq = tf.reduce_sum(residuals**2, 0)
                return ssq

        else:
            @tf.function
            def get_ssq(parameters):
                predictions_ = self.model._predict(
                    paradigm_[tf.newaxis, ...], parameters[tf.newaxis, ...], None)

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
            pbar = range(max_n_iterations)
            if progressbar:
                pbar = tqdm(pbar)

            best_r2 = tf.ones(y.shape[1]) * -1e3
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
                        if progressbar:
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

                if progressbar:
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

        if not self.estimated_parameters.index.name:
            self.estimated_parameters.index.name = 'source'

        self.predictions = self.model.predict(
            self.paradigm, self.estimated_parameters, self.model.weights)
        self.r2 = pd.Series(best_r2.numpy(), index=self.data.columns)

        return self.estimated_parameters

    def fit_grid(self, *args, fixed_pars=None,
                 use_correlation_cost=False,
                 positive_amplitude=True, **kwargs):

        # Calculate a proper chunk size for cutting up the parameter grid
        n_timepoints, n_voxels = self.data.shape
        chunk_size = self.memory_limit / n_voxels / n_timepoints
        chunk_size = int(kwargs.pop('chunk_size', chunk_size))
        print(f'Working with chunk size of {chunk_size}')

        if fixed_pars is not None:
            raise NotImplementedError()

        # Make sure that ranges for all parameters are given ing
        # *args or **kwargs
        if len(args) == len(self.model.parameter_labels):
            kwargs = dict(zip(self.model.parameter_labels, args))

        if not len(kwargs.keys()) == len(self.model.parameter_labels):
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

        # paradigm_ = self.model.stimulus._clean_paradigm(self.paradigm)
        paradigm_ = self.paradigm.values

        if use_correlation_cost:

            print("Using correlation cost!")

            data_demeaned = data - tf.reduce_mean(data, 0, True)
            ssq_data = tf.reduce_sum(data_demeaned**2, 0,True)

            @tf.function
            def _cost(par_grid):
                grid_predictions = self.model._predict(paradigm_[tf.newaxis, ...],
                                                       par_grid[tf.newaxis, ...], None)

                grid_predictions_demeaned = grid_predictions[0] -  tf.reduce_mean(grid_predictions[0], 0, True)
                ssq_predictions = tf.reduce_sum(grid_predictions_demeaned**2, 0,True)

                # time x features x parameters
                r = tf.reduce_sum(grid_predictions_demeaned[:, tf.newaxis, :]*data_demeaned[:, :, tf.newaxis], 0,True) / tf.math.sqrt(ssq_predictions[:, tf.newaxis,:]*ssq_data[:, :, tf.newaxis])
                r = r[0]

                # Make sure all items in r are finite (replace inf with nan)
                
                if positive_amplitude:
                    r = tf.where(tf.math.is_finite(r), r, tf.ones_like(r) * -1)
                else:
                    r = tf.where(tf.math.is_finite(r), r, tf.zeros_like(r))
                    r = r**2 # Differentiates better than abs(r)

                best_ixs = tf.argmax(r, 1)

                return -r, best_ixs
        else:
            @tf.function
            def _cost(par_grid):
                grid_predictions = self.model._predict(paradigm_[tf.newaxis, ...],
                                                       par_grid[tf.newaxis, ...], None)

                # time x features x parameters
                ssq = tf.math.reduce_sum(
                    (grid_predictions[0, :, tf.newaxis, :] - data[:, :, tf.newaxis])**2, 0)

                best_ixs = tf.argmin(ssq, 1)

                return ssq, best_ixs

        # n features x n_chunks x n_pars
        best_pars = np.zeros((n_features, n_chunks, n_pars))
        best_cost = np.zeros((n_features, n_chunks))

        vox_ix = tf.range(n_features, dtype=tf.int64)

        for chunk, pg in tqdm(par_grid.groupby('chunk')):
            cost_, best_ix = _cost(pg.values)
            gather_ix = tf.stack((vox_ix, best_ix), 1)
            best_cost[:, chunk] = tf.gather_nd(cost_, gather_ix)
            best_pars[:, chunk] = tf.gather(pg.values, best_ix)

        best_chunks = tf.argmin(best_cost, 1)
        best_pars = tf.gather_nd(
            best_pars, tf.stack((vox_ix, best_chunks), axis=1))

        best_pars = pd.DataFrame(best_pars.numpy(), index=self.data.columns,
                                 columns=self.model.parameter_labels).astype(np.float32)

        if not best_pars.index.name:
            best_pars.index.name = 'source'

        return best_pars

    def refine_baseline_and_amplitude(self, parameters, n_iterations=1, positive_amplitude=True, l2_alpha=1.0):

        data = self.data
        predictions = self.model.predict(parameters=parameters, paradigm=self.paradigm)
        parameters = parameters.copy()

        
        if isinstance(parameters.columns, pd.MultiIndex):

            amplitude_ix = ('amplitude_unbounded', 'Intercept')
            baseline_ix = ('baseline_unbounded', 'Intercept')

            assert (amplitude_ix in parameters.columns and (
                baseline_ix in parameters.columns)), "Need parameters with amplitude and baseline"

        else:

            assert(('baseline' in parameters.columns) and (
                'amplitude' in parameters)), "Need parameters with amplitude and baseline"

            amplitude_ix = 'amplitude'
            baseline_ix = 'baseline'


        orig_r2 = get_rsq(data, predictions)

        demeaned_predictions = (predictions / parameters.loc[:, amplitude_ix]) -  parameters.loc[:, baseline_ix].T

        # n batches (voxels) x n_timepoints x regressors (2)
        X = np.stack((np.ones(
            (predictions.shape[1], predictions.shape[0])), demeaned_predictions.values.T[:, :]), 2)

        # n batches (voxels) x n_timepoints x 1
        Y = data.T.values[..., np.newaxis]
        beta = tf.linalg.lstsq(X, Y, fast=True, l2_regularizer=l2_alpha).numpy()[..., 0].astype(np.float32)
        Y_ = tf.reduce_sum(beta[:, tf.newaxis, :] * X, 2).numpy().T

        new_parameters = parameters.copy().astype(np.float32)
        new_parameters.loc[:, baseline_ix] = beta[:, 0]
        new_parameters.loc[:, amplitude_ix] = beta[:, 1]

        if positive_amplitude:
            new_parameters[amplitude_ix] = np.clip(new_parameters[amplitude_ix], 1e-4, np.inf)

        new_pred = self.model.predict(parameters=new_parameters, paradigm=self.paradigm)
        new_r2 = get_rsq(data, new_pred)

        ix = (new_r2 > orig_r2) & (data.std() != 0.0)

        parameters.loc[ix] = new_parameters.loc[ix]

        combined_pred = self.model.predict(parameters=parameters, paradigm=self.paradigm)
        combined_r2 = get_rsq(data, combined_pred)

        if n_iterations == 1:
            return parameters
        else:
            return self.refine_baseline_and_amplitude(parameters, n_iterations - 1)

    def _partly_fit_grid(self, fixed_pars, n_voxels, chunk_size, **kwargs):

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
        

        # paradigm_ = self.model.stimulus._clean_paraigm(self.paradigm)

        @tf.function
        def _get_ssq_for_predictions(par_grid):

            # paradigm: n_batches x n_timepoints x n_stimulus_features
            # parameters:: n_batches x n_voxels x n_parameters
            # norm: n_batches x n_timepoints x n_voxels

            grid_predictions = self.model._predict(paradigm_[tf.newaxis, :, :],
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

        self.paradigm = self.model.get_paradigm(paradigm)

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
            init_beta=0.0,
            progressbar=True):

        n_voxels = self.data.shape[1]

        if residuals is None:
            residuals = (self.data - self.model.predict(paradigm=self.paradigm)).values

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

        if (not hasattr(self.model, 'weights')) or (self.model.weights is None):
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

                print_str = f'fit stat: {cost.numpy():0.4f} (best: {best_cost:0.4f}, rho: {rho.numpy():0.3f}, sigma2: {sigma2.numpy():0.3f}, mean tau: {mean_tau:0.4f}'

                if len(trainable_variables) == 4:
                    dof = softplus(trainable_variables[3]).numpy()
                    print_str += f', dof: {dof:0.1f}'

                return print_str

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

                print_str = f'fit stat: {cost.numpy():0.4f} (best: {best_cost:0.4f}, rho: {rho.numpy():0.3f}, sigma2: {sigma2.numpy():0.3f}, mean tau: {mean_tau:0.4f}, alpha: {alpha.numpy():0.3f}, beta: {beta.numpy():0.3f}'

                if len(trainable_variables) == 6:
                    dof = softplus(trainable_variables[5]).numpy()
                    print_str += f', dof: {dof:0.1f}'

                return print_str

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
        
        pbar = range(max_n_iterations)

        if progressbar:
            pbar = tqdm(pbar)

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
                    cost = tf.constant(np.inf)

                if progressbar:
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
            dof = softplus(best_variables[-1]).numpy()
            self.fitted_omega_parameters['dof'] = dof
            return omega, dof
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

    def __init__(self, data, model, omega, parameters=None, weights=None, dof=None, stimulus=None, mask=None):

        self.data = format_data(data)
        self.model = model
        self.model.omega = omega
        self.model.dof = dof

        if stimulus is None:
            self.stimulus = self.model.stimulus

        self.single_bijector = not isinstance(self.stimulus.bijectors, list)

        if parameters is not None:
            self.model.parameters = parameters

        if weights is not None:
            self.model.weights = weights

    def fit(self, init_pars=None, fit_vars=None, max_n_iterations=1000, min_n_iterations=100, lag=100, rtol=1e-6, learning_rate=0.01, progressbar=True,
            l2_norm=None,
            l1_norm=None,
            mask=None,
            legacy_adam=False):

        if progressbar:
            pbar = tqdm(range(max_n_iterations))
        else:
            pbar = range(max_n_iterations)

        self.costs = np.ones(max_n_iterations) * 1e12

        if isinstance(init_pars, pd.DataFrame):
            init_pars = init_pars.values

        if init_pars is None:
            init_pars = self.stimulus.generate_empty_stimulus(len(self.data))
        elif init_pars.shape[0] != len(self.data):
            raise ValueError(
                'init_pars should have the same number of rows as data')

        if mask is not None:

            if mask.ndim == 1:
                assert(len(mask) == init_pars.shape[1]), 'Mask should have the same length as the number of stimulus dimensions'
                mask = np.tile(mask, (len(self.data), 1))
            elif mask.ndim == 2:
                mask = mask.reshape(-1)
                assert(len(mask) == init_pars.shape), 'Mask should have the same shape as number of stimulus dimensions'
                mask = np.tile(mask.reshape(-1), (len(self.data), 1))
            elif mask.ndim == 3:
                assert(len(mask) == self.data.shape[0]), 'Mask should have the same length as the number of datapoints'
                mask = mask.reshape(len(self.data), -1)
                assert(mask.shape[1] == init_pars.shape[1]), 'Mask should have the same shape as number of stimulus dimensions'
            
            mask_ix = np.array(np.where(mask)).T
            init_pars = init_pars[mask]
        
        else:
            init_pars = np.asarray(init_pars)

        model_vars = []
        trainable_vars = []

        if self.single_bijector:
            # Code to handle when self.stimulus.bijectors is not iterable
            bijector = self.stimulus.bijectors
            tf_var = tf.Variable(name='stimulus',
                                shape=init_pars.shape,
                                initial_value=bijector.inverse(init_pars))

            model_vars.append(tf_var)
            trainable_vars.append(tf_var)

        else:
            if mask is not None:
                raise NotImplementedError('Masking not implemented for multiple bijectors')

            if fit_vars is None:
                fit_vars = self.stimulus.dimension_labels

            for label, bijector, values in zip(self.stimulus.dimension_labels, self.stimulus.bijectors, init_pars.T):
                assert(np.all(~np.isnan(bijector.inverse(values)))
                    ), f'Problem with init values of {label} (nans):\n\r{bijector.inverse(values)}'
                assert(np.all(np.isfinite(bijector.inverse(values)))
                    ), f'Problem with init values of {label} (infinite values):\nr{bijector}\nr{values}\n\r{bijector.inverse(values)}'

                tf_var = tf.Variable(name=label,
                                    shape=values.shape,
                                    initial_value=bijector.inverse(values))

                model_vars.append(tf_var)

                if label in fit_vars:
                    trainable_vars.append(tf_var)

        if mask is None:
            likelihood = self.build_likelihood()
        else:
            likelihood = self.build_likelihood(use_mask=True, mask_ix=mask_ix)

        pbar = tqdm(range(max_n_iterations))
        self.costs = np.ones(max_n_iterations) * 1e12

        if legacy_adam:
            opt = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
        else:
            opt = tf.optimizers.Adam(learning_rate=learning_rate)

        for step in pbar:
            with tf.GradientTape() as tape:
                if self.single_bijector:
                    # n_pars x datapoints
                    untransformed_pars = tf.transpose(self.stimulus.bijectors.forward(model_vars[0]))
                else:
                    untransformed_pars = [bijector.forward(
                        par) for bijector, par in zip(self.stimulus.bijectors, model_vars)]

                if mask is None:
                    ll = likelihood(*untransformed_pars)[0]
                else:
                    ll = likelihood(untransformed_pars)[0]

                cost = -ll
                if l1_norm is not None:
                    cost = cost + l1_norm * tf.reduce_sum([tf.reduce_sum(tf.abs(par)) for par in untransformed_pars])
                if l2_norm is not None:
                    cost += l2_norm * tf.reduce_sum([tf.reduce_sum(par**2) for par in untransformed_pars])

            gradients = tape.gradient(cost, trainable_vars)
            opt.apply_gradients(zip(gradients, trainable_vars))

            if progressbar:
                pbar.set_description(f'LL: {ll:6.4f}')

            self.costs[step] = cost.numpy()

            previous_cost = self.costs[np.max((step - lag, 0))]
            if step > min_n_iterations:
                if np.sign(previous_cost) == np.sign(cost):
                    if np.sign(cost) == 1:
                        if (cost / previous_cost) > 1 - rtol:
                            break
                    else:
                        if (cost / previous_cost) < 1 - rtol:
                            break

        if mask is None:
            fitted_pars_ = np.stack(
                [par.numpy() for par in untransformed_pars], axis=1)
        else:
            fitted_pars_ = self.stimulus.generate_empty_stimulus(len(self.data))
            fitted_pars_[mask] = untransformed_pars.numpy()


        fitted_pars = pd.DataFrame(fitted_pars_, columns=self.stimulus.dimension_labels,
                                    index=self.data.index,
                                    dtype=np.float32)
        return fitted_pars

    def build_likelihood(self, use_batch_dimension=False, use_mask=False, mask_ix=None):

        data = self.data.values[tf.newaxis, ...]
        parameters = self.model.parameters.values[tf.newaxis, ...]
        weights = None if self.model.weights is None else self.model.weights.values[tf.newaxis, ...]
        omega_chol = np.linalg.cholesky(self.model.omega)

        if use_mask:
            if use_batch_dimension:
                return NotImplementedError('Masking not implemented for use_batch_dimension=True')
            else:

                assert(isinstance(self.stimulus, ImageStimulus)), 'Masking only implemented for ImageStimulus'

                empty_stimulus = self.stimulus.generate_empty_stimulus(len(self.data))

                @tf.function
                def likelihood(pars):

                    pars = tf.tensor_scatter_nd_add(empty_stimulus, mask_ix, pars)

                    stimulus = self.stimulus._generate_stimulus(pars)[tf.newaxis, ...]
                    ll = self.model._likelihood(
                        stimulus,  data, parameters, weights, omega_chol, dof=self.model.dof, logp=True)

                    return tf.reduce_sum(ll, 1)

        else:
            if use_batch_dimension:
                @tf.function
                def likelihood(*pars):

                    pars = tf.stack(pars, axis=0)
                    stimulus = self.stimulus._generate_stimulus(pars)[:, tf.newaxis, ...]
                    ll = self.model._likelihood(
                        stimulus,  data, parameters, weights, omega_chol, dof=self.model.dof, logp=True)

                    return ll

            else:
                @tf.function
                def likelihood(*pars):

                    pars = tf.stack(pars, axis=1)
                    stimulus = self.stimulus._generate_stimulus(pars)[tf.newaxis, ...]
                    ll = self.model._likelihood(
                        stimulus,  data, parameters, weights, omega_chol, dof=self.model.dof, logp=True)

                    return tf.reduce_sum(ll, 1)

        return likelihood


    def fit_grid(self, *pars):

        assert(len(pars) == len(self.stimulus.dimension_labels)), 'Need to provide values for all stimulus dimensions: {self.stimulus.dimension_labels}'

        pars = [np.asarray(par).astype(np.float32) for par in pars]

        parameters = dict(zip(self.stimulus.dimension_labels, pars))

        # grid length x n_pars
        grid = pd.MultiIndex.from_product(
            parameters.values(), names=parameters.keys()).to_frame(index=False).astype(np.float32)

        logging.info(f'Built grid of {len(grid)} possible stimuli...')

        likelihood = self.build_likelihood(use_batch_dimension=True)

        stimulus = self.stimulus._generate_stimulus(grid.values)

        ll = likelihood(*grid.values.T)
        ll = pd.DataFrame(likelihood(*grid.values.T).numpy().T, columns=pd.MultiIndex.from_frame(
            grid), index=self.data.index)

        best_pars = ll.columns.to_frame().iloc[ll.values.argmax(1)]
        best_pars.index = self.data.index

        return best_pars

class CustomStimulusFitter(StimulusFitter):

    def __init__(self, data, model, stimulus,
                 omega, dof=None,
                 baseline_image=None):

        self.data = data
        self.model = model
        self.stimulus = stimulus
        self.model.omega = omega
        self.model.omega_chol = tf.linalg.cholesky(omega).numpy()
        self.model.dof = dof

        if baseline_image is None:
            self.baseline_image = None
        else:
            self.baseline_image = baseline_image.reshape(
                len(self.grid_coordinates))

    # Parameters is dictionary
    def fit_grid(self, parameters):

        assert(set(parameters.keys()) == set(self.stimulus.parameter_labels)
               ), f"Provide a dictionary with the following keys: {self.stimulus.parameter_labels}"

        grid = pd.MultiIndex.from_product(
            parameters.values(), names=parameters.keys()).to_frame(index=False).astype(np.float32)

        grid = grid[self.stimulus.parameter_labels]

        logging.info(f'Built grid of {len(grid)} bar settings...')

        images = self.stimulus.generate_images(grid, return_df=False)[0]

        return self._fit_grid(grid, images)

    def _fit_grid(self, grid, images):

        data = self.data.values
        model = self.model
        parameters = self.model.parameters.values[np.newaxis, ...]
        weights = None if model.weights is None else model.weights.values[np.newaxis, ...]

        if hasattr(self.model, 'hrf_model'):

            images = tf.concat((tf.zeros((images.shape[0],
                                          1,
                                          images.shape[1])),
                                images[:, tf.newaxis, :],
                                tf.zeros((images.shape[0],
                                          len(self.model.hrf_model.hrf)-1,
                                          images.shape[1]))),
                               1)

            hrf_shift = np.argmax(model.hrf_model.hrf)

            ts_prediction = model._predict(images, parameters, weights)

            baseline = ts_prediction[:, 0, :]

            ts_prediction_summed_over_hrf = tf.reduce_sum(
                ts_prediction - baseline[:, tf.newaxis, :], 1) / tf.reduce_sum(model.hrf_model.hrf) + baseline

            ll = self.model._likelihood_timeseries(data[tf.newaxis, ...],
                                                   ts_prediction_summed_over_hrf[:,
                                                                                 tf.newaxis, :],
                                                   self.model.omega_chol,
                                                   self.model.dof,
                                                   logp=True,
                                                   normalize=False)

            ll = tf.concat((tf.roll(ll, -hrf_shift, 1)
                            [:, :-hrf_shift], tf.ones((len(images), hrf_shift)) * -1e12), 1)

        else:
            images = images[:, tf.newaxis, :]

            ll = self.model._likelihood(images, data[tf.newaxis, ...], parameters, weights,
                                        self.model.omega_chol, self.model.dof, logp=True)

        ll = pd.DataFrame(ll.numpy().T, columns=pd.MultiIndex.from_frame(grid))

        best_pars = ll.columns.to_frame().iloc[ll.values.argmax(1)]
        best_pars.index = self.data.index

        return best_pars

    def fit(self, init_pars, learning_rate=0.01, max_n_iterations=500, min_n_iterations=100, lag=100,
            fit_vars=None,
            relevant_frames=None, rtol=1e-7, parameterization='xy', include_xy=True):

        model_vars = []
        trainable_vars = []

        if fit_vars is None:
            fit_vars = self.stimulus.parameter_labels

        if hasattr(init_pars, 'values'):
            init_pars = init_pars[self.stimulus.parameter_labels].values

        if (relevant_frames is not None) and (len(init_pars) > len(relevant_frames)):
            init_pars = init_pars[relevant_frames, :]

        for label, bijector, values in zip(self.stimulus.parameter_labels, self.stimulus.bijectors, init_pars.T):
            assert(np.all(~np.isnan(bijector.inverse(values)))
                   ), f'Problem with init values of {label} (nans)'
            assert(np.all(np.isfinite(bijector.inverse(values)))
                   ), f'Problem with init values of {label} (infinte values)'

            tf_var = tf.Variable(name=label,
                                 shape=values.shape,
                                 initial_value=bijector.inverse(values))

            model_vars.append(tf_var)

            if label in fit_vars:
                trainable_vars.append(tf_var)

        likelihood = self.build_likelihood_function(relevant_frames)

        pbar = tqdm(range(max_n_iterations))
        self.costs = np.ones(max_n_iterations) * 1e12

        opt = tf.optimizers.Adam(learning_rate=learning_rate)

        for step in pbar:
            with tf.GradientTape() as tape:

                untransformed_pars = [bijector.forward(
                    par) for bijector, par in zip(self.stimulus.bijectors, model_vars)]
                ll = likelihood(*untransformed_pars)[0]
                cost = -ll

            gradients = tape.gradient(cost,
                                      trainable_vars)

            opt.apply_gradients(zip(gradients, trainable_vars))

            pbar.set_description(f'LL: {ll:6.4f}')

            self.costs[step] = cost.numpy()

            previous_cost = self.costs[np.max((step - lag, 0))]
            if step > min_n_iterations:
                if np.sign(previous_cost) == np.sign(cost):
                    if np.sign(cost) == 1:
                        if (cost / previous_cost) > 1 - rtol:
                            break
                    else:
                        if (cost / previous_cost) < 1 - rtol:
                            break

        fitted_pars_ = np.concatenate(
            [par.numpy()[:, np.newaxis] for par in untransformed_pars], axis=1)

        if relevant_frames is None:
            fitted_pars = pd.DataFrame(fitted_pars_, columns=self.stimulus.parameter_labels,
                                       index=self.data.index,
                                       dtype=np.float32)
        else:

            fitted_pars = pd.DataFrame(np.nan * np.zeros((self.data.shape[0], len(model_vars))), columns=self.stimulus.parameter_labels,
                                       index=self.data.index,
                                       dtype=np.float32)
            fitted_pars.iloc[relevant_frames, :] = fitted_pars_

        print(fitted_pars)

        return fitted_pars

    def build_likelihood_function(self, relevant_frames=None, falloff_speed=1000., n_batches=1):

        data = self.data.values[tf.newaxis, ...]
        grid_coordinates = self.stimulus.grid_coordinates.values
        model = self.model
        parameters = self.model.parameters.values[tf.newaxis, ...]
        weights = None if model.weights is None else model.weights.values[tf.newaxis, ...]

        if self.baseline_image is None:
            @tf.function
            def add_baseline(images):
                return images
        else:
            print('Including base image (e.g., fixation image) into estimation')

            @tf.function
            def add_baseline(images):
                images = images + self.baseline_image[tf.newaxis, :]
                images = tf.clip_by_value(images, 0, self.max_intensity)
                return bars

        if relevant_frames is None:

            @tf.function
            def likelihood(*pars):
                images = self.stimulus._generate_images(*pars)
                images = add_baseline(images)
                ll = self.model._likelihood(
                    images, data, parameters, weights, self.model.omega_chol, dof=self.model.dof, logp=True)

                return tf.reduce_sum(ll, 1)

        else:

            relevant_frames = tf.constant(relevant_frames, tf.int32)

            size_ = (n_batches, data.shape[1], len(grid_coordinates))
            size_ = tf.constant(size_, dtype=tf.int32)

            time_ix, batch_ix = np.meshgrid(relevant_frames, range(n_batches))
            indices = np.zeros(
                (n_batches, len(relevant_frames), 2), dtype=np.int32)
            indices[..., 0] = batch_ix
            indices[..., 1] = time_ix

            @tf.function
            def likelihood(*pars):

                images = self.stimulus._generate_images(*pars)

                images = tf.scatter_nd(indices,
                                       images,
                                       size_)

                images = add_baseline(images)

                ll = self.model._likelihood(
                    images,  data, parameters, weights, self.model.omega_chol, dof=self.model.dof, logp=True)

                return tf.reduce_sum(ll, 1)

        return likelihood


    def sample_posterior(self,
                         init_pars,
                         n_chains,
                         relevant_frames=None,
                         step_size=0.0001,
                         n_burnin=10,
                         n_samples=10,
                         max_tree_depth=10,
                         unrolled_leapfrog_steps=1,
                         falloff_speed=1000.,
                         target_accept_prob=0.85,
                         parameterization='xy'):

        init_pars = init_pars.astype(np.float32)


        if (relevant_frames is not None) and (len(init_pars) > len(relevant_frames)):
            init_pars = init_pars.iloc[relevant_frames, :]

        initial_state = list(np.repeat(init_pars.values.T[:, np.newaxis, :], n_chains, 1))

        bijectors = self.stimulus.bijectors


        likelihood = self.build_likelihood_function(relevant_frames, falloff_speed=falloff_speed,
                                                    n_batches=n_chains)



        step_size = [tf.fill([n_chains] + [1] * (len(s.shape) - 1),
                             tf.constant(step_size, np.float32)) for s in initial_state]

        samples, stats = sample_hmc(
            initial_state, step_size, likelihood,
            bijectors,
            num_steps=n_samples, burnin=n_burnin,
            target_accept_prob=target_accept_prob, unrolled_leapfrog_steps=unrolled_leapfrog_steps,
            max_tree_depth=max_tree_depth)

        if relevant_frames is None:
            frame_index = self.data.index
        else:
            frame_index = self.data.index[relevant_frames]

        cleaned_up_chains = [cleanup_chain(chain.numpy(), init_pars.columns[ix], frame_index) for ix, chain in enumerate(samples)]

        samples = pd.concat(cleaned_up_chains, 1)

        return samples, stats


class SzinteStimulus(object):

    parameter_labels = ['x', 'width', 'height']

    def __init__(self, grid_coordinates, max_width=None, max_height=None, intensity=1.0):

        self.grid_coordinates = pd.DataFrame(
            grid_coordinates, columns=['x', 'y'])


        self.image_size = len(self.grid_coordinates['x'].unique()), len(self.grid_coordinates['y'].unique())
        
        self.intensity = intensity

        self.min_x, self.max_x = self.grid_coordinates['x'].min(
        ), self.grid_coordinates['x'].max()

        if max_width is None:
            max_width = self.max_x - self.min_x

        if max_height is None:
            max_height = self.grid_coordinates['y'].max(
            ) - self.grid_coordinates['y'].min()

        self.max_width = max_width
        self.max_height = max_height

        self.bijectors = [Periodic(low=self.min_x - self.max_width/2.,
                                   high=self.max_x + self.max_width/2.),  # x
                          tfb.Sigmoid(low=np.float32(0.0),
                                      high=self.max_width),  # width
                          tfb.Sigmoid(low=np.float32(0.0), high=self.max_height)]  # height

    def generate_images(self, parameters, falloff_speed=1000, return_3dimage=True):

        if not hasattr(parameters, 'values'):
            parameters = pd.DataFrame(parameters, columns=['x', 'width', 'height'])

        ims = self._generate_images(parameters['x'].values,
                                     parameters['width'].values,
                                     parameters['height'].values,
                                     falloff_speed)

        if return_3dimage:
            return pd.DataFrame(ims.numpy().reshape(ims.shape[1], width*height),
                     index=parameters.index,
                     columns=pd.MultiIndex.from_array(self.grid_coordinates, 
                         names=['x', 'y']))

        return ims

    def _generate_images(self, x, width, height, falloff_speed=1000):
        return make_aperture_stimuli(self.grid_coordinates.values,
                                     x, width, height, falloff_speed, self.intensity)

class SzinteStimulus2(object):

    parameter_labels = ['x', 'height']

    def __init__(self, grid_coordinates, bar_width=1.4111355368411007,
                 max_height=None, intensity=1.0):

        self.grid_coordinates = pd.DataFrame(
            grid_coordinates, columns=['x', 'y'])
        self.image_size = len(self.grid_coordinates['x'].unique()), len(self.grid_coordinates['y'].unique())

        self.intensity = intensity

        self.min_x, self.max_x = self.grid_coordinates['x'].min(
        ), self.grid_coordinates['x'].max()


        if max_height is None:
            max_height = self.grid_coordinates['y'].max(
            ) - self.grid_coordinates['y'].min()

        self.bar_width = np.array([bar_width])
        self.max_height = max_height

        self.bijectors = [Periodic(low=self.min_x - self.bar_width/2.,
                                   high=self.max_x + self.bar_width/2.),  # x
                          tfb.Sigmoid(low=np.float32(0.0), high=self.max_height)]  # height

    def generate_images(self, parameters, falloff_speed=1000, return_df=True):

        if not hasattr(parameters, 'values'):
            parameters = pd.DataFrame(parameters, columns=['x', 'height'])

        ims = self._generate_images(parameters['x'].values,
                                     parameters['height'].values,
                                     falloff_speed)

        if return_df:
            return pd.DataFrame(ims.numpy().reshape(ims.shape[1], -1),
                     index=parameters.index,
                     columns=pd.MultiIndex.from_frame(self.grid_coordinates, 
                         names=['x', 'y']))
        return ims

    def _generate_images(self, x, height, falloff_speed=1000):
        return make_aperture_stimuli(self.grid_coordinates.values,
                                     x, self.bar_width, height, falloff_speed, self.intensity)

def make_aperture_stimuli(grid_coordinates, x, width, height, falloff_speed=1000., intensity=1.0):

    x_ = grid_coordinates[:, 0]
    y_ = grid_coordinates[:, 1]

    x_distance = tf.abs(x[..., tf.newaxis] - x_[tf.newaxis, tf.newaxis, ...])
    y_distance = tf.abs(y_[tf.newaxis, tf.newaxis, ...])

    bar_x = tf.math.sigmoid(
        (-x_distance + width[..., tf.newaxis] / 2) *
        falloff_speed)

    bar_y = tf.math.sigmoid(
        (-y_distance + height[..., tf.newaxis] / 2) *
        falloff_speed)

    return bar_x*bar_y*intensity

