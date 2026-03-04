import pandas as pd
import numpy as np
import datetime
import tensorflow as tf
import os.path as op
import os
from tqdm.auto import tqdm
from ..utils import format_data, format_parameters, format_paradigm, get_rsq
import logging


class ParameterFitter(object):
    """Iterative optimizer that estimates model parameters for each voxel.

    Wraps TensorFlow optimizers (Adam by default), handles parameter transforms,
    shared/fixed constraints, logging, and exposes ``fit`` plus helper methods
    for diagnostics such as predictions, residuals, and R².
    """

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
            shared_pars=None,  # New: Parameters to be shared across voxels
            store_intermediate_parameters=False,
            r2_atol=0.000001,
            lag=100,
            learning_rate=0.01,
            progressbar=True,
            legacy_adam=False,
            **kwargs):
        """Run iterative optimization to estimate model parameters for each voxel."""

        n_voxels, n_pars = self.data.shape[1], len(self.model.parameter_labels)

        y = self.data.values

        if optimizer is None:
            if legacy_adam:
                opt = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate, **kwargs)
            else:
                opt = tf.optimizers.Adam(learning_rate=learning_rate, **kwargs)

        if init_pars is None:
            init_pars = self.model.get_init_pars(data=y, paradigm=self.paradigm, confounds=confounds)
            print('using get_init_pars')

        init_pars = self.model._get_parameters(init_pars)
        init_pars = self.model._transform_parameters_backward(init_pars.values.astype(np.float32))

        ssq_data = tf.reduce_sum((y - tf.reduce_mean(y, 0)[tf.newaxis, :])**2, 0)
        meaningful_ts = ssq_data > 0.0

        # Handle fixed parameters
        if fixed_pars is None:
            parameter_ix = range(n_pars)
        else:
            for fixed_par in fixed_pars:
                if fixed_par not in self.model.parameter_labels:
                    raise ValueError(f'Fixed parameter "{fixed_par}" not in model parameters: {self.model.parameter_labels}')

            parameter_ix = [ix for ix, label in enumerate(self.model.parameter_labels) if label not in fixed_pars]

        print('*** Fitting: ***')
        for ix in parameter_ix:
            print(f' * {self.model.parameter_labels[ix]}')

        if len(parameter_ix) != n_pars:
            print('*** Fixed Parameters: ***')
            for label in fixed_pars:
                print(f' * {label}')

        # Handle shared parameters
        if shared_pars is not None:

            for shared_par in shared_pars:
                if shared_par not in self.model.parameter_labels:
                    raise ValueError(f'Shared parameter "{shared_par}" not in model parameters: {self.model.parameter_labels}')

            shared_parameter_ix = tf.constant(
                [ix for ix, label in enumerate(self.model.parameter_labels) if label in shared_pars], dtype=tf.int32
        )
            voxel_specific_parameters_ix = [ix for ix in parameter_ix if ix not in shared_parameter_ix.numpy()]

            print('*** Shared Parameters: ***')
            for ix in shared_parameter_ix.numpy():
                print(f' * {self.model.parameter_labels[ix]}')

        else:
            voxel_specific_parameters_ix = parameter_ix

        n_meaningful_ts = tf.reduce_sum(tf.cast(meaningful_ts, tf.int32))

        # **Voxel-Specific Parameter Updates**
        voxel_parameter_update_ix, update_parameter_ix = tf.meshgrid(tf.cast(tf.where(meaningful_ts), tf.int32), voxel_specific_parameters_ix)
        voxel_parameter_update_ix = tf.stack((tf.reshape(voxel_parameter_update_ix, tf.size(voxel_parameter_update_ix)),
                            tf.reshape(update_parameter_ix, tf.size(update_parameter_ix))), 1)

        print(f'Number of problematic voxels (mask): {tf.reduce_sum(tf.cast(meaningful_ts == False, tf.int32))}')
        print(f'Number of voxels remaining (mask): {tf.reduce_sum(tf.cast(meaningful_ts == True, tf.int32))}')

        # **Initialize Trainable Parameters**
        trainable_voxel_specific_parameters = tf.Variable(
            initial_value=tf.gather_nd(init_pars, voxel_parameter_update_ix),
            shape=(n_meaningful_ts * len(voxel_specific_parameters_ix),),
            name='voxel_specific_parameters',
            dtype=tf.float32
        )

        if shared_pars is not None:
            # Compute the mean of selected parameters
            shared_parameter_init_values = tf.gather(init_pars, shared_parameter_ix, axis=1)

            if len(shared_parameter_ix.shape) == 0:  # If only one parameter is shared
                shared_parameter_init_values = shared_parameter_init_values[:, tf.newaxis]

            trainable_shared_parameters = tf.Variable(
                initial_value=tf.reduce_mean(shared_parameter_init_values, axis=0),
                shape=(len(shared_parameter_ix),),
                name='shared_parameters',
                dtype=tf.float32)

            # Combine trainable variables
            trainable_variables = [trainable_voxel_specific_parameters, trainable_shared_parameters]

            # Expand shared parameters across all voxels
            voxel_indices = tf.range(n_voxels, dtype=tf.int32)[:, tf.newaxis]  # Shape: [n_voxels, 1]

            # Create a meshgrid to assign shared parameters to every voxel
            voxel_indices, shared_parameter_ix = tf.meshgrid(voxel_indices, shared_parameter_ix, indexing="ij")  # Correct indexing
            shared_parameter_ix_expanded = tf.reshape(shared_parameter_ix, [-1])  # Flatten
            voxel_indices_expanded = tf.reshape(voxel_indices, [-1])  # Flatten only once!

            # Create the correct index tensor for scatter update
            shared_parameter_update_ix = tf.stack([voxel_indices_expanded, shared_parameter_ix_expanded], axis=-1)  # Shape: [n_voxels * n_shared_params, 2]

        else:
            trainable_variables = [trainable_voxel_specific_parameters]

        if store_intermediate_parameters:
            intermediate_parameters = []

        mean_best_r2s = []
        paradigm_ = self.model.stimulus._clean_paradigm(self.paradigm)

        @tf.function
        def get_ssq(parameters):
            predictions = self.model._predict(
                paradigm_[tf.newaxis, ...], parameters[tf.newaxis, ...], None)

            residuals = y - predictions[0]

            ssq = tf.reduce_sum(residuals**2, 0)
            return ssq

        # Optimization Loop
        pbar = range(max_n_iterations)
        if progressbar:
            from tqdm import tqdm
            pbar = tqdm(pbar)

        best_r2 = tf.ones(y.shape[1]) * -1e3
        best_parameters = tf.zeros(init_pars.shape)

        for step in pbar:
            with tf.GradientTape() as tape:
                # Update voxelwise parameters
                parameters = tf.tensor_scatter_nd_update(
                    init_pars, voxel_parameter_update_ix, trainable_voxel_specific_parameters)

                if shared_pars is not None:
                    tiled_trainable_shared_parameters = tf.reshape(
                        tf.tile(trainable_shared_parameters[tf.newaxis, :], [n_voxels, 1]),
                        [-1]
                    )
                    parameters = tf.tensor_scatter_nd_update(
                        parameters,
                        shared_parameter_update_ix,
                        tiled_trainable_shared_parameters)

                untransformed_parameters = self.model._transform_parameters_forward(
                    parameters)

                ssq = get_ssq(untransformed_parameters)
                cost = tf.reduce_sum(ssq)

            gradients = tape.gradient(cost, trainable_variables)
            r2 = (1 - (ssq / ssq_data))

            if shared_pars is None:
                improved_r2s = r2 > best_r2
                best_parameters = tf.where(
                    improved_r2s[:, tf.newaxis], untransformed_parameters, best_parameters)
                best_r2 = tf.where(improved_r2s, r2, best_r2)
            else:
                best_parameters = untransformed_parameters
                best_r2 = r2

            mean_current_r2 = r2[meaningful_ts].numpy().mean()
            mean_best_r2 = best_r2[meaningful_ts].numpy().mean()

            if step >= min_n_iterations:
                r2_diff = mean_best_r2 - mean_best_r2s[np.max((step - lag, 0))]
                if (r2_diff >= 0.0) & (r2_diff < r2_atol):
                    if progressbar:
                        pbar.close()
                    break

            mean_best_r2s.append(mean_best_r2)
            opt.apply_gradients(zip(gradients, trainable_variables))

            if progressbar:
                pbar.set_description(f'Current R2: {mean_current_r2:0.5f}/Best R2: {mean_best_r2:0.5f}')

            if store_intermediate_parameters:
                p = untransformed_parameters.numpy().T
                intermediate_parameters.append(
                    np.reshape(p, np.prod(p.shape)))
                intermediate_parameters[-1] = np.concatenate(
                    (intermediate_parameters[-1], r2), 0)


        self.estimated_parameters = format_parameters(
            best_parameters.numpy(), self.model.parameter_labels)

        self.estimated_parameters.index = self.data.columns

        if not self.estimated_parameters.index.name:
            self.estimated_parameters.index.name = 'source'

        if store_intermediate_parameters:
            columns = pd.MultiIndex.from_product([self.model.parameter_labels + ['r2'],
                                                    np.arange(n_voxels)],
                                                    names=['parameter', 'voxel'])

            self.intermediate_parameters = pd.DataFrame(intermediate_parameters,
                                                        columns=columns,
                                                        index=pd.Index(np.arange(len(intermediate_parameters)),
                                                                        name='step'))

        self.predictions = self.model.predict(self.paradigm, self.estimated_parameters, self.model.weights)
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

    def refine_baseline_and_amplitude(self, parameters, n_iterations=1, positive_amplitude=True, l2_alpha=1e-3):

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

        demeaned_predictions = (predictions -  parameters.loc[:, baseline_ix].T) / parameters.loc[:, amplitude_ix]

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

        parameters.update(new_parameters.loc[ix].astype(np.float32))

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
