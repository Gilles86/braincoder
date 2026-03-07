import pandas as pd
import numpy as np
import datetime
import keras
from keras import ops
import os.path as op
import os
from tqdm.auto import tqdm
from ..utils import format_data, format_parameters, format_paradigm, get_rsq
from ..utils.backend import compute_gradients
import logging


class ParameterFitter(object):
    """Iterative optimizer that estimates model parameters for each voxel."""

    def __init__(self, model, data, paradigm, memory_limit=666666666, log_dir=False):
        self.model = model
        self.data = data.astype(np.float32)
        self.paradigm = model.get_paradigm(paradigm)
        self.memory_limit = memory_limit
        self.log_dir = log_dir

        if log_dir is None:
            log_dir = op.abspath('logs/fit')

        if log_dir is not False:
            if not op.exists(log_dir):
                os.makedirs(log_dir)
            if keras.backend.backend() == 'tensorflow':
                import tensorflow as tf
                self.summary_writer = tf.summary.create_file_writer(
                    op.join(log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))

    def fit(self, max_n_iterations=1000,
            min_n_iterations=100,
            init_pars=None,
            confounds=None,
            optimizer=None,
            fixed_pars=None,
            shared_pars=None,
            store_intermediate_parameters=False,
            r2_atol=0.000001,
            lag=100,
            learning_rate=0.01,
            progressbar=True,
            **kwargs):
        """Run iterative optimization to estimate model parameters for each voxel."""

        n_voxels, n_pars = self.data.shape[1], len(self.model.parameter_labels)

        y = self.data.values

        if optimizer is None:
            opt = keras.optimizers.Adam(learning_rate=learning_rate, **kwargs)

        if init_pars is None:
            init_pars = self.model.get_init_pars(data=y, paradigm=self.paradigm, confounds=confounds)
            print('using get_init_pars')

        init_pars = self.model._get_parameters(init_pars)
        init_pars = np.array(self.model._transform_parameters_backward(init_pars.values.astype(np.float32)))

        ssq_data = ops.sum((y - ops.mean(y, axis=0)[None, :])**2, axis=0)
        meaningful_ts = np.array(ssq_data > 0.0)

        # Handle fixed parameters
        if fixed_pars is None:
            parameter_ix = list(range(n_pars))
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
        shared_parameter_ixs = []
        if shared_pars is not None:
            for shared_par in shared_pars:
                if shared_par not in self.model.parameter_labels:
                    raise ValueError(f'Shared parameter "{shared_par}" not in model parameters: {self.model.parameter_labels}')
            shared_parameter_ixs = [ix for ix, label in enumerate(self.model.parameter_labels)
                                     if label in shared_pars]
            voxel_specific_parameters_ix = [ix for ix in parameter_ix if ix not in shared_parameter_ixs]
            print('*** Shared Parameters: ***')
            for ix in shared_parameter_ixs:
                print(f' * {self.model.parameter_labels[ix]}')
        else:
            voxel_specific_parameters_ix = parameter_ix

        meaningful_ixs = np.where(meaningful_ts)[0]
        print(f'Number of problematic voxels (mask): {(~meaningful_ts).sum()}')
        print(f'Number of voxels remaining (mask): {meaningful_ts.sum()}')

        # Build a boolean mask (n_voxels, n_pars) marking which entries are trainable
        # voxel-specific: only meaningful voxels × voxel-specific params
        voxel_mask_np = np.zeros((n_voxels, n_pars), dtype=np.float32)
        for vi in meaningful_ixs:
            for pi in voxel_specific_parameters_ix:
                voxel_mask_np[vi, pi] = 1.0
        voxel_mask = ops.convert_to_tensor(voxel_mask_np)
        init_pars_t = ops.convert_to_tensor(init_pars)

        # Single (n_voxels, n_pars) variable — only masked positions receive gradients
        trainable_params = keras.Variable(
            init_pars.copy(), name='parameters', dtype='float32')

        trainable_variables = [trainable_params]

        if shared_pars is not None:
            shared_mask_np = np.zeros((n_voxels, n_pars), dtype=np.float32)
            for pi in shared_parameter_ixs:
                shared_mask_np[:, pi] = 1.0
            shared_mask = ops.convert_to_tensor(shared_mask_np)

            init_shared = np.mean(init_pars[:, shared_parameter_ixs], axis=0)
            trainable_shared = keras.Variable(
                init_shared, name='shared_parameters', dtype='float32')
            trainable_variables.append(trainable_shared)

        if store_intermediate_parameters:
            intermediate_parameters = []

        mean_best_r2s = []
        paradigm_ = self.model.stimulus._clean_paradigm(self.paradigm)

        # Precompute selector matrix for shared parameters: (n_pars, n_shared)
        # proj = trainable_shared @ selector^T  →  (1, n_pars) with values at shared columns
        if shared_pars is not None:
            selector_np = np.zeros((n_pars, len(shared_parameter_ixs)), dtype=np.float32)
            for j, pi in enumerate(shared_parameter_ixs):
                selector_np[pi, j] = 1.0
            selector_t = ops.convert_to_tensor(selector_np)  # (n_pars, n_shared)

            def build_parameters():
                proj = ops.matmul(
                    ops.reshape(trainable_shared, (1, -1)),  # (1, n_shared)
                    ops.transpose(selector_t)                # (n_shared, n_pars)
                )                                            # (1, n_pars)
                shared_expanded = ops.tile(proj, [n_voxels, 1])  # (n_voxels, n_pars)
                return (init_pars_t * (1 - voxel_mask - shared_mask)
                        + trainable_params * voxel_mask
                        + shared_expanded * shared_mask)
        else:
            def build_parameters():
                return init_pars_t * (1 - voxel_mask) + trainable_params * voxel_mask

        def get_ssq(parameters):
            predictions = self.model._predict(
                paradigm_[None, ...], parameters[None, ...], None)
            residuals = y - predictions[0]
            return ops.sum(residuals**2, 0)

        pbar = range(max_n_iterations)
        if progressbar:
            from tqdm import tqdm
            pbar = tqdm(pbar)

        best_r2 = ops.ones(y.shape[1]) * -1e3
        best_parameters = ops.zeros(init_pars.shape)

        for step in pbar:
            def loss_fn():
                parameters = build_parameters()
                untransformed = self.model._transform_parameters_forward(parameters)
                ssq = get_ssq(untransformed)
                return ops.sum(ssq)

            cost, gradients = compute_gradients(loss_fn, trainable_variables)

            # Re-compute for tracking (outside gradient tape)
            parameters = build_parameters()
            untransformed_parameters = self.model._transform_parameters_forward(parameters)
            ssq = get_ssq(untransformed_parameters)
            r2 = 1 - ssq / ssq_data

            if shared_pars is None:
                improved_r2s = r2 > best_r2
                best_parameters = ops.where(
                    improved_r2s[:, None], untransformed_parameters, best_parameters)
                best_r2 = ops.where(improved_r2s, r2, best_r2)
            else:
                best_parameters = untransformed_parameters
                best_r2 = r2

            mean_current_r2 = float(r2[meaningful_ts].numpy().mean())
            mean_best_r2 = float(best_r2[meaningful_ts].numpy().mean())

            if step >= min_n_iterations:
                r2_diff = mean_best_r2 - mean_best_r2s[np.max((step - lag, 0))]
                if (r2_diff >= 0.0) and (r2_diff < r2_atol):
                    if progressbar:
                        pbar.close()
                    break

            mean_best_r2s.append(mean_best_r2)
            opt.apply_gradients(zip(gradients, trainable_variables))

            if progressbar:
                pbar.set_description(f'Current R2: {mean_current_r2:0.5f}/Best R2: {mean_best_r2:0.5f}')

            if store_intermediate_parameters:
                p = untransformed_parameters.numpy().T
                intermediate_parameters.append(np.reshape(p, np.prod(p.shape)))
                intermediate_parameters[-1] = np.concatenate(
                    (intermediate_parameters[-1], r2), 0)

        self.estimated_parameters = format_parameters(
            best_parameters.numpy(), self.model.parameter_labels)
        self.estimated_parameters.index = self.data.columns

        if not self.estimated_parameters.index.name:
            self.estimated_parameters.index.name = 'source'

        if store_intermediate_parameters:
            columns = pd.MultiIndex.from_product(
                [self.model.parameter_labels + ['r2'], np.arange(n_voxels)],
                names=['parameter', 'voxel'])
            self.intermediate_parameters = pd.DataFrame(
                intermediate_parameters, columns=columns,
                index=pd.Index(np.arange(len(intermediate_parameters)), name='step'))

        self.predictions = self.model.predict(self.paradigm, self.estimated_parameters, self.model.weights)
        self.r2 = pd.Series(best_r2.numpy(), index=self.data.columns)

        return self.estimated_parameters

    def fit_grid(self, *args, fixed_pars=None,
                 use_correlation_cost=False,
                 positive_amplitude=True, **kwargs):

        n_timepoints, n_voxels = self.data.shape
        chunk_size = self.memory_limit / n_voxels / n_timepoints
        chunk_size = int(kwargs.pop('chunk_size', chunk_size))
        print(f'Working with chunk size of {chunk_size}')

        if fixed_pars is not None:
            raise NotImplementedError()

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
        par_grid = _create_grid(self.model, *grid_args).astype(np.float32)
        par_grid = par_grid.set_index(
            pd.Index(par_grid.index // chunk_size, name='chunk'), append=True)

        n_chunks = ((len(par_grid) - 1) // chunk_size) + 1
        n_pars = par_grid.shape[1]
        n_features = self.data.shape[1]

        logging.info('Built grid of {len(par_grid)} parameter settings...')

        data = self.data.values
        paradigm_ = self.paradigm.values

        if use_correlation_cost:
            print("Using correlation cost!")
            data_demeaned = data - ops.mean(data, axis=0, keepdims=True)
            ssq_data = ops.sum(data_demeaned**2, axis=0, keepdims=True)

            def _cost(par_grid):
                grid_predictions = self.model._predict(paradigm_[None, ...], par_grid[None, ...], None)
                gp_dm = grid_predictions[0] - ops.mean(grid_predictions[0], axis=0, keepdims=True)
                ssq_pred = ops.sum(gp_dm**2, axis=0, keepdims=True)
                r = (ops.sum(gp_dm[:, None, :] * data_demeaned[:, :, None], axis=0, keepdims=True)
                     / ops.sqrt(ssq_pred[:, None, :] * ssq_data[:, :, None]))
                r = r[0]
                if positive_amplitude:
                    r = ops.where(ops.isfinite(r), r, ops.ones_like(r) * -1)
                else:
                    r = ops.where(ops.isfinite(r), r, ops.zeros_like(r))
                    r = r**2
                best_ixs = ops.argmax(r, axis=1)
                return -r, best_ixs
        else:
            def _cost(par_grid):
                grid_predictions = self.model._predict(paradigm_[None, ...], par_grid[None, ...], None)
                ssq = ops.sum(
                    (grid_predictions[0, :, None, :] - data[:, :, None])**2, axis=0)
                best_ixs = ops.argmin(ssq, axis=1)
                return ssq, best_ixs

        best_pars = np.zeros((n_features, n_chunks, n_pars))
        best_cost = np.zeros((n_features, n_chunks))
        vox_ix = np.arange(n_features)

        for chunk, pg in tqdm(par_grid.groupby('chunk')):
            cost_, best_ix = _cost(pg.values)
            cost_np  = np.array(cost_)
            best_ix_np = np.array(best_ix)
            best_cost[:, chunk] = cost_np[vox_ix, best_ix_np]
            best_pars[:, chunk] = np.array(pg.values)[best_ix_np]

        best_chunks = np.array(ops.argmin(best_cost, axis=1))
        best_pars = best_pars[vox_ix, best_chunks]

        best_pars = pd.DataFrame(best_pars, index=self.data.columns,
                                 columns=self.model.parameter_labels).astype(np.float32)

        if not best_pars.index.name:
            best_pars.index.name = 'source'

        return best_pars

    def refine_baseline_and_amplitude(self, parameters, n_iterations=1,
                                       positive_amplitude=True, l2_alpha=1e-3):

        data = self.data
        predictions = self.model.predict(parameters=parameters, paradigm=self.paradigm)
        parameters = parameters.copy()

        if isinstance(parameters.columns, pd.MultiIndex):
            amplitude_ix = ('amplitude_unbounded', 'Intercept')
            baseline_ix  = ('baseline_unbounded', 'Intercept')
            assert (amplitude_ix in parameters.columns and
                    baseline_ix in parameters.columns), "Need parameters with amplitude and baseline"
        else:
            assert (('baseline' in parameters.columns) and
                    ('amplitude' in parameters)), "Need parameters with amplitude and baseline"
            amplitude_ix = 'amplitude'
            baseline_ix  = 'baseline'

        orig_r2 = get_rsq(data, predictions)

        demeaned_predictions = ((predictions - parameters.loc[:, baseline_ix].T)
                                / parameters.loc[:, amplitude_ix])

        X = np.stack((np.ones((predictions.shape[1], predictions.shape[0])),
                      demeaned_predictions.values.T[:, :]), 2).astype(np.float32)
        Y = data.T.values[..., np.newaxis].astype(np.float32)

        # Batched Tikhonov: (X^T X + alpha*I) beta = X^T Y for each voxel
        Xt   = ops.transpose(ops.convert_to_tensor(X), (0, 2, 1))   # (n_vox, 2, n_t)
        X_t  = ops.convert_to_tensor(X)
        Y_t  = ops.convert_to_tensor(Y)
        XtX  = ops.matmul(Xt, X_t)                                  # (n_vox, 2, 2)
        XtY  = ops.matmul(Xt, Y_t)                                  # (n_vox, 2, 1)
        reg  = l2_alpha * ops.eye(2)[None, :, :]
        beta = ops.solve(XtX + reg, XtY).numpy()[..., 0]            # (n_vox, 2)

        Y_ = ops.sum(ops.convert_to_tensor(beta[:, None, :]) * X_t, axis=2).numpy().T

        new_parameters = parameters.copy().astype(np.float32)
        new_parameters.loc[:, baseline_ix]  = beta[:, 0]
        new_parameters.loc[:, amplitude_ix] = beta[:, 1]

        if positive_amplitude:
            new_parameters[amplitude_ix] = np.clip(new_parameters[amplitude_ix], 1e-4, np.inf)

        new_pred = self.model.predict(parameters=new_parameters, paradigm=self.paradigm)
        new_r2 = get_rsq(data, new_pred)

        ix = (new_r2 > orig_r2) & (data.std() != 0.0)
        parameters.update(new_parameters.loc[ix].astype(np.float32))

        if n_iterations == 1:
            return parameters
        else:
            return self.refine_baseline_and_amplitude(parameters, n_iterations - 1)

    def _partly_fit_grid(self, fixed_pars, n_voxels, chunk_size, **kwargs):

        for key in kwargs.keys():
            if key in fixed_pars:
                print(f'Dropping {key} from fixed_pars since it is in the grid parameters')
                fixed_pars = fixed_pars.drop(key, 1)

        if not set(list(kwargs.keys()) + fixed_pars.columns.tolist()) == set(self.model.parameter_labels):
            raise ValueError(
                f'Please provide parameter ranges for all these parameters: {self.model.parameter_labels},'
                'either in the grid or in the fixed_pars')

        chunk_size = chunk_size // self.data.shape[1] + 1

        grid_key_ixs  = [self.model.parameter_labels.index(key) for key in kwargs.keys()]
        init_par_ixs  = [self.model.parameter_labels.index(key) for key in fixed_pars.columns]

        par_grid1 = pd.MultiIndex.from_product(
            kwargs.values(), names=kwargs.keys()).to_frame(index=False)
        par_grid1 = np.repeat(par_grid1.values[np.newaxis, :, :], n_voxels, 0)

        n_par_permutations = par_grid1.shape[1]
        n_chunks   = ((n_par_permutations - 1) // chunk_size) + 1
        n_features = self.data.shape[1]
        n_pars     = len(self.model.parameter_labels)

        par_grid = np.zeros((n_par_permutations, n_features, n_pars), dtype=np.float32)
        for old_ix, new_ix in enumerate(grid_key_ixs):
            par_grid[:, :, new_ix] = par_grid1[:, :, old_ix].T
        for old_ix, new_ix in enumerate(init_par_ixs):
            par_grid[:, :, new_ix] = fixed_pars.values[np.newaxis, :, old_ix]

        best_pars = np.zeros((n_features, n_chunks, n_pars))
        best_ssq  = np.zeros((n_features, n_chunks))
        vox_ix    = np.arange(n_features)

        data = self.data.values
        paradigm_ = self.paradigm.values

        def _get_ssq_for_predictions(par_grid):
            grid_predictions = self.model._predict(paradigm_[None, :, :], par_grid, None)
            resid = data[None, ...] - grid_predictions
            ssq   = ops.sum(resid**2, 1)
            return ssq, ops.argmin(ssq, 0)

        for chunk in tqdm(range(n_chunks)):
            pg = par_grid[chunk * chunk_size:(chunk + 1) * chunk_size, :, :]
            print(pg.shape)
            ssq_, best_ix = _get_ssq_for_predictions(pg)
            ssq_np   = np.array(ssq_)
            best_ix_np = np.array(best_ix)
            gather_ix = np.stack((best_ix_np, vox_ix), 1)
            best_ssq[:, chunk]    = ssq_np[gather_ix[:, 0], gather_ix[:, 1]]
            best_pars[:, chunk, :] = pg[gather_ix[:, 0], gather_ix[:, 1]]

        best_chunks = np.array(ops.argmin(best_ssq, axis=1))
        best_pars = best_pars[vox_ix, best_chunks]

        return pd.DataFrame(best_pars, index=self.data.columns,
                            columns=self.model.parameter_labels).astype(np.float32)

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
