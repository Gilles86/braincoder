"""Per-voxel iterative pRF / encoding-model fitter."""
import datetime
import logging
import os
import os.path as op

import numpy as np
import pandas as pd
import keras
from keras import ops
from tqdm.auto import tqdm

from ..utils import format_data, format_parameters, format_paradigm, get_rsq
from ..utils.backend import compute_gradients, softplus_inverse


_VALID_NOISE_MODELS = ('ssq', 'gaussian')


class ParameterFitter:
    """Iterative optimizer that estimates model parameters for each voxel.

    Single trainable ``(n_voxels, n_pars)`` keras Variable; only the
    masked positions (meaningful voxels × non-fixed params) receive
    gradients. Shared parameters are realized by a smaller Variable
    that's broadcast back across voxels via a selector matrix.

    Parameters
    ----------
    model : EncodingModel
    data : pd.DataFrame
        Timeseries, shape ``(n_timepoints, n_voxels)``.
    paradigm : array-like
        Stimulus paradigm; passed through ``model.get_paradigm``.
    memory_limit : int
        Soft budget for the grid-fit chunk size.
    log_dir : str | None | False
        TensorBoard log dir. ``False`` (default) disables logging.
    """

    def __init__(self, model, data, paradigm,
                 memory_limit=666666666, log_dir=False):
        self.model = model
        self.data = data.astype(np.float32)
        self.paradigm = model.get_paradigm(paradigm)
        self.memory_limit = memory_limit
        self.log_dir = log_dir

        if log_dir is None:
            log_dir = op.abspath('logs/fit')

        if log_dir is not False:
            os.makedirs(log_dir, exist_ok=True)
            if keras.backend.backend() == 'tensorflow':
                import tensorflow as tf
                self.summary_writer = tf.summary.create_file_writer(
                    op.join(log_dir,
                             datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))

    # ------------------------------------------------------------ fit
    def fit(self, max_n_iterations=1000,
            min_n_iterations=100,
            init_pars=None,
            confounds=None,
            optimizer=None,
            fixed_pars=None,
            shared_pars=None,
            noise_model='ssq',
            store_intermediate_parameters=False,
            r2_atol=1e-6,
            lag=100,
            learning_rate=0.01,
            progressbar=True,
            **kwargs):
        """Estimate per-voxel parameters by iterative optimization.

        Parameters
        ----------
        noise_model : {'ssq', 'gaussian'}, default ``'ssq'``
            Loss function. ``'ssq'`` minimizes the unweighted
            sum-of-squared-residuals (MLE under homoscedastic Gaussian
            noise). ``'gaussian'`` minimizes a per-voxel Gaussian
            negative log-likelihood with a *free* per-voxel σ²ᵥ; the
            θ-gradient is auto-rescaled by per-voxel noise, which under
            a shared Adam optimizer gives every voxel comparable
            per-iteration progress. Same fixed points as ``'ssq'``,
            but much faster convergence in practice on real fMRI
            (heteroskedastic noise). When set, ``self.estimated_sigma2``
            holds the fitted per-voxel σ² after ``fit()`` returns.
        fixed_pars, shared_pars : list[str] | None
            Parameter names to hold at their init value (``fixed_pars``)
            or to tie to a single shared value across voxels
            (``shared_pars``).
        r2_atol, lag, min_n_iterations : float, int, int
            Early-stop: stop when R² has improved by less than ``r2_atol``
            over the last ``lag`` steps, after ``min_n_iterations``.
        **kwargs
            Forwarded to ``keras.optimizers.Adam`` when ``optimizer is
            None``.

        Returns
        -------
        pd.DataFrame
            Per-voxel parameter estimates, indexed by voxel.
        """
        if noise_model not in _VALID_NOISE_MODELS:
            raise ValueError(
                f"noise_model must be one of {_VALID_NOISE_MODELS}, "
                f"got {noise_model!r}")

        n_voxels, n_pars = self.data.shape[1], len(self.model.parameter_labels)
        n_timepoints = self.data.shape[0]

        y = ops.convert_to_tensor(self.data.values, dtype='float32')
        ssq_data = ops.sum((y - ops.mean(y, axis=0)[None, :]) ** 2, axis=0)
        meaningful_ts = ops.convert_to_numpy(ssq_data > 0.0)
        meaningful_ixs = np.where(meaningful_ts)[0]
        print(f'Number of problematic voxels (mask): {(~meaningful_ts).sum()}')
        print(f'Number of voxels remaining (mask): {meaningful_ts.sum()}')

        # ---- init parameters in unconstrained space ----------------
        if init_pars is None:
            init_pars = self.model.get_init_pars(
                data=y, paradigm=self.paradigm, confounds=confounds)
            print('using get_init_pars')
        init_pars = self.model._get_parameters(init_pars)
        init_pars = ops.convert_to_numpy(
            self.model._transform_parameters_backward(
                init_pars.values.astype(np.float32)))
        init_pars_t = ops.convert_to_tensor(init_pars)

        # ---- which parameters are trainable, shared, fixed ----------
        parameter_ix, shared_parameter_ixs, voxel_specific_ixs = \
            self._resolve_param_partition(n_pars, fixed_pars, shared_pars)
        voxel_mask, shared_mask = self._build_masks(
            n_voxels, n_pars, meaningful_ixs, voxel_specific_ixs,
            shared_parameter_ixs)

        # ---- trainable Variables -----------------------------------
        trainable_params = keras.Variable(
            init_pars.copy(), name='parameters', dtype='float32')
        trainable_variables = [trainable_params]

        if shared_pars is not None:
            init_shared = np.mean(init_pars[:, shared_parameter_ixs], axis=0)
            trainable_shared = keras.Variable(
                init_shared, name='shared_parameters', dtype='float32')
            trainable_variables.append(trainable_shared)
            selector_t = ops.convert_to_tensor(
                self._selector_matrix(n_pars, shared_parameter_ixs))
        else:
            trainable_shared = None
            selector_t = None

        # Per-voxel log σ²ᵥ for Gaussian noise model.
        # Initialized from initial residuals; clamped to >= 1e-4 to
        # avoid log(0). Stored on softplus-inverse scale so the live
        # σ²ᵥ = softplus(log_sigma2) is positive automatically.
        if noise_model == 'gaussian':
            with_init_pred = self.model._predict(
                ops.convert_to_tensor(self.paradigm.values, dtype='float32')[None, ...],
                self.model._transform_parameters_forward(init_pars_t)[None, ...],
                None)[0]
            init_resid = y - with_init_pred
            init_sigma2 = ops.convert_to_numpy(
                ops.mean(init_resid ** 2, axis=0))
            init_sigma2 = np.maximum(init_sigma2, 1e-4).astype(np.float32)
            log_sigma2 = keras.Variable(
                ops.convert_to_numpy(softplus_inverse(
                    ops.convert_to_tensor(init_sigma2, dtype='float32'))),
                name='log_sigma2', dtype='float32')
            trainable_variables.append(log_sigma2)
        else:
            log_sigma2 = None

        # ---- closures used inside the optimization loop -------------
        paradigm_clean = self.model.stimulus._clean_paradigm(self.paradigm)
        build_parameters = self._make_parameter_assembler(
            init_pars_t, voxel_mask, shared_mask,
            trainable_params, trainable_shared, selector_t, n_voxels)

        def get_ssq(parameters):
            predictions = self.model._predict(
                paradigm_clean[None, ...], parameters[None, ...], None)
            return ops.sum((y - predictions[0]) ** 2, axis=0)

        # ---- optimizer ---------------------------------------------
        if optimizer is None:
            opt = keras.optimizers.Adam(learning_rate=learning_rate, **kwargs)
        else:
            opt = optimizer

        # ---- loss --------------------------------------------------
        # Both losses produce the same per-voxel fixed point. They
        # differ only in the gradient magnitudes (and SSQ has no σ²
        # state); see CHANGELOG 0.6 / classical_vs_ml_convergence
        # report for why Gaussian converges much faster on real data.
        if noise_model == 'ssq':
            def loss_fn():
                pars = build_parameters()
                native = self.model._transform_parameters_forward(pars)
                return ops.sum(get_ssq(native))
        else:  # 'gaussian'
            _LOG_2PI = float(np.log(2.0 * np.pi))
            meaningful_mask = ops.convert_to_tensor(
                meaningful_ts.astype(np.float32))
            n_t_f = ops.cast(n_timepoints, 'float32')

            def loss_fn():
                pars = build_parameters()
                native = self.model._transform_parameters_forward(pars)
                ssq = get_ssq(native)
                sigma2 = ops.softplus(log_sigma2)
                nll = 0.5 * (n_t_f * (_LOG_2PI + ops.log(sigma2))
                              + ssq / sigma2)
                # Only sum over meaningful voxels.
                return ops.sum(nll * meaningful_mask)

        # ---- main loop ---------------------------------------------
        pbar = range(max_n_iterations)
        if progressbar:
            pbar = tqdm(pbar)

        best_r2 = ops.ones(y.shape[1]) * -1e3
        best_parameters = ops.zeros(init_pars.shape)
        intermediate_parameters = [] if store_intermediate_parameters else None
        mean_best_r2s = []

        self._print_fit_header(parameter_ix, fixed_pars, shared_parameter_ixs)

        for step in pbar:
            _, gradients = compute_gradients(loss_fn, trainable_variables)

            # Re-compute outside the gradient tape for tracking.
            pars = build_parameters()
            untransformed = self.model._transform_parameters_forward(pars)
            ssq = get_ssq(untransformed)
            r2 = 1 - ssq / ssq_data

            # When no parameter is shared, track the best per-voxel
            # state seen so far. With shared params the "best" needs
            # to be defined jointly (the shared par changes for every
            # voxel simultaneously), so just take the current state.
            if shared_pars is None:
                improved = r2 > best_r2
                best_parameters = ops.where(
                    improved[:, None], untransformed, best_parameters)
                best_r2 = ops.where(improved, r2, best_r2)
            else:
                best_parameters = untransformed
                best_r2 = r2

            mean_current_r2 = float(ops.convert_to_numpy(
                r2[meaningful_ts]).mean())
            mean_best_r2 = float(ops.convert_to_numpy(
                best_r2[meaningful_ts]).mean())

            # Early-stop on R²-plateau.
            if step >= min_n_iterations:
                r2_diff = mean_best_r2 - mean_best_r2s[
                    max(step - lag, 0)]
                if 0.0 <= r2_diff < r2_atol:
                    if progressbar:
                        pbar.close()
                    break

            mean_best_r2s.append(mean_best_r2)
            opt.apply_gradients(zip(gradients, trainable_variables))

            if progressbar:
                pbar.set_description(
                    f'Current R2: {mean_current_r2:0.5f}/'
                    f'Best R2: {mean_best_r2:0.5f}')

            if store_intermediate_parameters:
                p = ops.convert_to_numpy(untransformed).T
                intermediate_parameters.append(np.concatenate(
                    (np.reshape(p, np.prod(p.shape)),
                     ops.convert_to_numpy(r2)), 0))

        # ---- assemble outputs --------------------------------------
        self.estimated_parameters = format_parameters(
            ops.convert_to_numpy(best_parameters),
            self.model.parameter_labels)
        self.estimated_parameters.index = self.data.columns
        if not self.estimated_parameters.index.name:
            self.estimated_parameters.index.name = 'source'

        if noise_model == 'gaussian':
            self.estimated_sigma2 = pd.Series(
                ops.convert_to_numpy(ops.softplus(log_sigma2)),
                index=self.data.columns, name='sigma2')
        else:
            self.estimated_sigma2 = None

        if store_intermediate_parameters:
            columns = pd.MultiIndex.from_product(
                [self.model.parameter_labels + ['r2'], np.arange(n_voxels)],
                names=['parameter', 'voxel'])
            self.intermediate_parameters = pd.DataFrame(
                intermediate_parameters, columns=columns,
                index=pd.Index(np.arange(len(intermediate_parameters)),
                                name='step'))

        self.predictions = self.model.predict(
            self.paradigm, self.estimated_parameters, self.model.weights)
        self.r2 = pd.Series(ops.convert_to_numpy(best_r2),
                             index=self.data.columns)

        return self.estimated_parameters

    # ------------------------------------------------------------ helpers
    def _resolve_param_partition(self, n_pars, fixed_pars, shared_pars):
        """Validate fixed/shared kwargs; return (trainable, shared, voxel-specific) indices."""
        labels = self.model.parameter_labels
        if fixed_pars is None:
            parameter_ix = list(range(n_pars))
        else:
            for p in fixed_pars:
                if p not in labels:
                    raise ValueError(
                        f'Fixed parameter "{p}" not in model parameters: {labels}')
            parameter_ix = [ix for ix, lbl in enumerate(labels)
                            if lbl not in fixed_pars]

        if shared_pars is None:
            shared_parameter_ixs = []
            voxel_specific_ixs = parameter_ix
        else:
            for p in shared_pars:
                if p not in labels:
                    raise ValueError(
                        f'Shared parameter "{p}" not in model parameters: {labels}')
            shared_parameter_ixs = [ix for ix, lbl in enumerate(labels)
                                     if lbl in shared_pars]
            voxel_specific_ixs = [ix for ix in parameter_ix
                                   if ix not in shared_parameter_ixs]
        return parameter_ix, shared_parameter_ixs, voxel_specific_ixs

    @staticmethod
    def _build_masks(n_voxels, n_pars, meaningful_ixs,
                     voxel_specific_ixs, shared_parameter_ixs):
        """``(n_voxels, n_pars)`` boolean masks for voxel-specific +
        shared trainable positions. Voxels with zero-variance data
        contribute neither (their gradients are masked out)."""
        voxel_mask = np.zeros((n_voxels, n_pars), dtype=np.float32)
        if len(voxel_specific_ixs):
            voxel_mask[np.ix_(meaningful_ixs, voxel_specific_ixs)] = 1.0
        voxel_mask = ops.convert_to_tensor(voxel_mask)

        if shared_parameter_ixs:
            shared_mask = np.zeros((n_voxels, n_pars), dtype=np.float32)
            shared_mask[:, shared_parameter_ixs] = 1.0
            shared_mask = ops.convert_to_tensor(shared_mask)
        else:
            shared_mask = None
        return voxel_mask, shared_mask

    @staticmethod
    def _selector_matrix(n_pars, shared_parameter_ixs):
        """``(n_pars, n_shared)`` selector that scatters a length-
        ``n_shared`` vector back to the right columns of an
        ``(n_voxels, n_pars)`` array."""
        s = np.zeros((n_pars, len(shared_parameter_ixs)), dtype=np.float32)
        for j, pi in enumerate(shared_parameter_ixs):
            s[pi, j] = 1.0
        return s

    @staticmethod
    def _make_parameter_assembler(init_pars_t, voxel_mask, shared_mask,
                                   trainable_params, trainable_shared,
                                   selector_t, n_voxels):
        """Return a closure that assembles the live ``(n_voxels, n_pars)``
        parameter tensor from the trainable Variables + init values."""
        if trainable_shared is None:
            def build():
                return (init_pars_t * (1.0 - voxel_mask)
                        + trainable_params * voxel_mask)
        else:
            def build():
                proj = ops.matmul(
                    ops.reshape(trainable_shared, (1, -1)),
                    ops.transpose(selector_t))           # (1, n_pars)
                shared_expanded = ops.tile(proj, [n_voxels, 1])
                return (init_pars_t * (1.0 - voxel_mask - shared_mask)
                        + trainable_params * voxel_mask
                        + shared_expanded * shared_mask)
        return build

    def _print_fit_header(self, parameter_ix, fixed_pars,
                          shared_parameter_ixs):
        labels = self.model.parameter_labels
        print('*** Fitting: ***')
        for ix in parameter_ix:
            print(f' * {labels[ix]}')
        if fixed_pars:
            print('*** Fixed Parameters: ***')
            for lbl in fixed_pars:
                print(f' * {lbl}')
        if shared_parameter_ixs:
            print('*** Shared Parameters: ***')
            for ix in shared_parameter_ixs:
                print(f' * {labels[ix]}')

    # ------------------------------------------------------------ grid fit
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

        if set(kwargs.keys()) != set(self.model.parameter_labels):
            raise ValueError(
                f'Please provide parameter ranges for all of: '
                f'{self.model.parameter_labels}')

        def _create_grid(*grid_args):
            return pd.MultiIndex.from_product(
                grid_args,
                names=self.model.parameter_labels).to_frame(index=False)

        grid_args = [kwargs[k] for k in self.model.parameter_labels]
        par_grid = _create_grid(*grid_args).astype(np.float32)
        par_grid = par_grid.set_index(
            pd.Index(par_grid.index // chunk_size, name='chunk'), append=True)

        n_chunks = ((len(par_grid) - 1) // chunk_size) + 1
        n_pars = par_grid.shape[1]
        n_features = self.data.shape[1]
        logging.info(f'Built grid of {len(par_grid)} parameter settings...')

        data = ops.convert_to_tensor(self.data.values, dtype='float32')
        paradigm_ = self.paradigm.values

        if use_correlation_cost:
            print("Using correlation cost!")
            data_demeaned = data - ops.mean(data, axis=0, keepdims=True)
            ssq_data = ops.sum(data_demeaned ** 2, axis=0, keepdims=True)

            def _cost(pg):
                pred = self.model._predict(paradigm_[None, ...],
                                            pg[None, ...], None)
                gp_dm = pred[0] - ops.mean(pred[0], axis=0, keepdims=True)
                ssq_pred = ops.sum(gp_dm ** 2, axis=0, keepdims=True)
                r = (ops.sum(gp_dm[:, None, :] * data_demeaned[:, :, None],
                             axis=0, keepdims=True)
                     / ops.sqrt(ssq_pred[:, None, :] * ssq_data[:, :, None]))
                r = r[0]
                if positive_amplitude:
                    r = ops.where(ops.isfinite(r), r, ops.ones_like(r) * -1)
                else:
                    r = ops.where(ops.isfinite(r), r, ops.zeros_like(r))
                    r = r ** 2
                return -r, ops.argmax(r, axis=1)
        else:
            def _cost(pg):
                pred = self.model._predict(paradigm_[None, ...],
                                            pg[None, ...], None)
                ssq = ops.sum(
                    (pred[0, :, None, :] - data[:, :, None]) ** 2, axis=0)
                return ssq, ops.argmin(ssq, axis=1)

        best_pars = np.zeros((n_features, n_chunks, n_pars))
        best_cost = np.zeros((n_features, n_chunks))
        vox_ix = np.arange(n_features)

        for chunk, pg in tqdm(par_grid.groupby('chunk')):
            cost_, best_ix = _cost(pg.values)
            cost_np = ops.convert_to_numpy(cost_)
            best_ix_np = ops.convert_to_numpy(best_ix)
            best_cost[:, chunk] = cost_np[vox_ix, best_ix_np]
            best_pars[:, chunk] = np.array(pg.values)[best_ix_np]

        best_chunks = ops.convert_to_numpy(
            ops.argmin(ops.convert_to_tensor(best_cost, dtype='float32'),
                       axis=1))
        best_pars = best_pars[vox_ix, best_chunks]

        out = pd.DataFrame(best_pars, index=self.data.columns,
                            columns=self.model.parameter_labels
                            ).astype(np.float32)
        if not out.index.name:
            out.index.name = 'source'
        return out

    # ------------------------------------------------------------ refine
    def refine_baseline_and_amplitude(self, parameters, n_iterations=1,
                                       positive_amplitude=True,
                                       l2_alpha=1e-3):
        data = self.data
        predictions = self.model.predict(parameters=parameters,
                                          paradigm=self.paradigm)
        parameters = parameters.copy()

        if isinstance(parameters.columns, pd.MultiIndex):
            amplitude_ix = ('amplitude_unbounded', 'Intercept')
            baseline_ix = ('baseline_unbounded', 'Intercept')
            assert (amplitude_ix in parameters.columns
                    and baseline_ix in parameters.columns), \
                "Need parameters with amplitude and baseline"
        else:
            assert (('baseline' in parameters.columns)
                    and ('amplitude' in parameters)), \
                "Need parameters with amplitude and baseline"
            amplitude_ix = 'amplitude'
            baseline_ix = 'baseline'

        orig_r2 = get_rsq(data, predictions)
        demeaned = ((predictions - parameters.loc[:, baseline_ix].T)
                    / parameters.loc[:, amplitude_ix])

        X = np.stack((np.ones((predictions.shape[1], predictions.shape[0])),
                      demeaned.values.T[:, :]), 2).astype(np.float32)
        Y = data.T.values[..., np.newaxis].astype(np.float32)

        Xt = ops.transpose(ops.convert_to_tensor(X), (0, 2, 1))
        X_t = ops.convert_to_tensor(X)
        Y_t = ops.convert_to_tensor(Y)
        XtX = ops.matmul(Xt, X_t)
        XtY = ops.matmul(Xt, Y_t)
        reg = l2_alpha * ops.eye(2)[None, :, :]
        beta = ops.convert_to_numpy(ops.solve(XtX + reg, XtY))[..., 0]

        new_parameters = parameters.copy().astype(np.float32)
        new_parameters.loc[:, baseline_ix] = beta[:, 0]
        new_parameters.loc[:, amplitude_ix] = beta[:, 1]
        if positive_amplitude:
            new_parameters[amplitude_ix] = np.clip(
                new_parameters[amplitude_ix], 1e-4, np.inf)

        new_pred = self.model.predict(parameters=new_parameters,
                                       paradigm=self.paradigm)
        new_r2 = get_rsq(data, new_pred)
        ix = (new_r2 > orig_r2) & (data.std() != 0.0)
        parameters.update(new_parameters.loc[ix].astype(np.float32))

        if n_iterations == 1:
            return parameters
        return self.refine_baseline_and_amplitude(
            parameters, n_iterations - 1)

    # ------------------------------------------------------------ misc API
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
        self._data = None if data is None else format_data(data)

    @property
    def paradigm(self):
        return self._paradigm

    @paradigm.setter
    def paradigm(self, paradigm):
        self._paradigm = format_paradigm(paradigm)

    # ------------------------------------------------------------ partial grid
    def _partly_fit_grid(self, fixed_pars, n_voxels, chunk_size, **kwargs):
        # Kept verbatim — only used by an experimental code path.
        for key in list(kwargs):
            if key in fixed_pars:
                print(f'Dropping {key} from fixed_pars since it is in '
                      f'the grid parameters')
                fixed_pars = fixed_pars.drop(key, 1)

        if set(list(kwargs) + fixed_pars.columns.tolist()) != \
                set(self.model.parameter_labels):
            raise ValueError(
                f'Please provide parameter ranges for all of: '
                f'{self.model.parameter_labels}, either in the grid '
                f'or in fixed_pars')

        chunk_size = chunk_size // self.data.shape[1] + 1
        grid_key_ixs = [self.model.parameter_labels.index(k)
                         for k in kwargs.keys()]
        init_par_ixs = [self.model.parameter_labels.index(k)
                         for k in fixed_pars.columns]

        par_grid1 = pd.MultiIndex.from_product(
            kwargs.values(), names=kwargs.keys()).to_frame(index=False)
        par_grid1 = np.repeat(par_grid1.values[np.newaxis, :, :], n_voxels, 0)

        n_perms = par_grid1.shape[1]
        n_chunks = ((n_perms - 1) // chunk_size) + 1
        n_features = self.data.shape[1]
        n_pars = len(self.model.parameter_labels)

        par_grid = np.zeros((n_perms, n_features, n_pars), dtype=np.float32)
        for old, new in enumerate(grid_key_ixs):
            par_grid[:, :, new] = par_grid1[:, :, old].T
        for old, new in enumerate(init_par_ixs):
            par_grid[:, :, new] = fixed_pars.values[np.newaxis, :, old]

        best_pars = np.zeros((n_features, n_chunks, n_pars))
        best_ssq = np.zeros((n_features, n_chunks))
        vox_ix = np.arange(n_features)

        data = self.data.values
        paradigm_ = self.paradigm.values

        def _ssq(pg):
            pred = self.model._predict(paradigm_[None, :, :], pg, None)
            return ops.sum((data[None, ...] - pred) ** 2, axis=1), \
                   ops.argmin(ops.sum((data[None, ...] - pred) ** 2, axis=1),
                              axis=0)

        for chunk in tqdm(range(n_chunks)):
            pg = par_grid[chunk * chunk_size:(chunk + 1) * chunk_size]
            ssq_, best_ix = _ssq(pg)
            ssq_np = ops.convert_to_numpy(ssq_)
            best_ix_np = ops.convert_to_numpy(best_ix)
            gather = np.stack((best_ix_np, vox_ix), 1)
            best_ssq[:, chunk] = ssq_np[gather[:, 0], gather[:, 1]]
            best_pars[:, chunk, :] = pg[gather[:, 0], gather[:, 1]]

        best_chunks = ops.convert_to_numpy(ops.argmin(best_ssq, axis=1))
        best_pars = best_pars[vox_ix, best_chunks]
        return pd.DataFrame(best_pars, index=self.data.columns,
                            columns=self.model.parameter_labels
                            ).astype(np.float32)
