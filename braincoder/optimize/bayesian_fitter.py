"""Empirical-Bayes pRF fitter with a GP prior on one or more parameters.

Follows the three-stage recipe in Daghlian et al. (2025):

  1. Classical SSQ fit per vertex (delegated to ``ParameterFitter``).
  2. For each parameter that has a prior attached, fit the GP hyperparameters
     (lengthscale, variance, nugget) by MLE on the classical estimates.
  3. MAP fit: jointly optimize the model parameters under a Gaussian
     likelihood (per-vertex sigma) plus the GP log-prior. The hyperparameters
     from stage 2 are held fixed.

The result is a regularized parameter map that borrows strength from the
spatial structure of the cortex while still respecting the data.
"""

import numpy as np
import pandas as pd
import keras
from keras import ops
from tqdm.auto import tqdm

from ..utils import format_parameters
from ..utils.backend import compute_gradients, softplus_inverse
from .parameter_fitter import ParameterFitter


_LOG_2PI = float(np.log(2.0 * np.pi))


class BayesianParameterFitter(object):
    """Three-stage empirical-Bayes fitter with a spatial GP prior.

    Parameters
    ----------
    model : EncodingModel
    data : DataFrame (n_timepoints, n_vertices)
    paradigm : DataFrame or array
        Stimulus paradigm, same convention as ``ParameterFitter``.
    priors : dict[str, GeodesicGPPrior]
        Mapping from parameter label to a prior. The number of vertices
        in each prior must equal the number of columns in ``data``.

    Notes
    -----
    The GP prior is applied to the parameter's *native* (constrained)
    values — e.g. the linear-space mu of a LogGaussianPRF — matching
    Daghlian et al.'s convention.
    """

    def __init__(self, model, data, paradigm, priors):
        if not isinstance(priors, dict):
            raise ValueError("priors must be a dict (use {} for a no-prior "
                              "ML fit with per-vertex sigma)")

        n_vx = data.shape[1]
        for name, prior in priors.items():
            if name not in model.parameter_labels:
                raise ValueError(
                    f"Prior on '{name}', but model parameters are "
                    f"{model.parameter_labels}")
            if prior.n_vx != n_vx:
                raise ValueError(
                    f"Prior on '{name}' has n_vx={prior.n_vx}, data has "
                    f"{n_vx} columns")

        self.model = model
        self.data = data.astype(np.float32)
        self.paradigm = model.get_paradigm(paradigm)
        self.priors = priors

        # Filled in by the corresponding stage.
        self.classical_estimates = None
        self.classical_r2 = None
        self.hyperparameter_history = None
        self.map_estimates = None
        self.map_sigma = None

    # ------------------------------------------------------------ public API

    def fit(self,
            max_n_iterations=500,
            learning_rate=0.01,
            classical_kwargs=None,
            hyperparam_kwargs=None,
            map_kwargs=None,
            shared_lengthscale=False,
            progressbar=True):
        """Run all three stages and return the MAP parameter estimates."""
        classical_kwargs = dict(classical_kwargs or {})
        hyperparam_kwargs = dict(hyperparam_kwargs or {})
        map_kwargs = dict(map_kwargs or {})

        self.fit_classical(progressbar=progressbar, **classical_kwargs)
        self.fit_hyperparameters(progressbar=progressbar,
                                  shared_lengthscale=shared_lengthscale,
                                  **hyperparam_kwargs)
        return self.fit_map(max_n_iterations=max_n_iterations,
                            learning_rate=learning_rate,
                            progressbar=progressbar,
                            **map_kwargs)

    # ----------------------------------------------------------- stage 1

    def fit_classical(self, init_pars=None, progressbar=True, **kwargs):
        """Stage 1 — classical per-vertex SSQ fit via ``ParameterFitter``."""
        fitter = ParameterFitter(self.model, self.data, self.paradigm,
                                 log_dir=False)
        self.classical_estimates = fitter.fit(
            init_pars=init_pars, progressbar=progressbar, **kwargs)
        self.classical_r2 = fitter.r2
        self._classical_fitter = fitter
        return self.classical_estimates

    # ----------------------------------------------------------- stage 2

    def fit_hyperparameters(self, progressbar=True,
                            shared_lengthscale=False, **kwargs):
        """Stage 2 — fit each prior's hyperparameters by MLE on stage-1 values.

        No-op when ``priors`` is empty (ML mode).

        If ``shared_lengthscale=True`` and there are ≥2 priors, ties all
        priors' ``_log_lengthscale`` Variables to a single shared one
        and runs **joint** MLE over (shared l, per-prior v, per-prior
        n). The motivation: the "topographic scale" of cortex is a
        property of the surface, not of which pRF parameter you're
        looking at — so adjacent voxels should be similar in *all*
        parameters at the same ~mm scale. Sharing also regularizes
        when the data is too thin to identify four lengthscales
        independently.
        """
        if self.classical_estimates is None:
            raise RuntimeError(
                "Run fit_classical() before fit_hyperparameters()")

        self.hyperparameter_history = {}
        if shared_lengthscale and len(self.priors) >= 2:
            return self._fit_hyperparameters_shared(progressbar=progressbar,
                                                     **kwargs)

        for name, prior in self.priors.items():
            values = self.classical_estimates[name].values.astype(np.float32)
            self.hyperparameter_history[name] = prior.fit_hyperparameters(
                values, progressbar=progressbar, **kwargs)

        return {name: prior.hyperparameters
                for name, prior in self.priors.items()}

    def _fit_hyperparameters_shared(self, max_n_iterations=500,
                                     learning_rate=0.05, tol=1e-4,
                                     patience=20, progressbar=True):
        """Joint MLE with one shared log_lengthscale + per-prior log_variance,
        log_nugget. Mutates each prior so that its ``_log_lengthscale``
        is the same Variable instance.
        """
        from tqdm.auto import tqdm
        from keras import ops
        import keras
        from ..utils.backend import compute_gradients

        priors_list = list(self.priors.values())
        shared_var = priors_list[0]._log_lengthscale
        for p in priors_list[1:]:
            p._log_lengthscale = shared_var

        trainable_vars = [shared_var]
        for p in priors_list:
            trainable_vars.extend([p._log_variance, p._log_nugget])

        values_tensors = {}
        for name, prior in self.priors.items():
            v = self.classical_estimates[name].values.astype(np.float32)
            values_tensors[name] = ops.convert_to_tensor(v, dtype='float32')

        opt = keras.optimizers.Adam(learning_rate=learning_rate)

        def joint_loss():
            total = 0.0
            for name, prior in self.priors.items():
                v_arr = values_tensors[name]
                total = total - prior._log_prob_tensor(
                    v_arr,
                    prior._log_lengthscale,
                    prior._log_variance,
                    prior._log_nugget,
                )
            return total

        history = []
        best = float('inf')
        best_state = [ops.convert_to_numpy(v) for v in trainable_vars]
        stall = 0

        pbar = range(max_n_iterations)
        if progressbar:
            pbar = tqdm(pbar, desc='GP hyperparams (shared l)')

        for step in pbar:
            loss, grads = compute_gradients(joint_loss, trainable_vars)
            opt.apply_gradients(zip(grads, trainable_vars))
            loss_val = float(ops.convert_to_numpy(loss))
            history.append(loss_val)
            if loss_val < best - tol:
                best = loss_val
                best_state = [ops.convert_to_numpy(v) for v in trainable_vars]
                stall = 0
            else:
                stall += 1
            if progressbar:
                pbar.set_description(
                    f'GP joint nLL {loss_val:.3f} | '
                    f'l={priors_list[0].lengthscale:.2f}')
            if stall >= patience:
                break

        for var, val in zip(trainable_vars, best_state):
            var.assign(val)

        shared_l = float(ops.convert_to_numpy(ops.softplus(shared_var)))
        self.hyperparameter_history['_shared'] = dict(
            history=np.asarray(history),
            best_neg_log_lik=best,
            shared_lengthscale=shared_l,
        )
        for name, prior in self.priors.items():
            self.hyperparameter_history[name] = dict(
                hyperparameters=prior.hyperparameters,
                shared_lengthscale=True,
            )

        return {name: prior.hyperparameters
                for name, prior in self.priors.items()}

    # ----------------------------------------------------------- stage 3

    def fit_map(self,
                max_n_iterations=500,
                learning_rate=0.01,
                init_pars=None,
                fixed_pars=None,
                tol=1e-4,
                patience=30,
                clipnorm=1.0,
                progressbar=True):
        """Stage 3 — joint MAP fit with Gaussian likelihood + GP log-prior.

        ``clipnorm`` (default 1.0) clips the global gradient norm at
        each Adam step. Belt-and-braces against the occasional
        run-away update that would otherwise feed enormous values
        into the Mahalanobis term and NaN out subsequent iterations.
        Set to ``None`` to disable.
        """
        if self.classical_estimates is None:
            raise RuntimeError("Run fit_classical() before fit_map()")

        if init_pars is None:
            init_pars = self.classical_estimates

        n_t, n_vx = self.data.shape
        n_pars = len(self.model.parameter_labels)

        y = ops.convert_to_tensor(self.data.values, dtype='float32')

        # Initial parameters in unconstrained (softplus-inverse, etc.) space.
        init_native = self.model._get_parameters(init_pars).values.astype(
            np.float32)
        init_uncon = ops.convert_to_numpy(
            self.model._transform_parameters_backward(init_native))

        # Mask: only meaningful (nonzero-variance) vertices get gradients.
        ssq_data = ops.sum((y - ops.mean(y, axis=0)[None, :]) ** 2, axis=0)
        meaningful_ts = ops.convert_to_numpy(ssq_data > 0.0)

        # Resolve fixed_pars → which columns of `params_var` are trainable.
        fixed_pars = list(fixed_pars or [])
        trainable_param_ix = [
            i for i, lbl in enumerate(self.model.parameter_labels)
            if lbl not in fixed_pars]
        param_mask_np = np.zeros((n_vx, n_pars), dtype=np.float32)
        for i in np.where(meaningful_ts)[0]:
            for j in trainable_param_ix:
                param_mask_np[i, j] = 1.0
        param_mask = ops.convert_to_tensor(param_mask_np)

        init_uncon_t = ops.convert_to_tensor(init_uncon)
        params_var = keras.Variable(
            init_uncon.copy(), dtype='float32', name='params')

        # Per-vertex Gaussian noise variance, parameterized via softplus.
        # Initial sigma^2 from classical residuals.
        with_initial_pred = self.model.predict(
            self.paradigm, self.classical_estimates, self.model.weights)
        residual = self.data.values - with_initial_pred.values
        init_sigma2 = np.maximum(np.var(residual, axis=0).astype(np.float32),
                                 1e-4)
        log_sigma2_uncon = ops.convert_to_numpy(softplus_inverse(
            ops.convert_to_tensor(init_sigma2, dtype='float32')))
        sigma2_var = keras.Variable(
            log_sigma2_uncon.copy(), dtype='float32', name='log_sigma2')

        trainable_variables = [params_var, sigma2_var]

        # Indices of parameters that have a GP prior.
        prior_indices = {name: self.model.parameter_labels.index(name)
                         for name in self.priors.keys()}

        # Freeze the Cholesky factor of each prior so the gradient
        # graph does not traverse cholesky(K) backward. TF's
        # CholeskyGrad NaN's on mild ill-conditioning, which used to
        # crash fit_map after a varying number of Adam steps.
        for prior in self.priors.values():
            prior.freeze_cholesky()

        paradigm_ = self.model.stimulus._clean_paradigm(self.paradigm)

        def build_params_native():
            uncon = (init_uncon_t * (1.0 - param_mask)
                     + params_var * param_mask)
            return self.model._transform_parameters_forward(uncon)

        def loss_fn():
            native = build_params_native()
            preds = self.model._predict(
                paradigm_[None, ...], native[None, ...], None)[0]  # (n_t, n_vx)
            ssq = ops.sum((y - preds) ** 2, axis=0)                # (n_vx,)
            sigma2 = ops.softplus(sigma2_var)                      # (n_vx,)

            n_t_f = ops.cast(n_t, 'float32')
            # Per-vertex Gaussian nLL, summed over vertices.
            nll = 0.5 * (n_t_f * (_LOG_2PI + ops.log(sigma2))
                         + ssq / sigma2)
            nll_total = ops.sum(nll)

            nlp_total = 0.0
            for name, prior in self.priors.items():
                idx = prior_indices[name]
                nlp_total = nlp_total - prior.log_prob(native[:, idx])

            return nll_total + nlp_total

        # Note: we implement gradient clipping manually rather than via
        # keras.optimizers.Adam(clipnorm=...) because the latter's
        # behavior under the PyTorch backend interacts poorly with a
        # transient NaN gradient (it scales by 1/NaN, poisoning Adam's
        # moment buffers and producing NaN parameters that propagate
        # for the rest of the loop). The manual path below scales by a
        # NaN-safe factor and skips the update entirely if the loss or
        # any gradient is non-finite.
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
        history = []
        best = float('inf')
        best_params = ops.convert_to_numpy(params_var)
        best_sigma2 = ops.convert_to_numpy(sigma2_var)
        stall = 0

        pbar = range(max_n_iterations)
        if progressbar:
            pbar = tqdm(pbar, desc='MAP')

        for step in pbar:
            loss, grads = compute_gradients(loss_fn, trainable_variables)
            loss_val = float(ops.convert_to_numpy(loss))

            # Manual global-norm clip, NaN-safe. If the loss or any
            # gradient is non-finite, skip the update so a single bad
            # step cannot poison Adam's moment buffers.
            grads_np = [ops.convert_to_numpy(g) for g in grads]
            finite = (np.isfinite(loss_val)
                      and all(np.all(np.isfinite(g)) for g in grads_np))
            if finite:
                if clipnorm is not None:
                    global_norm = float(np.sqrt(
                        sum(float((g ** 2).sum()) for g in grads_np)))
                    if global_norm > clipnorm and global_norm > 0.0:
                        scale = clipnorm / global_norm
                        grads = [ops.convert_to_tensor(g * scale,
                                                       dtype='float32')
                                 for g in grads_np]
                opt.apply_gradients(zip(grads, trainable_variables))
            history.append(loss_val)

            if finite and loss_val < best - tol:
                best = loss_val
                best_params = ops.convert_to_numpy(params_var)
                best_sigma2 = ops.convert_to_numpy(sigma2_var)
                stall = 0
            else:
                stall += 1

            if progressbar:
                pbar.set_description(f'MAP nLP {loss_val:.2f}')

            if stall >= patience and step > 50:
                break

        # Restore best-found values and produce native-space estimates.
        params_var.assign(best_params)
        sigma2_var.assign(best_sigma2)
        native_best = ops.convert_to_numpy(build_params_native())
        sigma2_native = ops.convert_to_numpy(ops.softplus(sigma2_var))

        self.map_estimates = format_parameters(
            native_best, self.model.parameter_labels)
        self.map_estimates.index = self.data.columns
        if not self.map_estimates.index.name:
            self.map_estimates.index.name = 'source'

        self.map_sigma = pd.Series(np.sqrt(sigma2_native),
                                   index=self.data.columns, name='sigma')
        self.map_history = np.asarray(history)

        # Release the cached Cholesky so subsequent calls to
        # fit_hyperparameters (or log_prob after hyperparams change)
        # see the live K, not the stale frozen factor.
        for prior in self.priors.values():
            prior.unfreeze_cholesky()

        return self.map_estimates
