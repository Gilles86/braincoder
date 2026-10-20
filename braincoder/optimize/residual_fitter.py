import warnings

import numpy as np
import keras
from keras import ops
from tqdm.auto import tqdm
from ..utils import format_data, format_paradigm, logit
from ..utils.backend import softplus_inverse, mvn_log_prob, mvt_log_prob, compute_gradients, safe_cholesky

# Exceptions that indicate a bug in the likelihood / model code rather
# than numerical divergence. Re-raised immediately by ResidualFitter.fit:
# lowering the learning rate cannot fix them. TypeError covers JAX's
# TracerArrayConversionError and ConcretizationTypeError (both subclass
# jax.errors.JAXTypeError -> TypeError), i.e. host conversions such as
# ``float(ops.convert_to_numpy(x))`` inside traced code.
_PROGRAMMING_ERRORS = (TypeError, AttributeError, NameError, ImportError,
                       NotImplementedError, KeyError, IndexError)


class ResidualFitter(object):
    """Fit a parametric noise covariance Ω (and Student-t ν) to residuals.

    Ω = ρ ττᵀ + (1 − ρ) diag(τ²) + σ² WWᵀ (optionally mixed with the sample
    covariance via ``lambd``).

    .. warning::
       Ω has a single shared factor. On real single-trial fMRI betas the
       residuals typically need tens of factors, and this Ω underestimates
       held-out noise (held-out Mahalanobis distances ~4× too large on 7T
       data), giving overconfident decoding posteriors. ``lambd > 0`` is
       unsafe when n_voxels > n_trials (rank-deficient sample covariance).
       The pseudo-WWᵀ from ``init_pseudoWWT`` is uncentred. For calibrated
       decoding see :class:`braincoder.noise.LowRankNoise` and the
       "Noise models for decoding" docs page.
    """

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
            use_wwt=True,
            learning_rate=0.02, rtol=1e-6, lag=100,
            init_alpha=0.99,
            init_beta=0.0,
            spherical=False,
            progressbar=True,
            max_consecutive_failures=50):
        """``use_wwt`` (default True): include the ``σ² · WᵀW`` term in
        Ω. When False, Ω is built without the tuning-similarity
        contribution — useful for testing whether ``WᵀW`` is doing
        real spatial-noise modeling or just creating
        signal-correlation structure that interacts with regularized
        encoding fits. ``σ²`` stays in the trainable set but has zero
        effect on Ω (gradient → 0; Adam leaves it inert).

        Failure handling: a step whose cost or gradients are non-finite,
        or which raises a numerical error (e.g. a Cholesky failure), is
        rolled back to the best values so far and the learning rate is
        reduced by 10%. ``max_consecutive_failures`` (default 50) such
        steps in a row raise ``RuntimeError``. Programming errors
        (``TypeError`` -- including JAX tracer-conversion errors --
        ``AttributeError``, ``NameError``, ...) are re-raised
        immediately. If no step ever produces a finite cost the fit
        raises instead of returning an unfitted Ω. The number of
        rolled-back steps is stored in ``self.n_failed_steps``.
        """

        n_voxels = self.data.shape[1]

        if residuals is None:
            residuals = (self.data - self.model.predict(paradigm=self.paradigm)).values

        sample_cov = np.cov(residuals.T)

        # ── λ=1 short-circuit ─────────────────────────────────────────────────
        # When lambd≥1 the parametric component vanishes entirely (gradient=0),
        # so running the full optimisation is wasteful and dof can drift to ∞.
        # Instead: use the sample covariance directly (with adaptive jitter) and
        # fit only dof via a short dedicated loop.
        if self.lambd >= 1.0:
            mean_var = float(np.mean(np.diag(sample_cov)))
            omega = sample_cov + np.eye(n_voxels) * (mean_var * 1e-4 + 1e-9)
            if method == 't':
                omega_t = ops.convert_to_tensor(omega, dtype='float32')
                omega_chol = ops.cholesky(omega_t)
                residuals_t = ops.convert_to_tensor(residuals, dtype='float32')
                dof_ = keras.Variable(softplus_inverse(init_dof), name='dof_trans', dtype='float32')
                opt = keras.optimizers.Adam(learning_rate=learning_rate)
                for _ in range(max_n_iterations):
                    def _dof_loss():
                        return -ops.sum(mvt_log_prob(residuals_t, omega_chol,
                                                     ops.softplus(dof_)))
                    _, grads = compute_gradients(_dof_loss, [dof_])
                    opt.apply_gradients(zip(grads, [dof_]))
                dof = float(ops.convert_to_numpy(ops.softplus(dof_)))
                return omega, dof
            return omega, None

        if init_tau is None:
            init_tau = residuals.std(0)[np.newaxis, :]

        print(f'init_tau: {init_tau.min()}, {init_tau.max()}')

        tau_ = keras.Variable(softplus_inverse(init_tau),
                              name='tau_trans', dtype='float32')

        if not spherical:
            rho_    = keras.Variable(logit(init_rho),
                                     name='rho_trans', dtype='float32')
            sigma2_ = keras.Variable(softplus_inverse(init_sigma2),
                                     name='sigma2_trans', dtype='float32')

            if use_wwt:
                if (not hasattr(self.model, 'weights')) or (self.model.weights is None):
                    print('USING A PSEUDO-WWT!')
                    WWT = self.model.get_pseudoWWT()
                else:
                    WWT = self.model.get_WWT()

                import pandas as pd
                if isinstance(WWT, (pd.DataFrame, pd.Series)):
                    WWT = WWT.values

                WWT = ops.convert_to_tensor(WWT, dtype='float32')
                WWT = ops.clip(WWT, -1e10, 1e10)
                print(f'WWT max: {float(ops.convert_to_numpy(ops.max(WWT)))}')
                if normalize_WWT:
                    WWT = WWT / ops.mean(WWT)
            else:
                # Strip the σ²·WᵀW term from Ω. sigma2_ stays trainable
                # but multiplies a zero matrix, so it contributes
                # nothing and Adam leaves it inert.
                print('USE_WWT=False: omitting σ²·WᵀW term from Ω')
                WWT = ops.zeros((n_voxels, n_voxels), dtype='float32')

            trainable_variables = [tau_, rho_, sigma2_]
        else:
            trainable_variables = [tau_]

        residuals_tensor = ops.convert_to_tensor(residuals, dtype='float32')

        if D is None:

            if spherical:

                def transform_variables(variables):
                    return ops.softplus(variables[0])

                def get_omega(variables):
                    tau = transform_variables(variables)
                    return ops.diag(ops.squeeze(tau**2))

                def get_pbar_description(cost, best_cost, variables):
                    tau = transform_variables(variables)
                    mean_tau = float(ops.convert_to_numpy(ops.mean(tau)))
                    return f'fit stat: {cost:0.4f} (best: {best_cost:0.4f},  mean tau: {mean_tau:0.4f}'

            else:

                def transform_variables(variables):
                    tau_   = variables[0]
                    rho_   = variables[1]
                    sigma2_= variables[2]
                    tau    = ops.softplus(tau_)
                    rho    = ops.sigmoid(rho_)
                    sigma2 = ops.softplus(sigma2_)
                    return tau, rho, sigma2

                def get_omega(variables):
                    tau, rho, sigma2 = transform_variables(variables)
                    if self.lambd > 0.0:
                        return self._get_omega_lambda(tau, rho, sigma2, WWT,
                                                      self.lambd, sample_cov)
                    return self._get_omega(tau, rho, sigma2, WWT)

                def get_pbar_description(cost, best_cost, variables):
                    tau, rho, sigma2 = transform_variables(variables)
                    mean_tau = float(ops.convert_to_numpy(ops.mean(tau)))
                    s = (f'fit stat: {cost:0.4f} (best: {best_cost:0.4f}, '
                         f'rho: {float(ops.convert_to_numpy(rho)):0.3f}, sigma2: {float(ops.convert_to_numpy(sigma2)):0.3f}, '
                         f'mean tau: {mean_tau:0.4f}')
                    if len(variables) == 4:
                        dof = float(ops.convert_to_numpy(ops.softplus(variables[3])))
                        s += f', dof: {dof:0.1f}'
                    return s

        else:

            alpha_ = keras.Variable(logit(init_alpha),
                                    name='alpha_trans', dtype='float32')
            beta_  = keras.Variable(init_beta,
                                    name='beta', dtype='float32')

            trainable_variables += [alpha_, beta_]

            def transform_variables(variables):
                tau_, rho_, sigma2_, alpha_, beta_ = variables[:5]
                tau    = ops.softplus(tau_)
                rho    = ops.sigmoid(rho_)
                sigma2 = ops.softplus(sigma2_)
                alpha  = ops.sigmoid(alpha_)
                return tau, rho, sigma2, alpha, beta_

            def get_omega(variables):
                tau, rho, sigma2, alpha, beta = transform_variables(variables)
                if self.lambd > 0.0:
                    return self._get_omega_lambda(tau, rho, sigma2, WWT,
                                                  self.lambd, sample_cov)
                return self._get_omega_distance(tau, rho, sigma2, WWT, alpha, beta, D)

            def get_pbar_description(cost, best_cost, variables):
                tau, rho, sigma2, alpha, beta = transform_variables(variables)
                mean_tau = float(ops.convert_to_numpy(ops.mean(tau)))
                s = (f'fit stat: {cost:0.4f} (best: {best_cost:0.4f}, '
                     f'rho: {float(ops.convert_to_numpy(rho)):0.3f}, sigma2: {float(ops.convert_to_numpy(sigma2)):0.3f}, '
                     f'mean tau: {mean_tau:0.4f}, alpha: {float(ops.convert_to_numpy(alpha)):0.3f}, '
                     f'beta: {float(ops.convert_to_numpy(ops.convert_to_tensor(beta))):0.3f}')
                if len(variables) == 6:
                    dof = float(ops.convert_to_numpy(ops.softplus(variables[5])))
                    s += f', dof: {dof:0.1f}'
                return s

        if method == 'gauss':
            def likelihood(omega):
                # Adaptive jitter via safe_cholesky: Adam can push α/β/ρ to
                # numerical edges where the assembled Ω dips below PSD.
                omega_chol = safe_cholesky(omega)
                return ops.sum(mvn_log_prob(residuals_tensor, omega_chol))

            fit_stat = likelihood

        elif method == 't':

            dof_ = keras.Variable(softplus_inverse(init_dof),
                                  name='dof_trans', dtype='float32')
            trainable_variables += [dof_]

            def likelihood(omega):
                omega_chol = safe_cholesky(omega)
                dof = ops.softplus(trainable_variables[-1])
                return ops.sum(mvt_log_prob(residuals_tensor, omega_chol, dof))

            fit_stat = likelihood

        elif method in ('ssq_cov', 'slogsq_cov'):
            raise NotImplementedError()

        opt = keras.optimizers.Adam(learning_rate=learning_rate)

        pbar = range(max_n_iterations)
        if progressbar:
            pbar = tqdm(pbar)

        self.costs = np.zeros(max_n_iterations)

        def save_values(variables):
            return [ops.convert_to_numpy(ops.convert_to_tensor(v)) for v in variables]

        def restore_values(variables, saved):
            for var, val in zip(variables, saved):
                var.assign(val)

        best_cost = np.inf
        best_omega = None
        best_values = save_values(trainable_variables)

        # Bookkeeping for failed steps. A step "fails" when the cost or
        # gradients are non-finite, or when a *numerical* exception is
        # raised (e.g. safe_cholesky giving up, a backend Cholesky
        # error). Those are treated as divergence: roll back to the best
        # values, shrink the learning rate by 10%, retry. Programming
        # errors (TypeError -- which includes JAX's
        # TracerArrayConversionError / ConcretizationTypeError --
        # AttributeError, NameError, ...) are re-raised immediately:
        # backing off the learning rate cannot fix them, and swallowing
        # them used to make a fit "complete" with Ω frozen at its
        # initial value.
        self.n_failed_steps = 0
        consecutive_failures = 0
        seen_failure_types = set()

        def _register_failure(reason, exc=None):
            nonlocal learning_rate, opt, consecutive_failures
            self.n_failed_steps += 1
            consecutive_failures += 1
            key = type(exc).__name__ if exc is not None else reason
            if key not in seen_failure_types:
                seen_failure_types.add(key)
                msg = (f'ResidualFitter step {step}: {reason}; restoring best '
                       f'values and reducing learning rate to '
                       f'{0.9 * learning_rate:.3g}.')
                if exc is not None:
                    msg += f' Exception: {type(exc).__name__}: {exc}'
                warnings.warn(msg, RuntimeWarning, stacklevel=3)
            learning_rate = 0.9 * learning_rate
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
            restore_values(trainable_variables, best_values)
            if consecutive_failures >= max_consecutive_failures:
                raise RuntimeError(
                    f'ResidualFitter: {consecutive_failures} consecutive '
                    f'failed optimisation steps (last: {reason}). '
                    f'Best cost so far: {best_cost}. Aborting rather than '
                    f'returning an unfitted Ω.') from exc

        for step in pbar:
            def loss_fn():
                omega = get_omega(trainable_variables)
                return -fit_stat(omega)

            try:
                cost_tensor, gradients = compute_gradients(loss_fn, trainable_variables)
                cost = float(ops.convert_to_numpy(ops.convert_to_tensor(cost_tensor)))
                grads_finite = all(
                    g is not None and
                    bool(np.all(np.isfinite(ops.convert_to_numpy(g))))
                    for g in gradients)
            except _PROGRAMMING_ERRORS:
                raise
            except Exception as exc:  # numerical failure -> back off
                self.costs[step] = np.inf
                cost = np.inf
                _register_failure('exception during cost/gradient evaluation', exc)
            else:
                if not np.isfinite(cost) or not grads_finite:
                    # Do NOT apply the update: NaN gradients would poison
                    # the variables for every later step.
                    reason = ('non-finite cost' if not np.isfinite(cost)
                              else 'non-finite gradients')
                    self.costs[step] = np.inf
                    cost = np.inf
                    _register_failure(reason)
                else:
                    consecutive_failures = 0
                    self.costs[step] = cost
                    if cost < best_cost:
                        # best values are the ones the cost was evaluated at
                        # (before this step's update is applied).
                        best_omega = ops.convert_to_numpy(get_omega(trainable_variables))
                        best_cost = cost
                        best_values = save_values(trainable_variables)
                    opt.apply_gradients(zip(gradients, trainable_variables))

            if progressbar:
                pbar.set_description(get_pbar_description(cost, best_cost, best_values))

            if not np.isfinite(cost):
                # A rolled-back step says nothing about convergence (and
                # inf / previous_cost would trigger the rtol break below).
                continue

            previous_cost = self.costs[np.max((step - lag, 0))]

            if (step > min_n_iterations) and (np.sign(previous_cost) == np.sign(cost)):
                if np.sign(cost) == -1:
                    if (cost / previous_cost) < 1 + rtol:
                        break
                else:
                    if (cost / previous_cost) > 1 - rtol:
                        break

        if not np.isfinite(best_cost):
            raise RuntimeError(
                'ResidualFitter: no optimisation step produced a finite cost '
                f'({self.n_failed_steps} failed steps). Ω was never fitted.')

        if self.n_failed_steps > 0:
            warnings.warn(
                f'ResidualFitter: {self.n_failed_steps} of {step + 1} steps '
                f'failed and were rolled back (final learning rate '
                f'{learning_rate:.3g}).', RuntimeWarning, stacklevel=2)

        omega = best_omega

        fitted_parameters = [ops.convert_to_numpy(ops.convert_to_tensor(v)) for v in
                             transform_variables(best_values)]
        self.fitted_omega_parameters = dict(zip(['tau', 'rho', 'sigma2'], fitted_parameters[:3]))

        if D is not None:
            self.fitted_omega_parameters['alpha'] = fitted_parameters[3]
            self.fitted_omega_parameters['beta']  = fitted_parameters[4]

        if method == 't':
            dof = float(ops.convert_to_numpy(ops.softplus(best_values[-1])))
            self.fitted_omega_parameters['dof'] = dof
            return omega, dof
        else:
            return omega, None

    def _get_omega(self, tau, rho, sigma2, WWT):
        return (rho * ops.transpose(tau) @ tau +
                (1 - rho) * ops.diag(tau[0, :]**2) +
                sigma2 * WWT)

    def _get_omega_distance(self, tau, rho, sigma2, WWT, alpha, beta, D):
        tautau = ops.transpose(tau) @ tau
        return (rho * (alpha * (ops.exp(-beta * D) * tautau) + (1 - alpha) * tautau) +
                (1 - rho) * ops.diag(tau[0, :]**2) +
                sigma2 * WWT)

    def _get_omega_lambda(self, tau, rho, sigma2, WWT, lambd, sample_covariance):
        """Interpolate between parametric omega (lambd=0) and sample covariance (lambd=1).

        Jitter is adaptive: 1e-4 × mean diagonal so Cholesky stays stable
        regardless of data scale.  This matters most when lambd→1 and the
        sample covariance dominates (its smallest eigenvalue can be very small
        when n_voxels is close to n_timepoints).
        """
        result = ((1 - lambd) * (rho * ops.transpose(tau) @ tau +
                                 (1 - rho) * ops.diag(tau[0, :]**2) +
                                 sigma2 * WWT) +
                  lambd * sample_covariance)
        # Adaptive jitter: scales with the matrix so it's never swamped by
        # the signal or invisible against large diagonal values.
        mean_diag = ops.mean(ops.diag(result))
        return result + ops.eye(tau.shape[1], dtype='float32') * (mean_diag * 1e-4 + 1e-9)
