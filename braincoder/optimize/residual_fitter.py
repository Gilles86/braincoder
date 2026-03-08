import numpy as np
import keras
from keras import ops
from tqdm.auto import tqdm
from ..utils import format_data, format_paradigm, logit
from ..utils.backend import softplus_inverse, mvn_log_prob, mvt_log_prob, compute_gradients, to_numpy


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
            spherical=False,
            progressbar=True):

        n_voxels = self.data.shape[1]

        if residuals is None:
            residuals = (self.data - self.model.predict(paradigm=self.paradigm)).values

        sample_cov = np.cov(residuals.T)

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
            print(f'WWT max: {float(to_numpy(ops.max(WWT)))}')
            if normalize_WWT:
                WWT = WWT / ops.mean(WWT)

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
                    mean_tau = float(to_numpy(ops.mean(tau)))
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
                    return self._get_omega(tau, rho, sigma2, WWT)

                def get_pbar_description(cost, best_cost, variables):
                    tau, rho, sigma2 = transform_variables(variables)
                    mean_tau = float(to_numpy(ops.mean(tau)))
                    s = (f'fit stat: {cost:0.4f} (best: {best_cost:0.4f}, '
                         f'rho: {float(to_numpy(rho)):0.3f}, sigma2: {float(to_numpy(sigma2)):0.3f}, '
                         f'mean tau: {mean_tau:0.4f}')
                    if len(variables) == 4:
                        dof = float(to_numpy(ops.softplus(variables[3])))
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
                return self._get_omega_distance(tau, rho, sigma2, WWT, alpha, beta, D)

            def get_pbar_description(cost, best_cost, variables):
                tau, rho, sigma2, alpha, beta = transform_variables(variables)
                mean_tau = float(to_numpy(ops.mean(tau)))
                s = (f'fit stat: {cost:0.4f} (best: {best_cost:0.4f}, '
                     f'rho: {float(to_numpy(rho)):0.3f}, sigma2: {float(to_numpy(sigma2)):0.3f}, '
                     f'mean tau: {mean_tau:0.4f}, alpha: {float(to_numpy(alpha)):0.3f}, '
                     f'beta: {float(to_numpy(ops.convert_to_tensor(beta))):0.3f}')
                if len(variables) == 6:
                    dof = float(to_numpy(ops.softplus(variables[5])))
                    s += f', dof: {dof:0.1f}'
                return s

        if method == 'gauss':
            def likelihood(omega):
                omega_chol = ops.cholesky(omega)
                return ops.sum(mvn_log_prob(residuals_tensor, omega_chol))

            fit_stat = likelihood

        elif method == 't':

            dof_ = keras.Variable(softplus_inverse(init_dof),
                                  name='dof_trans', dtype='float32')
            trainable_variables += [dof_]

            def likelihood(omega):
                omega_chol = ops.cholesky(omega)
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
            return [to_numpy(ops.convert_to_tensor(v)) for v in variables]

        def restore_values(variables, saved):
            for var, val in zip(variables, saved):
                var.assign(val)

        best_cost = np.inf
        best_omega = get_omega(trainable_variables)
        best_values = save_values(trainable_variables)

        for step in pbar:
            try:
                def loss_fn():
                    omega = get_omega(trainable_variables)
                    return -fit_stat(omega)

                cost_tensor, gradients = compute_gradients(loss_fn, trainable_variables)
                cost = float(to_numpy(ops.convert_to_tensor(cost_tensor)))

                opt.apply_gradients(zip(gradients, trainable_variables))
                self.costs[step] = cost

                if cost < best_cost:
                    best_omega = to_numpy(get_omega(trainable_variables))
                    best_cost = cost
                    best_values = save_values(trainable_variables)

            except Exception:
                learning_rate = 0.9 * learning_rate
                opt = keras.optimizers.Adam(learning_rate=learning_rate)
                restore_values(trainable_variables, best_values)
                self.costs[step] = np.inf
                cost = np.inf

            if progressbar:
                pbar.set_description(get_pbar_description(cost, best_cost, best_values))

            previous_cost = self.costs[np.max((step - lag, 0))]

            if (step > min_n_iterations) and (np.sign(previous_cost) == np.sign(cost)):
                if np.sign(cost) == -1:
                    if (cost / previous_cost) < 1 + rtol:
                        break
                else:
                    if (cost / previous_cost) > 1 - rtol:
                        break

        omega = best_omega

        fitted_parameters = [to_numpy(ops.convert_to_tensor(v)) for v in
                             transform_variables(best_values)]
        self.fitted_omega_parameters = dict(zip(['tau', 'rho', 'sigma2'], fitted_parameters[:3]))

        if D is not None:
            self.fitted_omega_parameters['alpha'] = fitted_parameters[3]
            self.fitted_omega_parameters['beta']  = fitted_parameters[4]

        if method == 't':
            dof = float(to_numpy(ops.softplus(best_values[-1])))
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

    def _get_omega_lambda(self, tau, rho, sigma2, WWT, lambd, sample_covariance, eps=1e-9):
        return ((1 - lambd) * (rho * ops.transpose(tau) @ tau +
                               (1 - rho) * ops.diag(tau[0, :]**2) +
                               sigma2 * WWT) +
                lambd * sample_covariance +
                ops.diag(ops.ones(tau.shape[1]) * eps))
