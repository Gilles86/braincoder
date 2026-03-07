import numpy as np
import tensorflow as tf
import keras
from keras import ops
from tqdm.auto import tqdm
from ..utils import format_data, format_paradigm, logit
from ..utils.backend import softplus_inverse, mvn_log_prob, mvt_log_prob


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

        # Compute sample covariance using numpy
        sample_cov = np.cov(residuals.T)

        if init_tau is None:
            init_tau = residuals.std(0)[np.newaxis, :]

        print(f'init_tau: {init_tau.min()}, {init_tau.max()}')

        tau_ = tf.Variable(initial_value=softplus_inverse(init_tau), shape=(
            1, n_voxels), name='tau_trans', dtype=tf.float32)

        if not spherical:
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

            WWT = ops.clip(WWT, -1e10, 1e10)
            print(f'WWT max: {np.max(WWT)}')
            if normalize_WWT:
                WWT /= np.mean(WWT)

            trainable_variables = [tau_, rho_, sigma2_]
        else:
            trainable_variables = [tau_]

        residuals_tensor = ops.convert_to_tensor(residuals, dtype='float32')

        if D is None:

            if spherical:

                transform_variables = lambda x: ops.softplus(x[0])

                @tf.function
                def get_omega(trainable_variables):
                    tau = transform_variables(trainable_variables)
                    return ops.diag(ops.squeeze(tau**2))

                def get_pbar_description(cost, best_cost, trainable_variables):
                    tau = transform_variables(trainable_variables)
                    mean_tau = ops.mean(tau).numpy()
                    return f'fit stat: {cost.numpy():0.4f} (best: {best_cost:0.4f},  mean tau: {mean_tau:0.4f}'

            else:

                @tf.function
                def transform_variables(trainable_variables):
                    tau_, rho_, sigma2_ = trainable_variables[:3]
                    tau = ops.softplus(tau_)
                    rho = ops.sigmoid(rho_)
                    sigma2 = ops.softplus(sigma2_)
                    return tau, rho, sigma2

                @tf.function
                def get_omega(trainable_variables):
                    tau, rho, sigma2 = transform_variables(trainable_variables)
                    omega = self._get_omega(tau, rho, sigma2, WWT)
                    return omega

                def get_pbar_description(cost, best_cost, trainable_variables):
                    tau, rho, sigma2 = transform_variables(trainable_variables)
                    mean_tau = ops.mean(tau).numpy()
                    print_str = f'fit stat: {cost.numpy():0.4f} (best: {best_cost:0.4f}, rho: {rho.numpy():0.3f}, sigma2: {sigma2.numpy():0.3f}, mean tau: {mean_tau:0.4f}'
                    if len(trainable_variables) == 4:
                        dof = ops.softplus(trainable_variables[3]).numpy()
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
                tau = ops.softplus(tau_)
                rho = ops.sigmoid(rho_)
                sigma2 = ops.softplus(sigma2_)
                alpha = ops.sigmoid(alpha_)
                return tau, rho, sigma2, alpha, beta

            @tf.function
            def get_omega(trainable_variables):
                tau, rho, sigma2, alpha, beta = transform_variables(trainable_variables)
                omega = self._get_omega_distance(tau, rho, sigma2, WWT, alpha, beta, D)
                return omega

            def get_pbar_description(cost, best_cost, trainable_variables):
                tau, rho, sigma2, alpha, beta = transform_variables(trainable_variables)
                mean_tau = ops.mean(tau).numpy()
                print_str = f'fit stat: {cost.numpy():0.4f} (best: {best_cost:0.4f}, rho: {rho.numpy():0.3f}, sigma2: {sigma2.numpy():0.3f}, mean tau: {mean_tau:0.4f}, alpha: {alpha.numpy():0.3f}, beta: {beta.numpy():0.3f}'
                if len(trainable_variables) == 6:
                    dof = ops.softplus(trainable_variables[5]).numpy()
                    print_str += f', dof: {dof:0.1f}'
                return print_str

        if method == 'gauss':
            @tf.function
            def likelihood(omega):
                omega_chol = ops.cholesky(omega)
                return ops.sum(mvn_log_prob(residuals_tensor, omega_chol))

            fit_stat = likelihood

        elif method == 't':

            dof_ = tf.Variable(initial_value=softplus_inverse(
                init_dof), name='tau_trans', dtype=tf.float32)

            trainable_variables += [dof_]

            @tf.function
            def likelihood(omega):
                omega_chol = ops.cholesky(omega)
                dof = ops.softplus(trainable_variables[-1])
                return ops.sum(mvt_log_prob(residuals_tensor, omega_chol, dof))

            fit_stat = likelihood

        elif method == 'ssq_cov':
            raise NotImplementedError()

        elif method == 'slogsq_cov':
            raise NotImplementedError()

        opt = keras.optimizers.Adam(learning_rate=learning_rate)

        pbar = range(max_n_iterations)

        if progressbar:
            pbar = tqdm(pbar)

        self.costs = np.zeros(max_n_iterations)

        def copy_variables(trainable_variables):
            return [tf.identity(e) for e in trainable_variables]

        best_cost = np.inf
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
                    opt = keras.optimizers.Adam(learning_rate=learning_rate)
                    trainable_variables = copy_variables(best_variables)
                    self.costs[step] = np.inf
                    cost = ops.convert_to_tensor(np.inf)

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
            dof = ops.softplus(best_variables[-1]).numpy()
            self.fitted_omega_parameters['dof'] = dof
            return omega, dof
        else:
            return omega, None

    @tf.function
    def _get_omega(self, tau, rho, sigma2, WWT):
        return rho * ops.transpose(tau) @ tau + \
            (1 - rho) * ops.diag(tau[0, :]**2) + \
            sigma2 * WWT

    @tf.function
    def _get_omega_distance(self, tau, rho, sigma2, WWT, alpha, beta, D):

        tautau = ops.transpose(tau) @ tau
        return rho * (alpha * (ops.exp(-beta * D) * tautau) + (1-alpha) * tautau) + \
            (1-rho) * ops.diag(tau[0, :]**2) + \
            sigma2 * WWT

    @tf.function
    def _get_omega_lambda(self, tau, rho, sigma2, WWT, lambd, sample_covariance, eps=1e-9):
        return (1-lambd) * (rho * ops.transpose(tau) @ tau +
                            (1 - rho) * ops.diag(tau[0, :]**2) +
                            sigma2 * WWT) + \
            lambd * sample_covariance +  \
            ops.diag(ops.ones(tau.shape[1]) * eps)
