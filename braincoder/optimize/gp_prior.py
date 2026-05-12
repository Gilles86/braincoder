"""Gaussian-Process prior over a pRF parameter, indexed by cortical distance.

Following Daghlian et al. (2025), this places a zero-mean multivariate-normal
prior on the per-vertex values of a single pRF parameter, with covariance
given by an RBF kernel over a precomputed pairwise distance matrix
(typically geodesic distance on the cortical surface). The hyperparameters
(lengthscale, variance, nugget) are estimated by maximum likelihood from
the classical (per-vertex) parameter estimates — stage 2 of the empirical
Bayes recipe in the paper. Stage 3 then uses this prior to regularize a
joint fit of the model parameters.
"""

import numpy as np
import keras
from keras import ops
from tqdm.auto import tqdm

from ..utils.backend import softplus_inverse, compute_gradients

_LOG_2PI = float(np.log(2.0 * np.pi))


class GeodesicGPPrior(object):
    """Zero-mean GP prior with an RBF kernel over a pairwise distance matrix.

    Covariance:
        K(i, j) = variance * exp(-d(i,j)^2 / (2 * lengthscale^2))
                 + nugget * delta(i, j)

    Parameters
    ----------
    distance_matrix : array (n_vx, n_vx)
        Symmetric, non-negative pairwise distances. Geodesic distance on
        the cortical surface is the typical choice.
    lengthscale_init, variance_init, nugget_init : float
        Initial values for the three RBF hyperparameters, in their
        natural (positive) space. Stored internally on a softplus scale
        so they remain positive during optimization.
    jitter : float
        Small constant added to the diagonal beyond ``nugget`` for
        Cholesky stability. Scales with the mean diagonal of K so it
        adapts to ``variance``.
    """

    def __init__(self, distance_matrix,
                 lengthscale_init=10.0,
                 variance_init=1.0,
                 nugget_init=0.1,
                 jitter=1e-4):
        d = np.asarray(distance_matrix, dtype=np.float32)
        if d.ndim != 2 or d.shape[0] != d.shape[1]:
            raise ValueError(
                f"distance_matrix must be square 2-D, got shape {d.shape}")

        self.n_vx = d.shape[0]
        self._distance_sq = ops.convert_to_tensor(d ** 2, dtype='float32')
        self.jitter = float(jitter)

        # Free variables in unconstrained (softplus-inverse) space.
        self._log_lengthscale = keras.Variable(
            _to_unconstrained(lengthscale_init), dtype='float32',
            name='gp_log_lengthscale')
        self._log_variance = keras.Variable(
            _to_unconstrained(variance_init), dtype='float32',
            name='gp_log_variance')
        self._log_nugget = keras.Variable(
            _to_unconstrained(nugget_init), dtype='float32',
            name='gp_log_nugget')

    # ------------------------------------------------------------------ API

    @property
    def lengthscale(self):
        return float(ops.convert_to_numpy(ops.softplus(self._log_lengthscale)))

    @property
    def variance(self):
        return float(ops.convert_to_numpy(ops.softplus(self._log_variance)))

    @property
    def nugget(self):
        return float(ops.convert_to_numpy(ops.softplus(self._log_nugget)))

    @property
    def hyperparameters(self):
        return dict(lengthscale=self.lengthscale,
                    variance=self.variance,
                    nugget=self.nugget)

    @property
    def trainable_variables(self):
        return [self._log_lengthscale, self._log_variance, self._log_nugget]

    def covariance(self):
        """Return the current covariance matrix K (with nugget + jitter)."""
        return self._build_covariance(self._log_lengthscale,
                                      self._log_variance,
                                      self._log_nugget)

    def log_prob(self, values):
        """Log-probability of a length-n_vx vector under the current K.

        ``values`` may be a numpy array or any Keras-backed tensor.
        """
        v = ops.convert_to_tensor(values, dtype='float32')
        v = ops.reshape(v, (-1,))
        return self._log_prob_tensor(v,
                                     self._log_lengthscale,
                                     self._log_variance,
                                     self._log_nugget)

    def fit_hyperparameters(self, values,
                            max_n_iterations=500,
                            learning_rate=0.05,
                            tol=1e-4,
                            patience=20,
                            progressbar=True):
        """Estimate (lengthscale, variance, nugget) by MLE from fixed values.

        This is stage 2 of the paper's empirical-Bayes recipe: the
        per-vertex parameter estimates from the classical fit are held
        fixed and the GP hyperparameters are optimized to maximize the
        marginal likelihood log p(values | K(theta)).
        """
        v = ops.convert_to_tensor(np.asarray(values, dtype=np.float32),
                                  dtype='float32')
        v = ops.reshape(v, (-1,))
        if int(ops.shape(v)[0]) != self.n_vx:
            raise ValueError(
                f"values has {int(ops.shape(v)[0])} entries, "
                f"distance matrix is {self.n_vx} x {self.n_vx}")

        opt = keras.optimizers.Adam(learning_rate=learning_rate)
        history = []
        best = float('inf')
        best_vars = [ops.convert_to_numpy(x) for x in self.trainable_variables]
        stall = 0

        pbar = range(max_n_iterations)
        if progressbar:
            pbar = tqdm(pbar, desc='GP hyperparams')

        for step in pbar:
            def loss_fn():
                return -self._log_prob_tensor(v,
                                              self._log_lengthscale,
                                              self._log_variance,
                                              self._log_nugget)

            loss, grads = compute_gradients(loss_fn, self.trainable_variables)
            opt.apply_gradients(zip(grads, self.trainable_variables))
            loss_val = float(ops.convert_to_numpy(loss))
            history.append(loss_val)

            if loss_val < best - tol:
                best = loss_val
                best_vars = [ops.convert_to_numpy(x)
                             for x in self.trainable_variables]
                stall = 0
            else:
                stall += 1

            if progressbar:
                pbar.set_description(
                    f'GP nLL {loss_val:.3f} | l={self.lengthscale:.2f} '
                    f'v={self.variance:.3f} nug={self.nugget:.3f}')

            if stall >= patience:
                break

        # Restore best-found hyperparameters.
        for var, value in zip(self.trainable_variables, best_vars):
            var.assign(value)

        return dict(history=np.asarray(history),
                    best_neg_log_lik=best,
                    hyperparameters=self.hyperparameters)

    # ------------------------------------------------------------- internals

    def _build_covariance(self, log_l, log_v, log_n):
        l = ops.softplus(log_l)
        v = ops.softplus(log_v)
        n = ops.softplus(log_n)
        K = v * ops.exp(-self._distance_sq / (2.0 * l * l))
        eye = ops.eye(self.n_vx, dtype='float32')
        # Adaptive jitter scales with the variance term so it stays
        # numerically relevant when v >> 1 (e.g. priors on a parameter
        # with a large native scale, like numerosity 'mu' in [10, 40]).
        K = K + (n + v * self.jitter + 1e-9) * eye
        return K

    def _log_prob_tensor(self, values, log_l, log_v, log_n):
        # Build the covariance and run Cholesky / triangular solves in
        # float64 — RBF kernels on ~hundreds of vertices are inherently
        # ill-conditioned and float32 Cholesky bottoms out at NaN.
        # Standard practice in TFP / GPyTorch / GPflow.
        K = ops.cast(
            self._build_covariance(log_l, log_v, log_n), 'float64')
        L = ops.cholesky(K)
        v64 = ops.cast(ops.reshape(values, (-1, 1)), 'float64')
        y = ops.solve_triangular(L, v64, lower=True)
        mahal = ops.sum(y * y)
        log_det = 2.0 * ops.sum(ops.log(ops.diag(L)))
        n_float = ops.cast(self.n_vx, 'float64')
        lp = -0.5 * (mahal + log_det + n_float * _LOG_2PI)
        return ops.cast(lp, 'float32')


def _to_unconstrained(x):
    """Map a positive scalar to its softplus-inverse value as float32."""
    x_arr = np.asarray(x, dtype=np.float32).reshape(())
    return ops.convert_to_numpy(
        softplus_inverse(ops.convert_to_tensor(x_arr, dtype='float32')))
