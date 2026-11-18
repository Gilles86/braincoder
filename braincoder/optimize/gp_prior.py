"""Gaussian-Process prior over a pRF parameter, indexed by cortical distance.

Following Daghlian et al. (2025), this places a multivariate-normal prior on
the per-vertex values of a single pRF parameter (zero mean by default, or a
constant mean; optionally on the log of the parameter), with covariance
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

from ..utils.backend import float64_device, to_float64, softplus_inverse, compute_gradients

_LOG_2PI = float(np.log(2.0 * np.pi))


class GeodesicGPPrior(object):
    """GP prior with an RBF kernel over a pairwise distance matrix.

    The prior is on ``z = g(theta)``, where ``theta`` is the per-vertex
    parameter value as the model uses it (the native, constrained value,
    e.g. the preferred numerosity of a ``LogGaussianPRF``) and ``g`` is
    ``transform``:

        z ~ MVN(m * 1, K)
        K(i, j) = variance * exp(-d(i,j)^2 / (2 * lengthscale^2))
                 + nugget * delta(i, j)

    Mean ``m``:

    * ``mean='zero'`` (default): m = 0. This is the original prior. Note that
      on a parameter whose values are far from 0 (a preferred numerosity of
      10-40, a log preferred numerosity of 2-4) it pulls every vertex towards
      0, and the kernel can only represent the offset through a large
      ``variance`` and a long ``lengthscale``.
    * ``mean='constant'``: m is a free scalar, estimated together with the
      kernel hyperparameters (type-II ML in ``fit_hyperparameters``, type-II
      MAP when ``BayesianParameterFitter.fit_map(joint_hyperparams=True)``).
      If ``mean_init`` is None it is initialised at the median of the first
      values the prior sees (``initialize_mean``).
    * ``mean=<float>``: m is fixed at that value (on the transformed scale),
      e.g. the median of the classical estimates.

    ``transform``: ``'identity'`` (default, the prior acts on the native
    value, the original behaviour) or ``'log'`` (the prior acts on
    ``log(theta)``; use it for positive parameters such as a preferred
    numerosity or a tuning width). A MAP fit with ``transform='log'`` finds
    the mode of ``log p(y | theta) + log p(log theta)``, i.e. the mode in the
    log coordinates (no Jacobian term).

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
                 jitter=1e-4,
                 psd_correction='mds',
                 mean='zero',
                 mean_init=None,
                 transform='identity'):
        """``psd_correction``: how to handle non-Euclidean input distances.

        * ``'mds'`` (default) — classical multi-dimensional scaling embeds
          the distance matrix into Euclidean space. RBF on the resulting
          Euclidean distances is provably PSD (Schoenberg 1938), so the
          Cholesky never NaNs even when the input is a graph distance
          (mesh-edge Dijkstra, geodesic, heat-method, …). Recommended.
        * ``'none'`` — use the raw distances as-is. Fine for genuinely
          Euclidean inputs; will fail Cholesky on graph distances.
        """
        d = np.asarray(distance_matrix, dtype=np.float32)
        if d.ndim != 2 or d.shape[0] != d.shape[1]:
            raise ValueError(
                f"distance_matrix must be square 2-D, got shape {d.shape}")

        if psd_correction == 'mds':
            d = mds_embed_distances(d)
        elif psd_correction != 'none':
            raise ValueError(
                f"psd_correction must be 'mds' or 'none', got {psd_correction!r}")

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

        # Frozen Cholesky factor used during MAP (set by freeze()).
        # Skipping the Cholesky op in the gradient graph avoids
        # TF's CholeskyGrad blowing up under mild ill-conditioning,
        # which is what triggers `Tensor had NaN values` in fit_map.
        self._cached_L = None
        self._cached_log_det = None

        if transform not in ('identity', 'log'):
            raise ValueError(
                f"transform must be 'identity' or 'log', got {transform!r}")
        self.transform = transform

        # Mean function.
        self._mean = None                 # keras.Variable (constant) or None
        self._fixed_mean = 0.0
        if isinstance(mean, str):
            if mean == 'zero':
                self.mean_type = 'zero'
            elif mean == 'constant':
                self.mean_type = 'constant'
                self._mean = keras.Variable(
                    np.float32(0.0 if mean_init is None else mean_init),
                    dtype='float32', name='gp_mean')
                self._mean_initialized = mean_init is not None
            else:
                raise ValueError(
                    "mean must be 'zero', 'constant' or a number, "
                    f"got {mean!r}")
        else:
            self.mean_type = 'fixed'
            self._fixed_mean = float(mean)

    # ------------------------------------------------------------------ API

    @property
    def mean(self):
        """Current mean on the transformed scale (0 for ``mean='zero'``)."""
        if self._mean is not None:
            return float(ops.convert_to_numpy(self._mean))
        return self._fixed_mean

    @property
    def needs_mean_init(self):
        return self.mean_type == 'constant' and not self._mean_initialized

    def transform_values(self, values):
        """Map native parameter values to the scale the prior acts on."""
        v = ops.convert_to_tensor(values, dtype='float32')
        if self.transform == 'log':
            return ops.log(v)
        return v

    def initialize_mean(self, values, force=False):
        """Set a constant mean to the median of ``transform(values)``.

        No-op unless ``mean='constant'`` and the mean has not been set
        (pass ``force=True`` to reset it).
        """
        if self.mean_type != 'constant':
            return
        if self._mean_initialized and not force:
            return
        z = np.asarray(ops.convert_to_numpy(
            self.transform_values(np.asarray(values, dtype=np.float32))))
        self._mean.assign(np.float32(np.median(z)))
        self._mean_initialized = True

    def centred(self, values):
        """``transform(values) - mean`` as a flat float32 tensor."""
        z = ops.reshape(self.transform_values(values), (-1,))
        if self._mean is not None:
            return z - self._mean
        if self._fixed_mean != 0.0:
            return z - self._fixed_mean
        return z

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
                    nugget=self.nugget,
                    mean=self.mean,
                    mean_type=self.mean_type,
                    transform=self.transform)

    @property
    def kernel_variables(self):
        return [self._log_lengthscale, self._log_variance, self._log_nugget]

    @property
    def trainable_variables(self):
        """Kernel hyperparameters, plus the mean if ``mean='constant'``."""
        out = list(self.kernel_variables)
        if self._mean is not None:
            out.append(self._mean)
        return out

    def covariance(self):
        """Return the current covariance matrix K (with nugget + jitter)."""
        return self._build_covariance(self._log_lengthscale,
                                      self._log_variance,
                                      self._log_nugget)

    def freeze_cholesky(self):
        """Precompute & cache the Cholesky factor at current hyperparams.

        After calling this, ``log_prob(values)`` uses the cached factor
        and ``stop_gradient``s it so the gradient w.r.t. ``values``
        flows through ``solve_triangular(L, v)`` only — not back
        through ``cholesky(K)``. This sidesteps TF's ``CholeskyGrad``
        producing NaN under mild ill-conditioning during MAP, which
        otherwise crashes ``fit_map`` after dozens of Adam steps.

        Call this once before starting the MAP loop with hyperparams
        held fixed. ``unfreeze_cholesky()`` clears the cache.
        """
        K = self._build_covariance(self._log_lengthscale,
                                   self._log_variance,
                                   self._log_nugget)
        with float64_device():
            L = ops.cholesky(to_float64(K))
            self._cached_L = ops.stop_gradient(L)
            self._cached_log_det = ops.stop_gradient(
                2.0 * ops.sum(ops.log(ops.diag(L))))
        return self._cached_L

    def unfreeze_cholesky(self):
        """Clear cached Cholesky factor so log_prob recomputes K."""
        self._cached_L = None
        self._cached_log_det = None

    def log_prob(self, values):
        """Log-probability of a length-n_vx vector of native parameter values.

        The vector is transformed and centred (``transform(values) - mean``)
        and evaluated under MVN(0, K). ``values`` may be a numpy array or
        any Keras-backed tensor. If ``freeze_cholesky()`` has been called,
        uses the cached factor instead of recomputing.
        """
        v = self.centred(values)
        if self._cached_L is not None:
            return self._log_prob_tensor_cached(v)
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
        """Estimate (lengthscale, variance, nugget[, mean]) by MLE.

        This is stage 2 of the paper's empirical-Bayes recipe: the
        per-vertex parameter estimates from the classical fit are held
        fixed and the GP hyperparameters are optimized to maximize the
        marginal likelihood log p(values | m, K(theta)). ``values`` are
        native parameter values; the prior's ``transform`` is applied
        here. A constant mean is initialised at their (transformed)
        median if it was not set, then estimated with the kernel.
        """
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        self.initialize_mean(values)
        v = ops.convert_to_tensor(values, dtype='float32')
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
                return -self._log_prob_tensor(self.centred(v),
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
                    f'v={self.variance:.3f} nug={self.nugget:.3f} '
                    f'm={self.mean:.3f}')

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
        # ``values`` are already transformed and centred (see ``centred``).
        # Build the covariance and run Cholesky / triangular solves in
        # float64 — RBF kernels on ~hundreds of vertices are inherently
        # ill-conditioned and float32 Cholesky bottoms out at NaN.
        # Standard practice in TFP / GPyTorch / GPflow.
        K = self._build_covariance(log_l, log_v, log_n)
        with float64_device():
            L = ops.cholesky(to_float64(K))
            v64 = to_float64(ops.reshape(values, (-1, 1)))
            y = ops.solve_triangular(L, v64, lower=True)
            mahal = ops.sum(y * y)
            log_det = 2.0 * ops.sum(ops.log(ops.diag(L)))
            n_float = ops.cast(self.n_vx, 'float64')
            lp = -0.5 * (mahal + log_det + n_float * _LOG_2PI)
            lp = ops.cast(lp, 'float32')
        return ops.convert_to_tensor(lp)

    def _log_prob_tensor_cached(self, values):
        # Uses self._cached_L (already stop_gradient'd).
        # Gradient w.r.t. `values` flows only through solve_triangular,
        # which is stable; Cholesky is not in the backward graph.
        with float64_device():
            v64 = to_float64(ops.reshape(values, (-1, 1)))
            y = ops.solve_triangular(self._cached_L, v64, lower=True)
            mahal = ops.sum(y * y)
            n_float = ops.cast(self.n_vx, 'float64')
            lp = -0.5 * (mahal + self._cached_log_det + n_float * _LOG_2PI)
            lp = ops.cast(lp, 'float32')
        return ops.convert_to_tensor(lp)


def _to_unconstrained(x):
    """Map a positive scalar to its softplus-inverse value as float32."""
    x_arr = np.asarray(x, dtype=np.float32).reshape(())
    return ops.convert_to_numpy(
        softplus_inverse(ops.convert_to_tensor(x_arr, dtype='float32')))


def mds_embed_distances(d, return_info=False):
    """Closest Euclidean approximation of a distance matrix, by classical MDS.

    Classical MDS works on the double-centered squared-distance matrix
    ``B = -0.5 * J D² J``. Its positive eigenvalues' coordinates give a
    Euclidean embedding; negative eigenvalues correspond to the
    non-Euclidean part of the input metric, which is dropped. The output
    is the matrix of Euclidean distances among the embedded coordinates.

    Consequences that callers rely on:

    * ``exp(-d²/2l²)`` and ``exp(-beta d)`` are positive semi-definite for
      every ``l, beta > 0`` on the output (Schoenberg 1938), which they need
      not be on the input: mesh-edge Dijkstra and other geodesic distances
      are generally not Euclidean.
    * Discarding negative eigenvalues can only lengthen distances
      (``output >= input`` up to rounding), so the output is not the input:
      report ``info['negative_eigenvalue_fraction']`` and compare the two.
    * Two points at distance 0 with identical distance rows (e.g. voxels
      that share a mesh vertex) stay at distance 0, so a kernel on the output
      is singular on that pair. It is PSD, not strictly PD, there.

    Memory is ``O(n²)``: the distances come from the Gram matrix
    ``C Cᵀ = V_+ diag(w_+) V_+ᵀ``, not from an ``n × n × r`` difference tensor.

    Parameters
    ----------
    d : array (n, n)
        Symmetric, finite, non-negative distances.
    return_info : bool
        Also return a dict with ``n_components``,
        ``negative_eigenvalue_fraction`` (Σ|negative eigenvalues| / Σ|all
        eigenvalues|; 0 for a Euclidean input).

    Returns
    -------
    d_euclidean : float32 array (n, n)
    info : dict, only if ``return_info``
    """
    d = np.asarray(d, dtype=np.float64)
    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError(f'distance matrix must be square 2-D, got shape {d.shape}')
    if not np.isfinite(d).all():
        raise ValueError(
            f'distance matrix has {int((~np.isfinite(d)).sum())} non-finite '
            'entries (unreachable vertices on a disconnected mesh or patch?); '
            'it cannot be embedded.')
    if (d < 0).any():
        raise ValueError('distance matrix has negative entries')
    n = d.shape[0]
    D2 = d ** 2
    # Double-centering without forming J = I - 11ᵀ/n
    row_mean = D2.mean(axis=1, keepdims=True)
    B = -0.5 * (D2 - row_mean - row_mean.T + D2.mean())
    # Symmetrize before eigendecomposition (numerical drift)
    B = 0.5 * (B + B.T)
    w, V = np.linalg.eigh(B)
    # Keep positive eigenvalues only: the negative ones are exactly the
    # non-Euclidean component we want to discard.
    pos = w > 1e-10 * max(np.abs(w).max(), np.finfo(np.float64).tiny)
    # Pairwise Euclidean distances from the Gram matrix of the embedding.
    G = (V[:, pos] * w[pos]) @ V[:, pos].T
    g = np.diag(G)
    d_euc = np.sqrt(np.clip(g[:, None] + g[None, :] - 2.0 * G, 0.0, None)).astype(np.float32)
    np.fill_diagonal(d_euc, 0.0)
    if return_info:
        total = np.abs(w).sum()
        neg = np.abs(w[w < 0]).sum()
        return d_euc, dict(n_components=int(pos.sum()),
                           negative_eigenvalue_fraction=float(neg / total) if total > 0 else 0.0)
    return d_euc


# Backwards-compatible private name.
_mds_embed_distances = mds_embed_distances
