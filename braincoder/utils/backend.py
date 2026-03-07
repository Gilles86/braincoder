"""Backend-agnostic utilities for Keras 3 multi-backend support.

Provides pure ``keras.ops`` implementations of operations that were
previously handled by TensorFlow-Probability, plus a thin abstraction
over gradient computation so the rest of the codebase is not coupled
to ``tf.GradientTape``.
"""

import numpy as np
import keras
from keras import ops


# ---------------------------------------------------------------------------
# Inverse transforms
# ---------------------------------------------------------------------------

def softplus_inverse(x):
    """Numerically stable inverse of softplus: log(exp(x) - 1).

    For large ``x`` (> 20), softplus(x) ≈ x so the inverse is also ≈ x.
    """
    return ops.where(x > 20.0, x, ops.log(ops.exp(x) - 1.0 + 1e-7))


# ---------------------------------------------------------------------------
# Multivariate Normal log-probability (replaces tfd.MultivariateNormalTriL)
# ---------------------------------------------------------------------------

def mvn_log_prob(x, L):
    """Log-probability under MVN(0, L Lᵀ) for each row of ``x``.

    Parameters
    ----------
    x : tensor, shape (n_timepoints, n_voxels)
        Residuals (mean assumed zero).
    L : tensor, shape (n_voxels, n_voxels)
        Lower-triangular Cholesky factor of the covariance matrix.

    Returns
    -------
    log_probs : tensor, shape (n_timepoints,)
    """
    k = ops.cast(ops.shape(x)[1], 'float32')
    # Solve L y = xᵀ  →  y shape (n_voxels, n_timepoints)
    y = ops.solve_triangular(L, ops.transpose(x), lower=True)
    # log|Sigma| = 2 * sum(log(diag(L)))
    log_det = 2.0 * ops.sum(ops.log(ops.diag(L)))
    mahal = ops.sum(y ** 2, axis=0)          # (n_timepoints,)
    log_2pi = ops.cast(ops.log(ops.convert_to_tensor(2.0 * np.pi)), 'float32')
    return -0.5 * (k * log_2pi + log_det + mahal)


# ---------------------------------------------------------------------------
# Multivariate Student-T log-probability
#   (replaces tfd.MultivariateStudentTLinearOperator)
# ---------------------------------------------------------------------------

def _lgamma(x):
    """Log-gamma via Lanczos approximation (backend-agnostic scalar)."""
    # For scalar / small tensors this is fine; for large batches prefer
    # a backend-native lgamma if available.
    import math
    # Lanczos g=7 coefficients
    g = 7
    c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028,
         771.32342877765313, -176.61502916214059, 12.507343278686905,
         -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7]
    # Use scipy as backend-agnostic implementation, called on numpy scalars
    try:
        from scipy.special import gammaln
        return ops.convert_to_tensor(
            np.float32(gammaln(float(ops.convert_to_tensor(x).numpy()))),
            dtype='float32')
    except Exception:
        # Fallback: Sterling approximation
        x_val = float(ops.convert_to_tensor(x).numpy())
        return ops.convert_to_tensor(
            np.float32(0.5 * np.log(2 * np.pi / x_val) + x_val * np.log(x_val + 1.0 / (12.0 * x_val))),
            dtype='float32')


def mvt_log_prob(x, L, dof):
    """Log-probability under multivariate Student-T(dof, 0, L Lᵀ) for each row.

    Parameters
    ----------
    x   : tensor, shape (n_timepoints, n_voxels)
    L   : tensor, shape (n_voxels, n_voxels) — Cholesky factor
    dof : scalar tensor — degrees of freedom

    Returns
    -------
    log_probs : tensor, shape (n_timepoints,)
    """
    k = ops.cast(ops.shape(x)[1], 'float32')
    nu = ops.cast(dof, 'float32')

    y = ops.solve_triangular(L, ops.transpose(x), lower=True)   # (k, T)
    mahal = ops.sum(y ** 2, axis=0)                              # (T,)
    log_det = 2.0 * ops.sum(ops.log(ops.diag(L)))

    lg_nu_k = _lgamma((nu + k) / 2.0)
    lg_nu   = _lgamma(nu / 2.0)
    log_pi  = ops.cast(ops.log(ops.convert_to_tensor(np.pi)), 'float32')

    log_prob = (lg_nu_k - lg_nu
                - 0.5 * k * ops.log(nu)
                - 0.5 * k * log_pi
                - 0.5 * log_det
                - 0.5 * (nu + k) * ops.log(1.0 + mahal / nu))
    return log_prob


# ---------------------------------------------------------------------------
# Sampling (replaces tfd sampling)
# ---------------------------------------------------------------------------

def sample_mvn(L, shape, seed=None):
    """Draw samples from MVN(0, L Lᵀ).

    Parameters
    ----------
    L     : tensor, shape (n_voxels, n_voxels)
    shape : tuple — leading batch dimensions, e.g. (n_batches, n_timepoints)
    seed  : optional int

    Returns
    -------
    samples : tensor, shape (*shape, n_voxels)
    """
    n_voxels = ops.shape(L)[0]
    flat_shape = (int(np.prod(shape)), n_voxels)
    z = keras.random.normal(flat_shape, seed=seed)          # (N, k)
    samples = ops.matmul(z, ops.transpose(L))               # (N, k)
    return ops.reshape(samples, (*shape, n_voxels))


def sample_mvt(L, dof, shape, seed=None):
    """Draw samples from multivariate Student-T(dof, 0, L Lᵀ).

    Uses the representation: x = z / sqrt(v/dof) where z ~ MVN(0, LLᵀ)
    and v ~ chi2(dof).
    """
    n_voxels = ops.shape(L)[0]
    flat_n = int(np.prod(shape))
    z = sample_mvn(L, (flat_n,), seed=seed)                 # (N, k)
    # chi2(dof) = Gamma(dof/2, 2), sample via normal: v = sum of dof normals^2
    dof_int = max(1, int(round(float(ops.convert_to_tensor(dof).numpy()))))
    normals = keras.random.normal((flat_n, dof_int), seed=seed)
    v = ops.sum(normals ** 2, axis=1, keepdims=True)        # (N, 1)
    samples = z / ops.sqrt(v / ops.cast(dof, 'float32'))
    return ops.reshape(samples, (*shape, n_voxels))


def sample_student_t(dof, scale, shape, seed=None):
    """Draw i.i.d. samples from Student-T(dof, 0, scale)."""
    n = int(np.prod(shape))
    dof_int = max(1, int(round(float(ops.convert_to_tensor(dof).numpy()))))
    z = keras.random.normal((n,), seed=seed)
    v = ops.sum(keras.random.normal((n, dof_int), seed=seed) ** 2, axis=1)
    samples = z * scale / ops.sqrt(v / ops.cast(dof, 'float32'))
    return ops.reshape(samples, shape)


# ---------------------------------------------------------------------------
# Backend-specific differentiable math ops
# ---------------------------------------------------------------------------

def lgamma(x):
    """Backend-agnostic log-gamma function (differentiable)."""
    backend = keras.backend.backend()
    if backend == 'tensorflow':
        import tensorflow as tf
        return tf.math.lgamma(x)
    elif backend == 'jax':
        import jax.scipy.special as jss
        return jss.gammaln(x)
    elif backend == 'torch':
        import torch
        return torch.lgamma(x)
    else:
        from scipy.special import gammaln
        return ops.convert_to_tensor(gammaln(np.array(x)), dtype='float32')


def bessel_i0(x):
    """Backend-agnostic modified Bessel function I₀ (differentiable)."""
    backend = keras.backend.backend()
    if backend == 'tensorflow':
        import tensorflow as tf
        return tf.math.bessel_i0(x)
    elif backend == 'jax':
        import jax.scipy.special as jss
        return jss.i0(x)
    elif backend == 'torch':
        import torch
        return torch.special.i0(x)
    else:
        from scipy.special import i0
        return ops.convert_to_tensor(i0(np.array(x)), dtype='float32')


def interp_regular_1d_grid(x, x_ref_min, x_ref_max, y_ref):
    """Backend-agnostic linear interpolation on a regular 1-D grid.

    Drop-in replacement for ``tfp.math.interp_regular_1d_grid`` with
    ``fill_value='constant_extension'`` (clamp to boundary).

    Parameters
    ----------
    x         : tensor, shape (n_query,) or (n_query, 1)
    x_ref_min : scalar — lower end of reference grid
    x_ref_max : scalar — upper end of reference grid
    y_ref     : tensor, shape (n_grid, n_cols)

    Returns
    -------
    tensor, shape (n_query, n_cols)
    """
    n = ops.cast(ops.shape(y_ref)[0], 'float32')

    alpha = (ops.reshape(x, (-1,)) - x_ref_min) / (x_ref_max - x_ref_min) * (n - 1.0)
    alpha = ops.clip(alpha, 0.0, n - 1.0)           # constant extension

    idx_lo = ops.cast(ops.floor(alpha), 'int32')
    idx_hi = ops.minimum(idx_lo + 1, ops.cast(n, 'int32') - 1)
    frac   = alpha - ops.cast(idx_lo, 'float32')    # (n_query,)

    y_lo = ops.take(y_ref, idx_lo, axis=0)           # (n_query, n_cols)
    y_hi = ops.take(y_ref, idx_hi, axis=0)

    return y_lo * (1.0 - frac[:, None]) + y_hi * frac[:, None]


# ---------------------------------------------------------------------------
# Jacobian computation
# ---------------------------------------------------------------------------

def compute_jacobian(fn, inputs):
    """Compute the Jacobian of ``fn(inputs)`` w.r.t. ``inputs``.

    Returns a tensor of shape ``(*output_shape, *input_shape)``.
    """
    backend = keras.backend.backend()
    if backend == 'tensorflow':
        import tensorflow as tf
        inputs_var = tf.Variable(ops.convert_to_tensor(inputs))
        with tf.GradientTape() as tape:
            outputs = fn(inputs_var)
        return tape.jacobian(outputs, inputs_var)
    elif backend == 'jax':
        import jax
        return jax.jacobian(fn)(inputs)
    elif backend == 'torch':
        import torch
        return torch.autograd.functional.jacobian(fn, inputs)
    else:
        raise NotImplementedError(f"compute_jacobian not implemented for backend: {backend!r}")


# ---------------------------------------------------------------------------
# Gradient computation abstraction
# ---------------------------------------------------------------------------

def compute_gradients(loss_fn, variables):
    """Compute gradients of ``loss_fn()`` w.r.t. ``variables``.

    Returns
    -------
    loss   : scalar tensor
    grads  : list of gradient tensors aligned with ``variables``
    """
    backend = keras.backend.backend()
    if backend == 'tensorflow':
        import tensorflow as tf
        with tf.GradientTape() as tape:
            loss = loss_fn()
        grads = tape.gradient(loss, variables)
        return loss, grads
    elif backend == 'jax':
        import jax
        import jax.numpy as jnp
        var_arrays = [jnp.array(v.numpy()) for v in variables]

        def _loss(*arrays):
            for var, arr in zip(variables, arrays):
                var.assign(arr)
            return loss_fn()

        argnums = tuple(range(len(variables)))
        loss_val, grads = jax.value_and_grad(_loss, argnums=argnums)(*var_arrays)
        return ops.convert_to_tensor(loss_val), [ops.convert_to_tensor(g) for g in grads]
    elif backend == 'torch':
        import torch
        for v in variables:
            v.value.requires_grad_(True)
        loss = loss_fn()
        loss.backward()
        grads = [v.value.grad for v in variables]
        return loss, grads
    else:
        raise NotImplementedError(f"compute_gradients not implemented for backend: {backend!r}")
