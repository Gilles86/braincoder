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
# Robust Cholesky
# ---------------------------------------------------------------------------

def safe_cholesky(M, jitter=1e-4, max_attempts=6):
    """Cholesky factor of ``M`` with adaptive diagonal jitter + retries.

    Fitted covariance matrices coming out of the residual fitter (and
    similar optimisation loops) can slip below the PSD boundary by a
    small amount due to numerical drift in the Adam updates (e.g. α
    going slightly negative, β shrinking to ~0, or ρ saturating). A
    plain ``ops.cholesky`` then returns NaN and crashes the caller.

    Strategy:
      1. Symmetrise ``M`` (kills tiny asymmetry from accumulated rounding).
      2. Add ``(jitter * mean(diag(M)) + 1e-9) * I`` and try to factorise.
      3. On failure, multiply jitter by 10× and retry, up to
         ``max_attempts`` times. The jitter scales with the matrix's own
         diagonal so the same code path works for residual covariances
         on the order of ~1 and on the order of ~1e4.

    Most calls succeed on the first attempt with the default jitter
    (~1e-4 relative). Subjects whose ResidualFitter pushed Ω deeper
    below PSD need 1e-3 or 1e-2; the retries handle that without
    inflating the default perturbation on healthy subjects.

    Parameters
    ----------
    M : tensor or array, shape (n, n)
        Square matrix that should be PSD but may be slightly off.
    jitter : float, optional
        Initial relative jitter (multiplier on ``mean(diag(M))``).
        Default ``1e-4``.
    max_attempts : int, optional
        How many jitter levels to try before giving up. Each retry uses
        10× the previous jitter. Default 6 — covers
        ``1e-4 → 10`` relative to mean(diag), which always succeeds
        unless the input contains NaN/Inf.

    Returns
    -------
    L : tensor, shape (n, n)
        Lower-triangular Cholesky factor of the jittered matrix.

    Raises
    ------
    RuntimeError
        If all attempts fail. The input is then not rescuable by
        diagonal jitter alone (NaN/Inf or many strongly negative
        eigenvalues) and the caller should drop the fold / subject
        rather than silently propagate NaN.

    Notes
    -----
    The output preserves the caller's numpy dtype (float32 in →
    float32 out, float64 in → float64 out). On backends that don't
    natively support float64 (JAX without ``jax_enable_x64``), the
    factorisation is computed in float32 internally and cast back at
    the end — so the dtype contract holds even though the precision
    of intermediate arithmetic is bounded by the backend.
    """
    # Remember caller's dtype so the contract "dtype in == dtype out"
    # holds even on backends that silently downcast (e.g. JAX default
    # mode coerces float64 → float32 inside ``ops.convert_to_tensor``).
    in_dtype = getattr(M, 'dtype', None)

    M_t = ops.convert_to_tensor(M)
    M_sym = 0.5 * (M_t + ops.transpose(M_t))
    n = ops.shape(M_sym)[0]
    diag_mean = ops.mean(ops.diag(M_sym))
    eye = ops.eye(n, dtype=M_sym.dtype)
    epsilon = ops.cast(1e-9, M_sym.dtype)

    current = float(jitter)
    last_exc = None
    L = None
    for _ in range(max_attempts):
        scale = ops.cast(current, M_sym.dtype) * diag_mean + epsilon
        try:
            L_try = ops.cholesky(M_sym + scale * eye)
        except Exception as exc:
            last_exc = exc
            current *= 10.0
            continue
        # JAX silently returns a NaN-filled matrix when the input is not
        # PSD (the Keras wrapper raises in eager mode but only on concrete
        # arrays; under tracing or with older Keras, NaN propagates).
        # Treat a NaN result the same as a raised exception: bump jitter
        # and retry. The bool() cast forces concrete evaluation; this
        # function is therefore not safe to call inside `jax.jit` —
        # callers that need a jittable version should pre-symmetrise and
        # jitter the matrix themselves.
        try:
            has_nan = bool(ops.convert_to_numpy(ops.any(ops.isnan(L_try))))
        except Exception:
            # If we can't materialise the check (e.g. inside a trace),
            # trust the cholesky result; the exception path above will
            # still catch the eager-mode failures.
            has_nan = False
        if has_nan:
            last_exc = RuntimeError(
                f"cholesky returned NaN (input not PSD at jitter={current:.1e})")
            current *= 10.0
            continue
        L = L_try
        break
    if L is None:
        raise RuntimeError(
            f"safe_cholesky: failed after {max_attempts} jitter levels "
            f"(final jitter ~{current / 10:.1e}). Input is not rescuable "
            f"by diagonal jitter — check for NaN/Inf or strongly negative "
            f"eigenvalues. Last error: {last_exc}")

    # Restore caller's dtype if it was a concrete numpy dtype (e.g.
    # float64 in, but JAX default mode computed in float32). We only
    # do this when the caller explicitly passed a numpy array; tensor
    # inputs keep the backend's native dtype.
    #
    # Under JAX default mode, ``ops.cast`` to float64 silently keeps
    # float32, so we round-trip through numpy to actually upcast. For
    # tensor backends (torch / tf) this is a no-op since the dtype
    # already matches.
    if in_dtype is not None and isinstance(in_dtype, np.dtype):
        if str(L.dtype) != str(np.dtype(in_dtype)):
            L = np.asarray(ops.convert_to_numpy(L)).astype(in_dtype)
    return L


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
    """Log-gamma (differentiable, backend-native).

    Delegates to the public ``lgamma`` (defined below) so that the
    autograd graph is preserved on every backend. The previous
    implementation pulled scalars out via ``float(ops.convert_to_numpy(...))``
    and called ``scipy.special.gammaln`` — that path:

      * Crashes on JAX under tracing (``ConcretizationTypeError`` whenever
        ``mvt_log_prob`` is called with ``method='t'`` and the dof is a
        ``value_and_grad`` tracer).
      * Silently breaks autograd on TF/torch, because scipy's gammaln has
        no autograd connection: ``∂nll/∂dof`` is missing the gammaln
        contribution and dof-fitting drifts to a biased optimum.

    Routing through ``lgamma`` (which uses ``tf.math.lgamma`` /
    ``jax.scipy.special.gammaln`` / ``torch.lgamma`` depending on the
    backend) fixes both. Kept as ``_lgamma`` for backwards compatibility
    of existing call sites.
    """
    return lgamma(ops.cast(x, 'float32'))


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


def _split_seed(seed, n):
    """Return ``n`` distinct seeds derived from a single integer ``seed``.

    Background: ``keras.random.normal(..., seed=int)`` on the JAX backend
    converts the integer into a ``jax.random.PRNGKey``, and that key is
    *deterministic in the integer*. Two calls with the same ``seed=int``
    therefore produce the **identical** sequence on JAX, while on
    TF/torch a global counter advances between calls and the streams
    differ. Calling ``sample_mvn(seed=42)`` followed by
    ``keras.random.normal(seed=42)`` in the Student-T construction
    therefore makes the numerator ``z`` and the χ²-scaler ``v``
    perfectly correlated on JAX, so the resulting "Student-T sample"
    collapses to ``sign(z) * sqrt(dof)`` — heavy-tailed structure gone.

    Fix: derive sub-seeds (``seed+1, seed+2, …``) so each downstream
    ``keras.random`` call has its own key. This is the simplest
    backend-agnostic split that preserves backwards compatibility for
    the ``seed=None`` case (fresh randomness from the backend's global
    counter).
    """
    if seed is None:
        return [None] * n
    return [int(seed) + i for i in range(n)]


def sample_mvt(L, dof, shape, seed=None):
    """Draw samples from multivariate Student-T(dof, 0, L Lᵀ).

    Uses the representation: x = z / sqrt(v/dof) where z ~ MVN(0, LLᵀ)
    and v ~ chi2(dof). The two random draws (z and v) MUST be
    independent; see ``_split_seed`` for the JAX PRNG subtlety this
    function had to defend against.
    """
    n_voxels = ops.shape(L)[0]
    flat_n = int(np.prod(shape))
    seed_z, seed_v = _split_seed(seed, 2)
    z = sample_mvn(L, (flat_n,), seed=seed_z)               # (N, k)
    # chi2(dof) = Gamma(dof/2, 2), sample via normal: v = sum of dof normals^2
    dof_int = max(1, int(round(float(ops.convert_to_numpy(ops.convert_to_tensor(dof))))))
    normals = keras.random.normal((flat_n, dof_int), seed=seed_v)
    v = ops.sum(normals ** 2, axis=1, keepdims=True)        # (N, 1)
    samples = z / ops.sqrt(v / ops.cast(dof, 'float32'))
    return ops.reshape(samples, (*shape, n_voxels))


def sample_student_t(dof, scale, shape, seed=None):
    """Draw i.i.d. samples from Student-T(dof, 0, scale).

    Same independence requirement as ``sample_mvt`` — see ``_split_seed``.
    """
    n = int(np.prod(shape))
    dof_int = max(1, int(round(float(ops.convert_to_numpy(ops.convert_to_tensor(dof))))))
    seed_z, seed_v = _split_seed(seed, 2)
    z = keras.random.normal((n,), seed=seed_z)
    v = ops.sum(keras.random.normal((n, dof_int), seed=seed_v) ** 2, axis=1)
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
        return torch.lgamma(ops.convert_to_tensor(x, dtype='float32'))
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
        return torch.special.i0(ops.convert_to_tensor(x, dtype='float32'))
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
        # Snapshot each Variable's current value as a concrete JAX array
        # so the trace operates on pure inputs, not on the live Variable.
        var_arrays = [ops.convert_to_tensor(v.value) for v in variables]

        # CRITICAL: do NOT `var.assign(<tracer>)` inside the closure.
        # `assign` writes the tracer into the Variable's backing array,
        # and the reference escapes the trace boundary — at the end of
        # ``jax.value_and_grad`` the Variable holds a ``LinearizeTracer``
        # instead of a concrete ``ArrayImpl``. Subsequent reads then
        # leak the entire forward graph one iteration at a time
        # (analogous to the torch ``v.grad``-accumulation bug fixed in
        # e5671a1). The user-visible symptom is monotonic host-memory
        # growth across optimizer iterations and an eventual OOM on long
        # runs.
        #
        # Fix: rebind the Variables *inside* a ``StatelessScope``, which
        # is Keras 3's official mechanism for purifying Variable reads
        # during a JAX trace. The scope binds (variable → traced array)
        # only for the body of the closure; outside, the Variables keep
        # their concrete values untouched. No tracer ever escapes.
        from keras.src.backend.common.stateless_scope import StatelessScope

        def _loss(*arrays):
            mapping = list(zip(variables, arrays))
            with StatelessScope(state_mapping=mapping):
                return loss_fn()

        argnums = tuple(range(len(variables)))
        loss_val, grads = jax.value_and_grad(_loss, argnums=argnums)(*var_arrays)
        return ops.convert_to_tensor(loss_val), [ops.convert_to_tensor(g) for g in grads]
    elif backend == 'torch':
        import torch
        # Use the functional `torch.autograd.grad` rather than
        # `.backward()` + read `.grad`. The latter accumulates into the
        # variables' `.grad` attribute across optimizer iterations and
        # returns live graph references — both leak the autograd graph
        # one iteration at a time, OOM'ing an 80 GiB A100 by ~iter 7
        # of the PRF gradient-descent loop. autograd.grad is the
        # idiomatic one-shot API: no `.grad` mutation, returns
        # standalone tensors that can be detached cleanly.
        for v in variables:
            v.value.requires_grad_(True)
        loss = loss_fn()
        inputs = [v.value for v in variables]
        grads = torch.autograd.grad(loss, inputs)
        return loss.detach(), [g.detach() for g in grads]
    else:
        raise NotImplementedError(f"compute_gradients not implemented for backend: {backend!r}")
