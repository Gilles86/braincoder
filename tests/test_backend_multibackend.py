"""Multi-backend regression tests for ``braincoder.utils.backend``.

These tests pin down four bugs found in the 2026-05-21 audit and are
designed to run under each of the three Keras 3 backends (TF / JAX /
PyTorch). The CI matrix in ``.github/workflows/tests.yml`` sets
``KERAS_BACKEND`` per job, so the same code below is exercised three
times.

Bugs covered:

  1. ``safe_cholesky`` silently returning NaN on JAX. The Keras 3
     wrapper raises in eager mode but the bare ``jnp.linalg.cholesky``
     does not, and earlier Keras versions / traced contexts propagate
     NaN. Fix: explicit ``ops.any(ops.isnan(L))`` check after each
     attempt, treat NaN like an exception.

  2. ``_lgamma`` was non-differentiable and JAX-incompatible because
     it pulled values out via ``ops.convert_to_numpy`` and called
     ``scipy.special.gammaln``. Fix: delegate to the public
     ``lgamma`` function, which uses backend-native lgamma.

  3. ``sample_mvt`` / ``sample_student_t`` reused the same JAX PRNG
     key for the numerator and the chi-squared scaler, perfectly
     correlating the two random draws on JAX so "Student-T samples"
     collapsed to ``sign(z) * sqrt(dof)``. Fix: derive sub-seeds.

  4. ``compute_gradients`` JAX path called ``var.assign(<tracer>)``
     inside the closure passed to ``jax.value_and_grad``. The tracer
     escaped the trace boundary, leaving Variables holding
     ``LinearizeTracer`` objects between iterations and accumulating
     the autograd graph one iteration at a time. Fix: bind variables
     via ``StatelessScope`` inside the closure; no tracer ever lands
     in the Variable's backing store.
"""
import numpy as np
import pytest
import keras
from keras import ops


# ---------------------------------------------------------------------------
# Bug 1: safe_cholesky must never return NaN on non-PSD input.
# ---------------------------------------------------------------------------

class TestBug1SafeCholeskyJaxNaN:
    """Regression for safe_cholesky on JAX (and any backend that silently
    NaN-propagates a non-PSD Cholesky)."""

    @staticmethod
    def _make_non_psd(n=8, neg_eig=-1e-3, seed=0):
        """Symmetric matrix with one eigenvalue clearly < 0 (plain Cholesky
        returns NaN on this on every backend that doesn't raise)."""
        rng = np.random.default_rng(seed)
        A = rng.standard_normal((n, n))
        Q, _ = np.linalg.qr(A)
        eigs = rng.uniform(0.1, 2.0, size=n).astype(np.float64)
        eigs[0] = neg_eig
        M = (Q * eigs) @ Q.T
        return (0.5 * (M + M.T)).astype(np.float32)

    def test_non_psd_input_does_not_return_nan(self):
        """The whole point of safe_cholesky: never NaN. On JAX, the bare
        cholesky returns NaN silently -- the explicit NaN check in
        safe_cholesky must catch that and retry with bigger jitter."""
        from braincoder.utils.backend import safe_cholesky
        M = self._make_non_psd(n=10, neg_eig=-1e-3)
        L = safe_cholesky(M)
        L_np = np.asarray(ops.convert_to_numpy(L))
        assert np.all(np.isfinite(L_np)), (
            f"safe_cholesky returned non-finite output on backend "
            f"{keras.backend.backend()!r}: any-NaN={np.isnan(L_np).any()}, "
            f"any-Inf={np.isinf(L_np).any()}")

    def test_non_psd_input_reconstructs_close_to_M(self):
        """Output should be a valid Cholesky factor of a jittered M."""
        from braincoder.utils.backend import safe_cholesky
        M = self._make_non_psd(n=8, neg_eig=-1e-4)
        L = safe_cholesky(M)
        L_np = np.asarray(ops.convert_to_numpy(L))
        recon = L_np @ L_np.T
        # Reconstruction within the relative jitter scale (default 1e-4).
        # Mean diag of M ~ 1, so absolute tolerance ~ a few x 1e-3.
        assert np.allclose(recon, M, atol=5e-3), (
            f"L Lt not close to M (max abs diff "
            f"{np.max(np.abs(recon - M))})")

    def test_strongly_non_psd_input_still_succeeds(self):
        """A larger negative eigenvalue -- adaptive jitter should still
        find a working level."""
        from braincoder.utils.backend import safe_cholesky
        M = self._make_non_psd(n=6, neg_eig=-1e-2)
        L = safe_cholesky(M)
        L_np = np.asarray(ops.convert_to_numpy(L))
        assert np.all(np.isfinite(L_np))


# ---------------------------------------------------------------------------
# Bug 2: _lgamma must be differentiable on every backend.
# ---------------------------------------------------------------------------

class TestBug2LgammaDifferentiable:
    """The previous ``_lgamma`` used ``scipy.special.gammaln`` on numpy
    scalars and had no autograd connection. Any code that takes a
    gradient through ``mvt_log_prob`` w.r.t. ``dof`` therefore had a
    biased gradient (TF/torch) or crashed entirely
    (``ConcretizationTypeError`` on JAX). These tests pin down the fix:
    ``_lgamma`` now delegates to backend-native lgamma."""

    def test_lgamma_value_matches_scipy(self):
        """Function value should still agree with scipy.special.gammaln."""
        from scipy.special import gammaln
        from braincoder.utils.backend import _lgamma
        for x in (0.5, 1.5, 3.5, 10.0):
            got = float(ops.convert_to_numpy(_lgamma(ops.convert_to_tensor(x))))
            assert np.isclose(got, gammaln(x), atol=1e-4), (
                f"_lgamma({x}) = {got}, expected {gammaln(x)}")

    def test_lgamma_gradient_matches_digamma(self):
        """``d lgamma(x) / dx = digamma(x)`` -- the gradient must be
        present and correct, not zero/None. This is the key autograd
        regression test."""
        from scipy.special import digamma
        from braincoder.utils.backend import _lgamma, compute_gradients
        x_val = 3.5
        v = keras.Variable(np.asarray([x_val], dtype=np.float32))

        def loss_fn():
            return ops.sum(_lgamma(v.value))

        loss, grads = compute_gradients(loss_fn, [v])
        grad = float(ops.convert_to_numpy(grads[0])[0])
        assert np.isclose(grad, digamma(x_val), atol=1e-3), (
            f"d/dx lgamma({x_val}) = {grad}, expected digamma = "
            f"{digamma(x_val)} (backend={keras.backend.backend()!r})")

    def test_mvt_log_prob_dof_gradient_is_nonzero(self):
        """``mvt_log_prob`` depends on ``dof`` through ``_lgamma`` -- if
        the lgamma gradient is missing, the dof gradient picks up only
        the algebraic terms and the lgamma contribution is lost.
        This test fails under the broken ``_lgamma``."""
        from braincoder.utils.backend import mvt_log_prob, compute_gradients
        rng = np.random.default_rng(0)
        n_t, k = 50, 3
        x = rng.standard_normal((n_t, k)).astype(np.float32)
        L = np.eye(k, dtype=np.float32)
        x_t = ops.convert_to_tensor(x)
        L_t = ops.convert_to_tensor(L)

        dof_var = keras.Variable(np.asarray([5.0], dtype=np.float32))

        def loss_fn():
            return -ops.sum(mvt_log_prob(x_t, L_t, dof_var.value[0]))

        _, grads = compute_gradients(loss_fn, [dof_var])
        g = float(ops.convert_to_numpy(grads[0])[0])
        assert np.isfinite(g), f"dof gradient is non-finite: {g}"
        assert abs(g) > 1e-3, (
            f"dof gradient ~ 0 ({g}) -- looks like the lgamma terms have "
            f"no autograd connection (backend={keras.backend.backend()!r})")

