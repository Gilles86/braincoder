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

