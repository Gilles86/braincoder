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


# ---------------------------------------------------------------------------
# Bug 3: sample_mvt / sample_student_t must use independent random draws.
# ---------------------------------------------------------------------------

class TestBug3StudentTSampling:
    """Under the bug, ``sample_student_t(dof=5)`` collapsed to roughly
    ``sign(z) * sqrt(dof)`` because the same JAX PRNG key was used for
    the numerator and the chi-squared scaler. The kurtosis test below
    is the sharpest distributional fingerprint of that failure mode."""

    def test_student_t_variance_matches_dof(self):
        """For dof > 2, Student-T variance is ``dof / (dof - 2)``. For
        dof=5 that's 5/3 ~= 1.667. Under the bug, the variance is fixed
        at ``dof`` (5) because every sample is ``+/-sqrt(dof)``."""
        from braincoder.utils.backend import sample_student_t
        n = 20000
        dof = 5.0
        samples = sample_student_t(dof, 1.0, (n,), seed=42)
        arr = np.asarray(ops.convert_to_numpy(samples))
        v = float(arr.var())
        expected = dof / (dof - 2.0)  # 1.667
        # 2-sigma band for variance of t(5) with n=20000 is comfortably
        # within +/-0.25 of the mean. The bugged version returns var ~= 5.
        assert abs(v - expected) < 0.3, (
            f"Student-T(5) variance = {v}, expected ~{expected}. "
            f"This usually means the chi-squared scaler and z are correlated "
            f"(JAX PRNG-reuse bug). backend={keras.backend.backend()!r}")

    def test_student_t_samples_are_not_two_valued(self):
        """The bugged version returns approximately two distinct
        absolute values (sqrt(dof) +/- noise). Real Student-T has a
        continuous distribution -- the IQR of |x| should be at least
        0.5 (it's ~1.5 for true t(5); ~0.05 under the bug)."""
        from braincoder.utils.backend import sample_student_t
        n = 10000
        samples = sample_student_t(5.0, 1.0, (n,), seed=7)
        arr = np.asarray(ops.convert_to_numpy(samples))
        abs_x = np.abs(arr)
        iqr = np.subtract(*np.percentile(abs_x, [75, 25]))
        assert iqr > 0.5, (
            f"IQR(|x|) = {iqr:.3f} -- samples look quantised to two "
            f"values. Likely the chi-squared scaler is correlated with z.")

    def test_student_t_kurtosis_finite(self):
        """Sample kurtosis for t(5) is ~9 in theory. We only check that
        it's finite and noticeably > 3 (the Gaussian value). Under the
        bug the sample is essentially +/-sqrt(5), which gives kurtosis ~1
        (bimodal) -- well below 3."""
        from scipy.stats import kurtosis
        from braincoder.utils.backend import sample_student_t
        n = 20000
        samples = sample_student_t(5.0, 1.0, (n,), seed=42)
        arr = np.asarray(ops.convert_to_numpy(samples))
        k = kurtosis(arr, fisher=False)
        assert np.isfinite(k)
        # Anything > 4 is a clear sign of heavy tails (Gaussian = 3).
        # Under the bug we see k < 2.
        assert k > 4.0, (
            f"Sample kurtosis = {k}, expected heavy-tailed (>4). "
            f"backend={keras.backend.backend()!r}")

    def test_multivariate_student_t_variance(self):
        """Same check for ``sample_mvt`` -- the multivariate path."""
        from braincoder.utils.backend import sample_mvt
        n = 10000
        dof = 5.0
        L = ops.convert_to_tensor(np.eye(3, dtype=np.float32))
        samples = sample_mvt(L, dof, (n,), seed=123)
        arr = np.asarray(ops.convert_to_numpy(samples))
        v = arr.var(axis=0)
        expected = dof / (dof - 2.0)
        for i, vi in enumerate(v):
            assert abs(vi - expected) < 0.4, (
                f"MVT dim {i}: var = {vi}, expected ~{expected}")


# ---------------------------------------------------------------------------
# Bug 4: compute_gradients JAX path must not leak tracers / accumulate graphs.
# ---------------------------------------------------------------------------

class TestBug4ComputeGradientsJaxMemory:
    """The original JAX path called ``var.assign(<tracer>)`` inside the
    ``value_and_grad`` closure. The tracer escaped the trace boundary,
    leaving ``v.value`` as a ``LinearizeTracer`` between iterations
    and accumulating the autograd graph one iteration at a time.

    Direct symptom: ``type(v.value).__name__`` after a single grad call
    contains "Tracer". Indirect symptom: memory grows monotonically
    across iterations. The first is cheap and definitive; the second
    is expensive but catches the deeper OOM scenario."""

    def test_no_tracer_leaks_into_variable(self):
        """After ``compute_gradients`` returns, the Variable's value
        must be a concrete array, not a JAX tracer."""
        from braincoder.utils.backend import compute_gradients
        v = keras.Variable(np.asarray([3.0, 4.0], dtype=np.float32))

        def loss_fn():
            return ops.sum(v.value ** 2)

        loss, grads = compute_gradients(loss_fn, [v])
        type_name = type(v.value).__name__
        assert 'Tracer' not in type_name, (
            f"Variable.value is a leaked tracer ({type_name}) after "
            f"compute_gradients. backend={keras.backend.backend()!r}")
        # Gradient values are correct.
        g = np.asarray(ops.convert_to_numpy(grads[0]))
        np.testing.assert_allclose(g, [6.0, 8.0], rtol=1e-5)

    def test_repeated_grad_calls_stay_concrete(self):
        """Across 50 iterations, the Variable must keep concrete values
        and the gradient values must remain finite. This is the
        OOM-stand-in test from the audit: under the bug the autograd
        graph accumulates and JAX eventually fails, but well before
        that the Variable's value type stops being concrete."""
        from braincoder.utils.backend import compute_gradients
        v = keras.Variable(np.asarray([3.0, 4.0], dtype=np.float32))
        opt = keras.optimizers.Adam(learning_rate=0.01)

        def loss_fn():
            return ops.sum(v.value ** 2)

        for i in range(50):
            loss, grads = compute_gradients(loss_fn, [v])
            opt.apply_gradients(zip(grads, [v]))
            if i % 10 == 0:
                type_name = type(v.value).__name__
                assert 'Tracer' not in type_name, (
                    f"At iter {i}, Variable became a tracer "
                    f"({type_name}). backend={keras.backend.backend()!r}")
                assert np.all(np.isfinite(
                    np.asarray(ops.convert_to_numpy(v.value)))), (
                    f"Variable became non-finite at iter {i}")

    def test_parameter_fitter_jax_50_iter_integration(self):
        """End-to-end: fit a tiny GaussianPRF for 50 iterations.

        Under the JAX tracer-leak bug, the leaked tracer eventually
        causes ``jax.errors.UnexpectedTracerError`` (or an OOM on larger
        problems). The simplest check that catches this on CI without
        a GPU is "does the fit finish without crashing"."""
        from braincoder.models import GaussianPRF
        from braincoder.optimize import ParameterFitter
        import pandas as pd

        rng = np.random.default_rng(0)
        n_vox = 5
        paradigm = np.linspace(-5, 5, 40, dtype=np.float32)[:, np.newaxis]
        true_pars = pd.DataFrame({
            'mu':        np.linspace(-3, 3, n_vox, dtype=np.float32),
            'sd':        np.ones(n_vox, dtype=np.float32),
            'amplitude': np.ones(n_vox, dtype=np.float32),
            'baseline':  np.zeros(n_vox, dtype=np.float32),
        })
        model = GaussianPRF(paradigm=paradigm, parameters=true_pars)
        data = model.simulate(noise=0.1)

        fitter = ParameterFitter(model, data, paradigm)
        init_pars = pd.DataFrame({
            'mu':        np.zeros(n_vox, dtype=np.float32),
            'sd':        np.ones(n_vox, dtype=np.float32) * 2,
            'amplitude': np.ones(n_vox, dtype=np.float32),
            'baseline':  np.zeros(n_vox, dtype=np.float32),
        })
        estimated = fitter.fit(init_pars=init_pars,
                               max_n_iterations=50,
                               min_n_iterations=50,
                               progressbar=False)
        assert np.all(np.isfinite(estimated.values)), (
            f"ParameterFitter produced non-finite output on "
            f"backend={keras.backend.backend()!r}")


# ---------------------------------------------------------------------------
# Bonus: end-to-end ResidualFitter dof fit (cross-cuts Bugs 1 + 2 + 4).
# ---------------------------------------------------------------------------

class TestResidualFitterStudentT:
    """Full ResidualFitter run with ``method='t'``. Touches:

      * safe_cholesky (Omega may dip slightly below PSD during fitting)
      * mvt_log_prob -> _lgamma (must be differentiable in dof)
      * compute_gradients (no tracer leak across iterations)

    A pass here means all three live wiring sites work end-to-end."""

    def test_residual_fitter_t_method_finishes(self):
        """Tiny synthetic dataset; just verify ``method='t'`` finishes
        and returns a finite dof. Under the broken ``_lgamma``, the
        JAX run crashes with ``ConcretizationTypeError`` and the TF/torch
        runs return a biased dof or fail to update at all."""
        from braincoder.models import GaussianPRF
        from braincoder.optimize import ResidualFitter
        import pandas as pd

        rng = np.random.default_rng(0)
        n_vox = 4
        paradigm = np.linspace(-3, 3, 60, dtype=np.float32)[:, np.newaxis]
        params = pd.DataFrame({
            'mu':        np.linspace(-2, 2, n_vox, dtype=np.float32),
            'sd':        np.ones(n_vox, dtype=np.float32),
            'amplitude': np.ones(n_vox, dtype=np.float32),
            'baseline':  np.zeros(n_vox, dtype=np.float32),
        })
        model = GaussianPRF(paradigm=paradigm, parameters=params)
        data = model.simulate(noise=0.3)
        fitter = ResidualFitter(model, data, paradigm=paradigm,
                                parameters=params)
        omega, dof = fitter.fit(method='t', max_n_iterations=30,
                                min_n_iterations=10,
                                use_wwt=False,
                                progressbar=False)
        assert np.all(np.isfinite(omega)), "Omega contains NaN/Inf"
        assert dof is not None and np.isfinite(dof) and dof > 0, (
            f"dof is invalid: {dof}")
