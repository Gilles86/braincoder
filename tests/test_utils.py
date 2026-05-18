"""Tests for braincoder utility functions (formatting, math, stats)."""
import numpy as np
from keras import ops
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# format_paradigm
# ---------------------------------------------------------------------------

class TestFormatParadigm:

    def test_none_returns_none(self):
        from braincoder.utils.formatting import format_paradigm
        assert format_paradigm(None) is None

    def test_dataframe_passthrough(self):
        from braincoder.utils.formatting import format_paradigm
        df = pd.DataFrame({'a': [1.0, 2.0]})
        result = format_paradigm(df)
        assert result is df

    def test_series_to_frame(self):
        from braincoder.utils.formatting import format_paradigm
        s = pd.Series([1.0, 2.0, 3.0])
        result = format_paradigm(s)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (3, 1)

    def test_1d_array_to_column(self):
        from braincoder.utils.formatting import format_paradigm
        arr = np.array([1.0, 2.0, 3.0])
        result = format_paradigm(arr)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (3, 1)

    def test_2d_array_preserved(self):
        from braincoder.utils.formatting import format_paradigm
        arr = np.ones((10, 3))
        result = format_paradigm(arr)
        assert result.shape == (10, 3)

    def test_3d_array_flattened(self):
        from braincoder.utils.formatting import format_paradigm
        arr = np.ones((10, 4, 4))
        result = format_paradigm(arr)
        assert result.shape == (10, 16)

    def test_dtype_is_float32(self):
        from braincoder.utils.formatting import format_paradigm
        arr = np.ones((5, 2), dtype=np.float64)
        result = format_paradigm(arr)
        assert result.dtypes.iloc[0] == np.float32

# ---------------------------------------------------------------------------
# format_parameters
# ---------------------------------------------------------------------------

class TestFormatParameters:

    def test_none_returns_none(self):
        from braincoder.utils.formatting import format_parameters
        assert format_parameters(None) is None

    def test_dataframe_passthrough_float32(self):
        from braincoder.utils.formatting import format_parameters
        df = pd.DataFrame({'x': [1.0], 'y': [2.0]})
        result = format_parameters(df)
        assert isinstance(result, pd.DataFrame)
        assert result.dtypes.iloc[0] == np.float32

    def test_array_with_labels(self):
        from braincoder.utils.formatting import format_parameters
        arr = np.ones((3, 2), dtype=np.float32)
        result = format_parameters(arr, parameter_labels=['a', 'b'])
        assert list(result.columns) == ['a', 'b']
        assert result.shape == (3, 2)

    def test_array_without_labels_uses_default(self):
        from braincoder.utils.formatting import format_parameters
        arr = np.ones((2, 3), dtype=np.float32)
        result = format_parameters(arr)
        assert list(result.columns) == ['par1', 'par2', 'par3']

# ---------------------------------------------------------------------------
# format_data
# ---------------------------------------------------------------------------

class TestFormatData:

    def test_dataframe_passthrough(self):
        from braincoder.utils.formatting import format_data
        df = pd.DataFrame(np.ones((5, 3)))
        result = format_data(df)
        assert result is df

    def test_array_to_dataframe(self):
        from braincoder.utils.formatting import format_data
        arr = np.ones((10, 4), dtype=np.float32)
        result = format_data(arr)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (10, 4)

    def test_dtype_is_float32(self):
        from braincoder.utils.formatting import format_data
        arr = np.ones((5, 2), dtype=np.float64)
        result = format_data(arr)
        assert result.dtypes.iloc[0] == np.float32

    def test_index_name_is_time(self):
        from braincoder.utils.formatting import format_data
        arr = np.ones((5, 2), dtype=np.float32)
        result = format_data(arr)
        assert result.index.name == 'time'

# ---------------------------------------------------------------------------
# format_weights
# ---------------------------------------------------------------------------

class TestFormatWeights:

    def test_none_returns_none(self):
        from braincoder.utils.formatting import format_weights
        assert format_weights(None) is None

    def test_dataframe_passthrough(self):
        from braincoder.utils.formatting import format_weights
        df = pd.DataFrame(np.ones((3, 4)))
        result = format_weights(df)
        assert result is df

    def test_array_to_dataframe(self):
        from braincoder.utils.formatting import format_weights
        arr = np.ones((3, 4), dtype=np.float32)
        result = format_weights(arr)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (3, 4)

    def test_index_name_is_population(self):
        from braincoder.utils.formatting import format_weights
        arr = np.ones((3, 4), dtype=np.float32)
        result = format_weights(arr)
        assert result.index.name == 'population'

# ---------------------------------------------------------------------------
# gamma_pdf (math utility in hrf.py)
# ---------------------------------------------------------------------------

class TestGammaPdf:

    def test_output_shape(self):
        from braincoder.hrf import gamma_pdf
        t = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
        result = ops.convert_to_numpy(gamma_pdf(t, a=6.0, d=1.0))
        assert result.shape == (3, 1)

    def test_positive_values(self):
        from braincoder.hrf import gamma_pdf
        t = np.linspace(0.1, 20.0, 100, dtype=np.float32)[:, np.newaxis]
        result = ops.convert_to_numpy(gamma_pdf(t, a=6.0, d=1.0))
        assert np.all(result >= 0), "Gamma PDF values should be non-negative"

    def test_peak_near_mode(self):
        """Mode of Gamma(a, d) is (a-1)*d."""
        from braincoder.hrf import gamma_pdf
        a, d = 6.0, 1.0
        t = np.linspace(0.1, 20.0, 1000, dtype=np.float32)[:, np.newaxis]
        result = ops.convert_to_numpy(gamma_pdf(t, a=a, d=d))
        peak_t = t.flatten()[np.argmax(result)]
        expected_mode = (a - 1) * d  # = 5.0
        assert abs(peak_t - expected_mode) < 0.5, \
            f"Peak at {peak_t:.2f}, expected ~{expected_mode:.1f}"


# ---------------------------------------------------------------------------
# R² mixture posterior / p_signal threshold
# ---------------------------------------------------------------------------

class TestR2Posterior:
    """The p_signal>0.5 path is now the default voxel-selection rule
    in fit_gp_prior.py. Locks in: (i) p_signal aligns with mixture
    component (high R² → signal); (ii) threshold sits between the
    two component means; (iii) p_signal is monotone in R²."""

    def _make_fit(self, rng, n=2000, w_signal=0.3,
                  noise_mu=-3.0, signal_mu=0.5,
                  noise_sigma=0.6, signal_sigma=0.5):
        """Sample logit-Gaussian mixture, fit it, return (r2, fit)."""
        from braincoder.utils.stats import _inv_logit, fit_r2_mixture
        n_signal = rng.binomial(n, w_signal)
        z = np.concatenate([
            rng.normal(noise_mu,  noise_sigma, n - n_signal),
            rng.normal(signal_mu, signal_sigma, n_signal),
        ])
        r2 = _inv_logit(z)
        return r2, fit_r2_mixture(r2)

    def test_p_signal_is_monotone_in_r2(self):
        from braincoder.utils.stats import r2_posterior_signal
        rng = np.random.default_rng(0)
        r2, fit = self._make_fit(rng)
        grid = np.linspace(0.01, 0.95, 100)
        p = r2_posterior_signal(grid, fit)
        assert np.all(np.diff(p) >= -1e-9), \
            "p_signal must be monotone non-decreasing in R²"

    def test_p_signal_outside_unit_interval_is_zero(self):
        from braincoder.utils.stats import r2_posterior_signal
        rng = np.random.default_rng(1)
        _, fit = self._make_fit(rng)
        r2 = np.array([-0.1, 0.0, 1.0, 1.5, np.nan, np.inf])
        p = r2_posterior_signal(r2, fit)
        assert np.all(p == 0.0)

    def test_threshold_lies_between_component_means(self):
        from braincoder.utils.stats import (
            _inv_logit, r2_p_signal_threshold)
        rng = np.random.default_rng(2)
        _, fit = self._make_fit(rng)
        t = r2_p_signal_threshold(fit, p=0.5)
        # Threshold should be above the noise mean R² and below the
        # signal mean R² for a non-degenerate mixture.
        assert _inv_logit(fit['noise_mu']) < t < _inv_logit(fit['signal_mu'])

    def test_threshold_consistent_with_posterior(self):
        """Voxels with r² > t* should have p_signal ≥ 0.5, and vice versa."""
        from braincoder.utils.stats import (
            r2_posterior_signal, r2_p_signal_threshold)
        rng = np.random.default_rng(3)
        r2, fit = self._make_fit(rng)
        t = r2_p_signal_threshold(fit, p=0.5)
        p = r2_posterior_signal(r2, fit)
        # Tiny grid-quantization tolerance.
        above = r2 > t + 1e-3
        below = r2 < t - 1e-3
        assert (p[above] >= 0.5 - 1e-2).all()
        assert (p[below] <  0.5 + 1e-2).all()


# ---------------------------------------------------------------------------
# safe_cholesky — robust factorisation of near-PSD covariance matrices
# ---------------------------------------------------------------------------

class TestSafeCholesky:
    """Guard against the Cholesky-on-non-PSD failure that crashes
    ``get_stimulus_pdf`` when the residual fitter drives Ω to the edge
    of the PSD cone (α negative, β tiny, ρ saturated → smallest eigenvalue
    just below zero from numerical drift)."""

    @staticmethod
    def _make_near_psd(n=8, neg_eig=-1e-6, seed=0):
        """Build a symmetric matrix whose smallest eigenvalue is ``neg_eig``.

        Plain ``np.linalg.cholesky`` should fail on this; ``safe_cholesky``
        should succeed.
        """
        rng = np.random.default_rng(seed)
        # Random orthogonal basis.
        A = rng.standard_normal((n, n))
        Q, _ = np.linalg.qr(A)
        # Diagonal of moderate positive eigenvalues plus one slightly
        # negative one to simulate numerical drift.
        eigs = rng.uniform(0.1, 2.0, size=n).astype(np.float64)
        eigs[0] = neg_eig
        M = (Q * eigs) @ Q.T
        # Re-symmetrise to kill rounding noise in the construction.
        return 0.5 * (M + M.T)

    def test_plain_cholesky_fails_on_near_psd(self):
        """Sanity: the bare ``ops.cholesky`` does fail on this input.

        Depending on backend, Cholesky either raises ``ValueError``
        (TF wraps "Cholesky decomposition failed") or silently returns
        NaN (JAX / pure numpy). Both count as "fails" — we just need
        the fixture to actually be ill-conditioned enough to trigger
        one of the two."""
        from keras import ops as _ops
        M = self._make_near_psd().astype(np.float32)
        try:
            L = _ops.cholesky(_ops.convert_to_tensor(M))
        except (ValueError, RuntimeError, Exception) as exc:
            # Backends that raise (TF) — this is the bug being patched.
            assert 'holesky' in str(exc) or 'decomp' in str(exc).lower(), (
                f"Unexpected exception type from plain cholesky: {exc!r}")
            return
        # Backends that propagate NaN (JAX / scipy) — also a failure.
        L_np = np.asarray(_ops.convert_to_numpy(L))
        assert np.isnan(L_np).any(), (
            "Test fixture is not near-PSD enough to make plain "
            "Cholesky fail; tighten ``neg_eig``.")

    def test_safe_cholesky_succeeds_on_near_psd(self):
        """``safe_cholesky`` should produce a finite lower-triangular factor."""
        from braincoder.utils.backend import safe_cholesky
        M = self._make_near_psd().astype(np.float32)
        L = safe_cholesky(M)
        L_np = np.asarray(ops.convert_to_numpy(L))
        assert np.isfinite(L_np).all()
        # Lower-triangular: strict upper triangle should be ~0.
        upper = np.triu(L_np, k=1)
        assert np.allclose(upper, 0.0, atol=1e-5)
        # And the diagonal must be strictly positive.
        assert (np.diag(L_np) > 0).all()

    def test_safe_cholesky_reconstructs_psd_input(self):
        """For a well-conditioned PSD matrix, ``L Lᵀ ≈ M`` up to jitter."""
        from braincoder.utils.backend import safe_cholesky
        rng = np.random.default_rng(7)
        A = rng.standard_normal((6, 6)).astype(np.float32)
        M = (A @ A.T + np.eye(6, dtype=np.float32))  # comfortably PSD
        L = safe_cholesky(M, jitter=1e-6)
        L_np = np.asarray(ops.convert_to_numpy(L))
        M_rec = L_np @ L_np.T
        # Jitter is ~ 1e-6 * mean(diag(M)); reconstruction is correspondingly close.
        np.testing.assert_allclose(M_rec, M, atol=5e-4)

    def test_safe_cholesky_on_residual_fitter_omega(self):
        """End-to-end: build a real residual-fitter-style Ω whose Adam-driven
        parameters have wandered off the PSD edge, and verify safe_cholesky
        recovers."""
        from braincoder.utils.backend import safe_cholesky
        rng = np.random.default_rng(2)
        n = 12
        tau = rng.uniform(0.5, 1.5, size=(1, n)).astype(np.float32)
        # WWᵀ = some PSD basis-overlap matrix.
        W = rng.standard_normal((4, n)).astype(np.float32)
        WWT = W.T @ W
        # α just below zero, β small — exactly the pathology that bit on the cluster.
        alpha, beta, rho, sigma2 = -1e-4, 1e-3, 0.95, 1e-3
        D = np.abs(rng.standard_normal((n, n))).astype(np.float32)
        D = 0.5 * (D + D.T); np.fill_diagonal(D, 0.0)
        tt = tau.T @ tau
        omega = (rho * (alpha * np.exp(-beta * D) * tt + (1 - alpha) * tt)
                 + (1 - rho) * np.diag(np.squeeze(tau ** 2))
                 + sigma2 * WWT)
        omega = 0.5 * (omega + omega.T)
        # Plain cholesky on this should NaN (or be unstable).
        L = safe_cholesky(omega.astype(np.float32))
        L_np = np.asarray(ops.convert_to_numpy(L))
        assert np.isfinite(L_np).all()
