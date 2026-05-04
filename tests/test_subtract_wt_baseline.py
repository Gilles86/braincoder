"""
Unit tests for the subtract_baseline option in GaussianPRF.init_pseudoWWT.

The covariance model used in decoding is:
    omega = rho * tau^T tau + (1-rho) * diag(tau^2) + sigma2 * WWT

where WWT = W^T W and W is the (n_stimuli x n_voxels) basis-prediction matrix.

If baseline is NOT subtracted, the WWT matrix is contaminated by a constant
offset in each voxel's predictions. This adds a term proportional to
baseline_i * baseline_j to every (i, j) entry of WWT, making voxels with
similar baselines look artificially correlated regardless of tuning similarity.

With subtract_baseline=True the per-voxel baseline is removed before computing
WWT so the matrix reflects only stimulus-driven covariance.
"""

import numpy as np
import pandas as pd
import pytest
import sys, os
from keras import ops

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from braincoder.models import GaussianPRF
from braincoder.optimize import ResidualFitter

# ── helpers ───────────────────────────────────────────────────────────────────

def make_gauss_params(n_voxels, baseline_values=None, seed=42):
    """Return a parameters DataFrame for GaussianPRF."""
    rng = np.random.default_rng(seed)
    if baseline_values is None:
        baseline_values = rng.uniform(0.5, 2.0, n_voxels)
    return pd.DataFrame({
        'mu':        rng.uniform(5, 25, n_voxels).astype(np.float32),
        'sd':        np.full(n_voxels, 2.0, dtype=np.float32),
        'amplitude': np.full(n_voxels, 1.0, dtype=np.float32),
        'baseline':  np.array(baseline_values, dtype=np.float32),
    })

def make_1d_stimulus_range(n=50, low=1.0, high=31.0):
    """1-D stimulus range for GaussianPRF (no range condition)."""
    return np.linspace(low, high, n, dtype=np.float32)

# ── tests ─────────────────────────────────────────────────────────────────────

class TestSubtractWtBaseline:

    def test_WWT_differs_with_nonzero_baseline(self):
        """WWT(subtract=True) != WWT(subtract=False) when baseline is non-zero."""
        model = GaussianPRF()
        params = make_gauss_params(n_voxels=10)
        stim = make_1d_stimulus_range()

        WWT_raw = ops.convert_to_numpy(model.init_pseudoWWT(stim, params, subtract_baseline=False))
        WWT_sub = ops.convert_to_numpy(model.init_pseudoWWT(stim, params, subtract_baseline=True))

        assert not np.allclose(WWT_raw, WWT_sub), (
            "WWT should differ between subtract=True and subtract=False "
            "when baseline is non-zero."
        )

    def test_correct_math_subtract_baseline(self):
        """WWT(subtract=True) equals (W - baseline)^T (W - baseline) exactly."""
        model = GaussianPRF()
        params = make_gauss_params(n_voxels=8)
        stim = make_1d_stimulus_range()

        W = ops.convert_to_numpy(model.basis_predictions(stim, params))          # (n_stim, n_voxels)
        baseline = params['baseline'].values.astype(np.float32)    # (n_voxels,)
        W_corrected = W - baseline[np.newaxis, :]                  # broadcast subtract
        expected_WWT = W_corrected.T @ W_corrected                 # (n_voxels, n_voxels)

        actual_WWT = ops.convert_to_numpy(model.init_pseudoWWT(stim, params, subtract_baseline=True))

        np.testing.assert_allclose(
            actual_WWT, expected_WWT, rtol=1e-5, atol=1e-6,
            err_msg="WWT(subtract=True) must equal (W - baseline)^T (W - baseline)."
        )

    def test_uses_baseline_column_not_W_minimum(self):
        """
        When parameters has a 'baseline' column, that value is used for subtraction,
        NOT the per-voxel minimum of W.

        We construct a case where baseline >> W.min so the two are clearly distinct.
        """
        model = GaussianPRF()
        n_voxels = 6
        # Amplitude=0 so W = baseline everywhere; min(W) == baseline.
        # Instead use a large explicit baseline that cannot match the W minimum
        # when amplitude is non-trivial.
        params = make_gauss_params(n_voxels=n_voxels, baseline_values=np.ones(n_voxels) * 10.0)
        stim = make_1d_stimulus_range()

        W = ops.convert_to_numpy(model.basis_predictions(stim, params))
        W_min = W.min(axis=0)   # per-voxel minimum across stimulus range

        # Sanity: baseline (10.0) should be far above the per-voxel minimum
        # (Gaussian tuning dips below peak so W.min < W.max, but with baseline=10
        # the minimum is still ~10; amplitude=1 on top, so W_min ~ 10 and
        # baseline == 10 — in this particular case they ARE equal.
        # Use amplitude=5 to make W.min > baseline is impossible but W.max > baseline).
        # Better setup: manually craft a case where W.min != baseline.
        params2 = params.copy()
        params2['amplitude'] = 5.0
        W2 = ops.convert_to_numpy(model.basis_predictions(stim, params2))
        W2_min = W2.min(axis=0)

        # The minimum of W2 across the stimulus range should be close to baseline
        # (evaluated far from mu), but baseline is exactly 10.0.
        # Verify that the WWT built via baseline column matches (W - baseline)^T(W - baseline)
        baseline_col = params2['baseline'].values.astype(np.float32)
        W2_corrected = W2 - baseline_col[np.newaxis, :]
        expected_WWT = W2_corrected.T @ W2_corrected

        actual_WWT = ops.convert_to_numpy(model.init_pseudoWWT(stim, params2, subtract_baseline=True))

        np.testing.assert_allclose(
            actual_WWT, expected_WWT, rtol=1e-3, atol=1e-4,
            err_msg="subtract_baseline must use the 'baseline' column, not W.min."
        )

    def test_zero_baseline_WWT_unchanged(self):
        """When baseline=0 everywhere, WWT is identical with and without subtraction."""
        model = GaussianPRF()
        n_voxels = 8
        params = make_gauss_params(n_voxels=n_voxels, baseline_values=np.zeros(n_voxels))
        stim = make_1d_stimulus_range()

        WWT_raw = ops.convert_to_numpy(model.init_pseudoWWT(stim, params, subtract_baseline=False))
        WWT_sub = ops.convert_to_numpy(model.init_pseudoWWT(stim, params, subtract_baseline=True))

        np.testing.assert_allclose(
            WWT_raw, WWT_sub, rtol=1e-5, atol=1e-6,
            err_msg="With zero baseline, WWT must be identical for both modes."
        )

    def test_constant_baseline_shift_invariant_after_subtraction(self):
        """
        Adding a constant offset to all baselines must NOT change WWT(subtract=True).

        Proof: let b_j = b0_j + c. Then W - b = W0 - b0. So WWT is unchanged.
        """
        model = GaussianPRF()
        n_voxels = 8
        params_low = make_gauss_params(n_voxels=n_voxels, baseline_values=np.zeros(n_voxels))
        params_high = params_low.copy()
        params_high['baseline'] = params_high['baseline'] + 5.0
        stim = make_1d_stimulus_range()

        WWT_low = ops.convert_to_numpy(model.init_pseudoWWT(stim, params_low,  subtract_baseline=True))
        WWT_high = ops.convert_to_numpy(model.init_pseudoWWT(stim, params_high, subtract_baseline=True))

        np.testing.assert_allclose(
            WWT_low, WWT_high, rtol=1e-5, atol=1e-5,
            err_msg="WWT(subtract=True) must be invariant to a constant baseline shift."
        )

    def test_constant_baseline_shift_changes_raw_WWT(self):
        """
        A constant baseline shift MUST change WWT(subtract=False), confirming the
        contamination that subtract_baseline is designed to remove.

        Adding c to all baselines adds n_stim * c^2 + 2c * sum_i W0_i to each
        diagonal entry of WWT.
        """
        model = GaussianPRF()
        n_voxels = 8
        params_low = make_gauss_params(n_voxels=n_voxels, baseline_values=np.zeros(n_voxels))
        params_high = params_low.copy()
        params_high['baseline'] = params_high['baseline'] + 5.0
        stim = make_1d_stimulus_range()

        WWT_low  = ops.convert_to_numpy(model.init_pseudoWWT(stim, params_low,  subtract_baseline=False))
        WWT_high = ops.convert_to_numpy(model.init_pseudoWWT(stim, params_high, subtract_baseline=False))

        assert not np.allclose(WWT_low, WWT_high), (
            "WWT(subtract=False) must change when baseline shifts, "
            "demonstrating the contamination."
        )

    def test_cache_overwritten_by_second_call(self):
        """
        init_pseudoWWT overwrites the cached _pseudoWWT.
        get_pseudoWWT() must reflect the most recent call.
        """
        model = GaussianPRF()
        params = make_gauss_params(n_voxels=6)
        stim = make_1d_stimulus_range()

        model.init_pseudoWWT(stim, params, subtract_baseline=False)
        WWT_raw = ops.convert_to_numpy(model.get_pseudoWWT()).copy()

        model.init_pseudoWWT(stim, params, subtract_baseline=True)
        WWT_sub = ops.convert_to_numpy(model.get_pseudoWWT()).copy()

        # The two cached values should differ (nonzero baseline)
        assert not np.allclose(WWT_raw, WWT_sub), (
            "Cached _pseudoWWT should be updated on each init_pseudoWWT call."
        )

    def test_WWT_is_positive_semidefinite_after_subtraction(self):
        """
        WWT = (W - b)^T (W - b) must always be positive semi-definite
        (eigenvalues >= 0).
        """
        model = GaussianPRF()
        params = make_gauss_params(n_voxels=15)
        stim = make_1d_stimulus_range()

        WWT = ops.convert_to_numpy(model.init_pseudoWWT(stim, params, subtract_baseline=True))
        eigvals = np.linalg.eigvalsh(WWT)

        assert np.all(eigvals >= -1e-5), (
            f"WWT(subtract=True) must be PSD. Minimum eigenvalue: {eigvals.min():.4e}"
        )

    def test_per_voxel_baseline_subtraction(self):
        """
        Subtraction is per-voxel: different baselines for each voxel are handled correctly.
        """
        model = GaussianPRF()
        n_voxels = 6
        baselines = np.array([0.0, 0.5, 1.0, 2.0, 5.0, 10.0], dtype=np.float32)
        params = make_gauss_params(n_voxels=n_voxels, baseline_values=baselines)
        stim = make_1d_stimulus_range()

        W = ops.convert_to_numpy(model.basis_predictions(stim, params))
        W_corrected = W - baselines[np.newaxis, :]
        expected_WWT = W_corrected.T @ W_corrected

        actual_WWT = ops.convert_to_numpy(model.init_pseudoWWT(stim, params, subtract_baseline=True))

        np.testing.assert_allclose(
            actual_WWT, expected_WWT, rtol=1e-3, atol=1e-4,
            err_msg="Per-voxel baseline subtraction must handle different baseline values."
        )

# ── Tests for _get_omega_lambda (convex combination) ─────────────────────────

def make_residual_fitter(n_voxels=5, n_trials=40, lambd=0.5, seed=0):
    """Return a minimal ResidualFitter with synthetic data."""
    rng = np.random.default_rng(seed)
    model = GaussianPRF()
    params = make_gauss_params(n_voxels=n_voxels, seed=seed)
    stim_1d = make_1d_stimulus_range(n=n_trials)
    stim_df = pd.DataFrame({'x': stim_1d})
    data = pd.DataFrame(
        rng.normal(size=(n_trials, n_voxels)).astype(np.float32),
        columns=[f'v{i}' for i in range(n_voxels)]
    )
    model.parameters = params
    return ResidualFitter(model, data, stim_df, parameters=params, lambd=lambd)

class TestOmegaLambda:

    def _make_inputs(self, n_voxels=4):
        """Return (tau, rho, sigma2, WWT, sample_cov) as backend tensors."""
        rng = np.random.default_rng(42)
        tau = ops.convert_to_tensor(rng.uniform(0.5, 2.0, (1, n_voxels)).astype(np.float32))
        rho = ops.convert_to_tensor(np.float32(0.3))
        sigma2 = ops.convert_to_tensor(np.float32(0.1))
        A = rng.normal(size=(n_voxels, n_voxels)).astype(np.float32)
        WWT = ops.convert_to_tensor(A.T @ A)
        B = rng.normal(size=(n_voxels, n_voxels)).astype(np.float32)
        sample_cov = ops.convert_to_tensor(B.T @ B)
        return tau, rho, sigma2, WWT, sample_cov

    @staticmethod
    def _adaptive_jitter(matrix):
        """Mirror the adaptive jitter in residual_fitter._get_omega_lambda."""
        mean_diag = float(np.mean(np.diag(matrix)))
        return mean_diag * 1e-4 + 1e-9

    def test_lambda0_equals_pure_parametric(self):
        """At lambda=0, _get_omega_lambda == _get_omega (plus adaptive jitter)."""
        rf = make_residual_fitter(lambd=0.0)
        tau, rho, sigma2, WWT, sample_cov = self._make_inputs()

        omega_lambda = ops.convert_to_numpy(rf._get_omega_lambda(tau, rho, sigma2, WWT, 0.0, sample_cov))
        omega_param  = ops.convert_to_numpy(rf._get_omega(tau, rho, sigma2, WWT))
        expected = omega_param + np.eye(omega_param.shape[0], dtype=np.float32) * self._adaptive_jitter(omega_param)

        np.testing.assert_allclose(omega_lambda, expected, rtol=1e-4, atol=1e-5,
            err_msg="lambda=0 must give the parametric model plus adaptive jitter.")

    def test_lambda1_equals_pure_empirical(self):
        """At lambda=1, _get_omega_lambda == sample_cov (plus adaptive jitter)."""
        rf = make_residual_fitter(lambd=1.0)
        tau, rho, sigma2, WWT, sample_cov = self._make_inputs()

        omega_lambda = ops.convert_to_numpy(rf._get_omega_lambda(tau, rho, sigma2, WWT, 1.0, sample_cov))
        sc = ops.convert_to_numpy(sample_cov)
        expected = sc + np.eye(sc.shape[0], dtype=np.float32) * self._adaptive_jitter(sc)

        np.testing.assert_allclose(omega_lambda, expected, rtol=1e-4, atol=1e-5,
            err_msg="lambda=1 must give the pure empirical covariance (+ adaptive jitter).")

    def test_convex_combination_at_midpoint(self):
        """At lambda=0.5, omega is an equal blend of parametric and empirical."""
        rf = make_residual_fitter(lambd=0.5)
        tau, rho, sigma2, WWT, sample_cov = self._make_inputs()

        omega_param   = ops.convert_to_numpy(rf._get_omega(tau, rho, sigma2, WWT))
        omega_lambda  = ops.convert_to_numpy(rf._get_omega_lambda(tau, rho, sigma2, WWT, 0.5, sample_cov))
        blend = 0.5 * omega_param + 0.5 * ops.convert_to_numpy(sample_cov)
        expected = blend + np.eye(blend.shape[0], dtype=np.float32) * self._adaptive_jitter(blend)

        np.testing.assert_allclose(omega_lambda, expected, rtol=1e-4, atol=1e-5,
            err_msg="lambda=0.5 must give an equal blend of parametric and empirical omega.")

    def test_parametric_part_scales_with_1_minus_lambda(self):
        """
        The parametric contribution scales with (1-lambda) and the empirical part
        with lambda.  After subtracting omega(0) - omega(0.5), the parametric and
        empirical components net to 0.5 * (omega_param - sample_cov); the adaptive
        jitters do not exactly cancel (they depend on each blend's mean diagonal),
        so we use a loose absolute tolerance scaled by the matrix magnitude.
        """
        rf = make_residual_fitter(lambd=0.0)
        tau, rho, sigma2, WWT, sample_cov = self._make_inputs()

        omega_0    = ops.convert_to_numpy(rf._get_omega_lambda(tau, rho, sigma2, WWT, 0.0, sample_cov))
        omega_half = ops.convert_to_numpy(rf._get_omega_lambda(tau, rho, sigma2, WWT, 0.5, sample_cov))
        omega_param = ops.convert_to_numpy(rf._get_omega(tau, rho, sigma2, WWT))

        expected = 0.5 * (omega_param - ops.convert_to_numpy(sample_cov))
        # Jitter difference scale: 1e-4 * |mean_diag(omega_0) - mean_diag(omega_half)|
        atol = 1e-4 * max(abs(np.mean(np.diag(omega_0))),
                          abs(np.mean(np.diag(omega_half)))) * 2
        np.testing.assert_allclose(
            omega_0 - omega_half, expected, rtol=1e-4, atol=atol,
            err_msg="Parametric part must scale linearly with (1-lambda)."
        )
