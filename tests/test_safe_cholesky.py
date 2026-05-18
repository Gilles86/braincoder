"""Regression tests for ``braincoder.utils.backend.safe_cholesky``.

The fix this exercises: plain ``ops.cholesky`` returns NaN (and raises
on the matplotlib-aware Keras backend) when the input matrix has
slipped slightly below the PSD boundary — a routine outcome of the
residual fitter pushing α/β/ρ to numerical edges. The model's
``get_stimulus_pdf`` decoder then NaN-crashes downstream.

``safe_cholesky`` symmetrises the input and adds adaptive jitter
``(rel_jitter * mean(diag) + 1e-9) * I`` before factorising. These
tests pin down the contract:

  1. On a clearly-PSD matrix it agrees with ``numpy.linalg.cholesky``
     to ~jitter accuracy.
  2. On a matrix with one slightly-negative eigenvalue (the sub-20
     failure mode) it succeeds and the resulting factor times its
     transpose recovers the input up to the jitter perturbation.
  3. The output never contains NaN, even when the input is
     numerically nasty.
  4. The fix is wired into ``EncodingModel.get_stimulus_pdf`` so
     callers don't have to remember to call safe_cholesky themselves.
"""
import numpy as np
import pytest
from keras import ops

from braincoder.utils.backend import safe_cholesky


# Random-matrix helpers --------------------------------------------------

def _random_psd(n, eigvals=None, seed=0):
    """Build an n×n symmetric PSD matrix with the requested eigenvalues."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    Q, _ = np.linalg.qr(A)
    if eigvals is None:
        eigvals = np.linspace(0.1, 2.0, n)
    return (Q * eigvals) @ Q.T


def _make_borderline_omega(n=8, neg_eig=-1e-8, seed=0):
    """One eigenvalue just below zero — plain cholesky will NaN on this."""
    eigvals = np.linspace(0.01, 1.5, n)
    eigvals[0] = neg_eig
    return _random_psd(n, eigvals=eigvals, seed=seed)


# Tests ------------------------------------------------------------------

class TestSafeCholeskyContract:
    """Behavioural contract: safe_cholesky succeeds where plain Cholesky
    fails, and stays close to the true factorisation when the input is
    already PSD."""

    def test_clean_psd_matches_numpy(self):
        """On a healthy PSD matrix, safe_cholesky agrees with numpy's
        Cholesky to within the jitter perturbation."""
        M = _random_psd(n=10, eigvals=np.linspace(0.5, 5.0, 10)).astype(np.float32)
        L_ref = np.linalg.cholesky(M).astype(np.float32)
        L = np.asarray(safe_cholesky(M))
        # Same shape and lower-triangular
        assert L.shape == L_ref.shape
        assert np.allclose(L, np.tril(L), atol=1e-6)
        # Reconstruction within jitter scale (default 1e-4 × mean(diag))
        recon = L @ L.T
        # Adaptive jitter on a matrix with mean(diag) ~ 2.75: ~2.75e-4
        # absolute. Use a loose tolerance.
        assert np.allclose(recon, M, atol=5e-3), (
            f"max abs diff = {np.max(np.abs(recon - M))}")

    def test_borderline_negative_eigenvalue_does_not_nan(self):
        """The sub-20 failure mode: one eigenvalue just below zero. Plain
        cholesky returns NaN; safe_cholesky must succeed."""
        M = _make_borderline_omega(n=12, neg_eig=-1e-6).astype(np.float32)
        # Sanity: numpy cholesky raises on this
        with pytest.raises(np.linalg.LinAlgError):
            np.linalg.cholesky(M)
        # safe_cholesky should succeed without NaNs
        L = np.asarray(safe_cholesky(M))
        assert L.shape == (12, 12)
        assert np.all(np.isfinite(L)), (
            "safe_cholesky returned NaN/inf on a borderline-PSD matrix")
        # Recovered matrix is close to M, up to the jitter perturbation
        recon = L @ L.T
        # The negative eigenvalue is at the limit of float32; jitter
        # shifts the eigenspectrum by mean(diag) * 1e-4 ≈ 7e-5 absolute.
        assert np.allclose(recon, M, atol=5e-3)

    def test_moderately_negative_eigenvalue_still_works(self):
        """More aggressive: one eigenvalue at −1e-3 (still small relative to
        the matrix scale). Adaptive jitter should still rescue this."""
        M = _make_borderline_omega(n=15, neg_eig=-1e-3).astype(np.float32)
        L = np.asarray(safe_cholesky(M))
        assert np.all(np.isfinite(L)), (
            "safe_cholesky NaN'd on a matrix with eigvalue=-1e-3")

    def test_output_lower_triangular(self):
        """Cholesky factor must be lower-triangular."""
        M = _random_psd(n=8, eigvals=np.linspace(0.2, 3.0, 8)).astype(np.float32)
        L = np.asarray(safe_cholesky(M))
        upper_off_diag = np.triu(L, k=1)
        assert np.allclose(upper_off_diag, 0.0, atol=1e-6), (
            "Output is not lower-triangular")

    def test_symmetrises_asymmetric_input(self):
        """A slightly-asymmetric input (e.g. from rounding) should still
        factorise: safe_cholesky symmetrises by averaging M with M.T."""
        M = _random_psd(n=10, eigvals=np.linspace(0.5, 2.0, 10)).astype(np.float32)
        M_asym = M + 1e-5 * np.random.default_rng(1).standard_normal(M.shape).astype(np.float32)
        L = np.asarray(safe_cholesky(M_asym))
        assert np.all(np.isfinite(L))

    def test_dtype_preserved(self):
        """Output dtype matches input dtype (no silent float32 → float64)."""
        for dtype in (np.float32, np.float64):
            M = _random_psd(n=6, eigvals=np.linspace(0.5, 2.0, 6)).astype(dtype)
            L = np.asarray(safe_cholesky(M))
            assert L.dtype == dtype, (
                f"Expected {dtype}, got {L.dtype}")

    def test_jitter_parameter_scales_correctly(self):
        """Larger `jitter` → larger perturbation. Sanity-check the kwarg."""
        M = _make_borderline_omega(n=10, neg_eig=-1e-4).astype(np.float32)
        L_default = np.asarray(safe_cholesky(M))                  # jitter=1e-4
        L_large = np.asarray(safe_cholesky(M, jitter=1e-2))
        recon_default = L_default @ L_default.T
        recon_large   = L_large @ L_large.T
        # Larger jitter inflates the diagonal more
        added_default = np.diag(recon_default - M).mean()
        added_large   = np.diag(recon_large - M).mean()
        assert added_large > added_default, (
            f"larger jitter should inflate diagonal more "
            f"({added_default} vs {added_large})")


class TestSafeCholeskyIntegration:
    """Wire-up regression tests: safe_cholesky is plumbed into the right
    call sites so callers don't have to remember to use it themselves."""

    def test_used_in_get_stimulus_pdf(self):
        """The decoder path that crashed on sub-20 (Cholesky NaN inside
        ``EncodingModel.get_stimulus_pdf``) must now route through
        safe_cholesky. Regression test for that wiring."""
        # Tiny end-to-end: build a GaussianPRF, supply a borderline-PSD
        # omega, ask for a stimulus posterior, expect no NaN crash.
        from braincoder.models import GaussianPRF
        import pandas as pd

        rng = np.random.default_rng(42)
        n_vox = 6
        n_trials = 8
        stim_grid = np.linspace(0, 10, 21)
        # Build a small valid model + data
        params = pd.DataFrame({
            "mu": rng.uniform(2, 8, n_vox),
            "sd": rng.uniform(0.5, 1.5, n_vox),
            "amplitude": np.ones(n_vox),
            "baseline": np.zeros(n_vox),
        })
        model = GaussianPRF(parameters=params)
        paradigm = pd.Series(rng.uniform(0, 10, n_trials), name="x")
        data = model.simulate(paradigm=paradigm, noise=0.5)
        # Borderline-PSD omega — would NaN under plain cholesky
        omega = _make_borderline_omega(n=n_vox, neg_eig=-1e-6).astype(np.float32)
        # Call the path that previously crashed
        pdf = model.get_stimulus_pdf(data, omega=omega,
                                       stimulus_range=stim_grid,
                                       normalize=False)
        # No NaNs, finite output, right shape
        arr = np.asarray(pdf.values)
        assert arr.shape[0] == n_trials
        assert np.all(np.isfinite(arr)), (
            "get_stimulus_pdf returned NaN — safe_cholesky wiring regressed")
